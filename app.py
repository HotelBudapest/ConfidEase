from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify, request
import uuid
from extractor import extract_phrases
from wordcloud import WordCloud
from transformers import pipeline
from threading import Thread
import time, datetime, urllib.parse, feedparser
from news_fetcher import (
    get_news_for_industry,
    get_available_industries,
    get_cached_news,
    extract_image,
    clean_html,
    get_entry_date,
    CACHE_DURATION,
)
from linkedin_search import fetch_company_people
from flask_session import Session
from collections import Counter
import io
import html
import json
import matplotlib.pyplot as plt
from collections import Counter
import hashlib
import os
import tempfile
from resume_matcher import ResumeJobMatcher
import pdfplumber
import re
from markupsafe import Markup

app = Flask(__name__)
# tell Flask-Session to store session data in server-side files
app.secret_key = 'your_secret_key_here' 
app.config.update({
    "SESSION_PERMANENT": False,      # Sessions aren’t permanent by default
    "SESSION_TYPE": "filesystem",    # other options: "redis", "mongodb", etc.
    "SESSION_FILE_DIR": "./.flask_session",  
    "SESSION_FILE_THRESHOLD": 1000,  # how many files before cleanup kicks in
})
Session(app)
summarizer = pipeline("text2text-generation", model="google/flan-t5-large")

summary_cache = {}
_resume_store = {}
COMPANY_NEWS_CACHE = {}
PEOPLE_CACHE = {}
SUGGESTION_CACHE = {}  # { job_id: { phrase: {category,suggestion} } }

def llm_categorize(phrase):
    prompt = (
        f"Categorize this keyword into exactly one of: Technical, Interpersonal, Academic.\n"
        f"Keyword: “{phrase}”\n"
        f"Answer with exactly one word."
    )
    resp = summarizer(
        prompt,
        max_length=16,
        do_sample=False   # deterministic is OK for a tiny classification
    )[0]["generated_text"].strip()
    for cat in ("Technical", "Interpersonal", "Academic"):
        if cat.lower() in resp.lower():
            return cat
    return "Technical"

def llm_suggest(phrase, category, resume_text):
    prompt = (
        f"You are a career coach.  Given the **resume excerpt** below and the **skill** “{phrase}” "
        f"(category: {category}), write **one concise bullet point** showing how to weave that skill "
        f"into the candidate’s current role.  **Do not** repeat the resume; output only the new bullet.\n\n"
        f"Resume excerpt:\n"
        f"{resume_text[:300]}\n\n"
        f"Skill: {phrase}\n"
    )
    return summarizer(
        prompt,
        max_length=120,
        do_sample=False,
        temperature=0.3
    )[0]["generated_text"].strip().rstrip(".") + "."

# And in case you ever want to clear and regenerate:
@app.route('/suggestions/<job_id>/refresh')
def suggestions_refresh(job_id):
    SUGGESTION_CACHE.pop(job_id, None)
    return redirect(url_for('suggestions', job_id=job_id))

@app.template_filter('highlight')
def highlight_filter(text, term):
    """
    Wrap every occurrence of `term` in <span class="highlight">…</span>,
    case-insensitive. Returns a Markup so it's not auto-escaped.
    """
    if not term or not text:
        return text
    pattern = re.compile(r'(' + re.escape(term) + r')', re.IGNORECASE)
    return Markup(pattern.sub(r'<span class="highlight">\1</span>', text))

def get_job(job_id):
    """Retrieve a specific job by its ID from the session."""
    jobs = session.get('jobs', [])
    for job in jobs:
        if job.get('id') == job_id:
            return job
    return None

@app.route('/')
def landing():
    if 'current_job_id' in session:
        session.pop('current_job_id')
    return render_template('index.html')


from threading import Thread
from flask import request, session, render_template

def get_cached_people(company: str):
    """Return cached LinkedIn profiles for this company if still fresh."""
    entry = PEOPLE_CACHE.get(company.lower())
    if not entry:
        return None
    ts, profiles = entry
    if time.time() - ts < CACHE_DURATION:
        return profiles
    return None

def cache_people(company: str, profiles):
    """Cache the LinkedIn profiles under this company key."""
    PEOPLE_CACHE[company.lower()] = (time.time(), profiles)

@app.route('/track_json/people/<job_id>', methods=['POST'])
def track_people_json(job_id):
    job = get_job(job_id)
    url = request.json.get("url")
    if not job or not url:
        return jsonify(score=0), 400

    # Ensure we have a list and record only once
    clicked = job.setdefault('clicked_people', [])
    if url not in clicked:
        clicked.append(url)
        session.modified = True

    # Recompute score against current profiles list
    profiles = get_cached_people(job['company']) or []
    total = len(profiles)
    people_score = int(len(clicked) / total * 100) if total else 0

    return jsonify(score=people_score)

@app.route('/people/<job_id>')
def people(job_id):
    job = get_job(job_id)
    if not job or not job.get("company"):
        return "No company specified for this job.", 400

    session['current_job_id'] = job_id
    company = job["company"]

    profiles = get_cached_people(company)
    if profiles is None:
        try:
            profiles = fetch_company_people(company, size=10)
            cache_people(company, profiles)
        except Exception as e:
            app.logger.error(f"LinkedIn search error: {e}")
            profiles = []

    # ——— PEOPLE SECTION SCORING ———
    clicked = job.get('clicked_people', [])
    total   = len(profiles)
    people_score = int(len(clicked) / total * 100) if total > 0 else 0

    return render_template(
        "people.html",
        job=job,                    # ← add this
        company=company,
        profiles=profiles,
        clicked_people=clicked,
        people_score=people_score
    )


@app.route('/overview/<job_id>')
def overview(job_id):
    # 1) grab the job
    job = get_job(job_id)
    if not job:
        return "Job not found", 404

    # 2) keywords_score → full credit once they've visited the keywords page
    keywords_score = 100 if job.get('keywords_visited', False) else 0
    job['keywords_score'] = keywords_score

    # 3) resume_score → pull match_percentage from your resume_analysis
    resume_score = 0
    if job.get('resume_analysis'):
        resume_score = job['resume_analysis'].get('match_percentage', 0)
    job['resume_score'] = resume_score

    # 4) people_score → % of LinkedIn profiles they've clicked
    profiles = get_cached_people(job.get('company', '')) or []
    total_people   = len(profiles)
    clicked_people = job.get('clicked_people', [])
    people_score = int(len(clicked_people) / total_people * 100) if total_people else 0
    job['people_score'] = people_score

    company = job.get('company')
    industry = job.get('industry', 'technology')

    # fetch each (or empty list)
    company_items = get_cached_company_news(company) or [] if company else []
    market_items  = get_cached_news(industry) or []

    # unify & dedupe by link
    all_items = company_items + market_items
    all_urls  = { item['link'] for item in all_items }

    clicked_news = job.get('clicked_news', [])
    hits        = sum(1 for u in clicked_news if u in all_urls)
    total_news  = len(all_urls)

    news_score = int(hits / total_news * 100) if total_news else 0
    job['news_score'] = news_score

    # mark session modified so that these get saved
    session.modified = True

    # 6) overall with your weights
    overall_score = round(
        keywords_score * 0.1 +
        resume_score   * 0.4 +
        people_score   * 0.3 +
        news_score     * 0.2,
        2
    )

    return render_template(
        'overview.html',
        job=job,
        keywords_score=keywords_score,
        resume_score=resume_score,
        people_score=people_score,
        news_score=news_score,
        overall_score=overall_score
    )

@app.route('/news/<job_id>')
def news(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404

    session['current_job_id'] = job_id

    # Default to company if a company is set, otherwise market
    raw_source = request.args.get('source')
    source = raw_source if raw_source in ('market','company') else \
             ('company' if job.get('company') else 'market')

    news_items = []
    loading    = False

    if source == 'company' and job.get('company'):
        # COMPANY NEWS: try cache, else fetch and cache immediately
        company = job['company']
        cached  = get_cached_company_news(company)
        if cached is None:
            try:
                items = fetch_company_news(company)
                cache_company_news(company, items)
                news_items = items
            except Exception as e:
                app.logger.error(f"Company news fetch error: {e}")
                news_items = []
        else:
            news_items = cached
        industries = None

    else:
        # MARKET NEWS: try cache, else fetch and cache immediately
        industry = job.get('industry', 'technology')
        cached   = get_cached_news(industry)
        if cached is None:
            try:
                items = get_news_for_industry(industry)
                news_items = items
            except Exception as e:
                app.logger.error(f"Market news fetch error: {e}")
                news_items = []
        else:
            news_items = cached
        industries = get_available_industries()

    # ——— NEWS SCORING ———
    clicked = job.setdefault('clicked_news', [])
    current_urls  = [item['link'] for item in news_items]
    clicked_count = sum(1 for u in clicked if u in current_urls)
    total         = len(current_urls)
    news_score    = int(clicked_count / total * 100) if total else 0

    return render_template(
      'news.html',
      job=job,
      source=source,
      industry=job.get('industry','technology'),
      industries=industries,
      news_items=news_items,
      loading=loading,
      clicked_news=clicked,
      news_score=news_score
    )

@app.route('/jobs', methods=['GET', 'POST'])
def jobs():
    # clear any existing job context
    session.pop('current_job_id', None)

    jobs_list = session.setdefault('jobs', [])
    if request.method == 'POST':
        title    = request.form.get('job_title', '').strip()
        text     = request.form.get('job_text', '').strip()
        industry = request.form.get('industry')
        company  = request.form.get('company', '').strip() or None

        if title and text:
            new_job = {
                'id': str(uuid.uuid4()),
                'title': title,
                'text': text,
                'phrases': [],
                'highlighted_text': text,
                'resume_analysis': None,
                'resume_key': None,
                'industry': industry,
                'company': company
            }
            new_job['keywords_visited'] = False
            new_job['clicked_news'] = []
            new_job['clicked_people'] = []
            jobs_list.append(new_job)
            session['jobs'] = jobs_list

        return redirect(url_for('jobs'))

    # on GET, pass the list of possible industries into the template
    return render_template(
        'jobs.html',
        jobs=jobs_list,
        industries=get_available_industries()
    )
@app.route('/track_json/news/<job_id>', methods=['POST'])
def track_news_json(job_id):
    job = get_job(job_id)
    url = request.json.get("url")
    if not job or not url:
        return jsonify(score=0), 400

    # record unique clicks
    clicked = job.setdefault('clicked_news', [])
    if url not in clicked:
        clicked.append(url)
        session.modified = True

    # recompute against current news items
    # (we’ll look up what was just fetched)
    # Note: you could cache news_items in session if you like; here we re-fetch
    resp_items = get_cached_company_news(job['company']) if request.args.get('source')=='company' else get_cached_news(job.get('industry','technology'))
    news_items = resp_items or []

    urls = [item['link'] for item in news_items]
    hit_count = sum(1 for u in clicked if u in urls)
    total     = len(urls)
    score     = int(hit_count / total * 100) if total>0 else 0

    return jsonify(score=score)

def get_cached_company_news(company):
    entry = COMPANY_NEWS_CACHE.get(company.lower())
    if not entry: 
        return None
    timestamp, items = entry
    if time.time() - timestamp < CACHE_DURATION:
        return items
    return None

def cache_company_news(company, items):
    COMPANY_NEWS_CACHE[company.lower()] = (time.time(), items)

def fetch_company_news(company):
    """Grab top ~15 items from Google News RSS for the company."""
    q   = urllib.parse.quote(company)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    out = []
    for entry in feed.entries[:15]:
        img     = extract_image(entry)
        title   = clean_html(entry.get('title','No Title'))
        link    = entry.get('link','#')
        pub     = get_entry_date(entry)
        # pick earliest content field we can
        summary = ""
        for fld in ("summary","description","content"):
            if hasattr(entry,fld) and getattr(entry,fld):
                v = getattr(entry,fld)
                if isinstance(v, list): v = v[0].get('value','')
                summary = clean_html(v); break
        summary = summary[:200] + ("…" if len(summary)>200 else "")
        out.append({
          "title": title, "link": link,
          "published": pub, "summary": summary,
          "image": img, "source": company
        })
    return out

@app.route('/jobs/delete/<job_id>', methods=['POST'])
def delete_job(job_id):
    jobs_list = session.get('jobs', [])
    session['jobs'] = [job for job in jobs_list if job.get('id') != job_id]
    return redirect(url_for('jobs'))

def escape_js_string(text):
    return html.escape(text).replace('\n', '\\n').replace('\r', '').replace("'", "\\'").replace('"', '\\"')

app.jinja_env.filters['escapejs'] = escape_js_string


@app.route('/annotate_resume/<job_id>', methods=['GET'])
def annotate_resume(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404
    raw = session.get('resume_text', '')
    lines = _resume_store.get(job.get('resume_key'), [])
    return render_template('annotate.html', job=job, lines=lines)

@app.route('/annotate_resume/<job_id>', methods=['POST'])
def annotate_resume_post(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404
    bounds = {
        'skills_start': int(request.form['skills_start']),
        'skills_end':   int(request.form['skills_end']),
        'work_start':   int(request.form['work_start']),
        'work_end':     int(request.form['work_end']),
    }
    session['section_bounds'] = bounds
    return redirect(url_for('annotate_confirm', job_id=job_id))

@app.route('/annotate_confirm/<job_id>')
def annotate_confirm(job_id):
    job = get_job(job_id)
    if not job or job.get('resume_key') not in _resume_store:
        return "No resume loaded", 400

    lines = _resume_store[job['resume_key']]
    b = session.get('section_bounds', {})
    if not all(k in b for k in ('skills_start','skills_end','work_start','work_end')):
        return redirect(url_for('annotate_resume', job_id=job_id))

    skills = lines[b['skills_start']: b['skills_end'] + 1]
    work   = lines[b['work_start']:   b['work_end']   + 1]

    return render_template(
        'annotate_confirm.html',
        job=job,
        skills=skills,
        work=work
    )

def generate_cache_key(text, phrases):
    """Generate a unique cache key based on text and phrases."""
    sorted_phrases = sorted(phrases)
    key_content = text + '|' + '|'.join(sorted_phrases)
    return hashlib.md5(key_content.encode()).hexdigest()

@app.route('/upload_resume/<job_id>', methods=['GET'])
def upload_resume(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404

    session['current_job_id'] = job_id
    if request.args.get('new'):
        job['resume_analysis'] = None
        job['resume_key']      = None
        session.modified = True

    if job.get('resume_analysis'):
        return redirect(url_for('compare_resume', job_id=job_id))

    return render_template('resume_upload.html', job=job)

def highlight_phrases(text, phrases, selected_phrase=None):
    phrases = sorted(phrases, key=len, reverse=True)
    
    if selected_phrase:
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(selected_phrase) + r'(?![a-zA-Z0-9_])'
        repl = rf'<span class="highlight">{selected_phrase}</span>'
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text
    
    for phrase in phrases:
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(phrase) + r'(?![a-zA-Z0-9_])'
        repl = rf'<span class="highlight">{phrase}</span>'
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    return text

def get_word_positions(text):
    words_in_phrases = {}
    words = text.split()
    for i, word in enumerate(words):
        words_in_phrases[i] = word
    return words_in_phrases

def summarize_keyword_in_context(text, keyword, max_length=100):
    try:
        keyword_position = text.lower().find(keyword.lower())
        if keyword_position != -1:
            start = max(0, keyword_position - 150)
            end = min(len(text), keyword_position + 150)
            context = text[start:end]
        else:
            context = text[:min(len(text), 300)]
        prompt = (
            f"What is '{keyword}'? Provide a clear, brief explanation based on this context: {context}\n"
            f"Respond with a 1-2 sentence definition that explains what {keyword} is and its main purpose.\n"
            f"Your response should talk about how {keyword} means in the relevant industry of the context"
        )
        
        response = summarizer(prompt, max_length=max_length, do_sample=False)
        summary = response[0]['generated_text'].strip()
        summary = ' '.join(summary.split())
        if len(summary) < 20 or summary.lower() == keyword.lower():
            prompt = (
                f"Define and explain what '{keyword}' is in 1-2 clear sentences, "
                f"based on this context: {context}"
            )
            response = summarizer(prompt, max_length=max_length, do_sample=False)
            summary = response[0]['generated_text'].strip()
            summary = ' '.join(summary.split())

        return summary

    except Exception as e:
        print(f"Error summarizing keyword '{keyword}': {e}")
        return f"Unable to generate summary for '{keyword}'"

def summarize_keywords_in_context(text, keywords, max_length=512, use_cache=True):
    try:
        cache_key = generate_cache_key(text, keywords)
        if use_cache and cache_key in summary_cache:
            print("Using cached summaries")
            return summary_cache[cache_key]
        
        print("Generating new summaries")
        summaries = {}
        
        for keyword in keywords:
            try:
                keyword_cache_key = generate_cache_key(text, [keyword])
                if use_cache and keyword_cache_key in summary_cache:
                    keyword_summary = summary_cache[keyword_cache_key].get(keyword)
                    if keyword_summary:
                        summaries[keyword] = keyword_summary
                        continue
                
                keyword_position = text.lower().find(keyword.lower())
                if keyword_position != -1:
                    start = max(0, keyword_position - 200)
                    end = min(len(text), keyword_position + 200)
                    context = text[start:end]
                else:
                    context = text[:min(len(text), 400)]
                prompt = (
                    f"Based on this context, provide a clear 1-2 sentence explanation of what '{keyword}' is "
                    f"and its main purpose or significance. Context: {context}"
                )

                response = summarizer(prompt, max_length=max_length, do_sample=False)
                summary = response[0]['generated_text'].strip()
                summary = ' '.join(summary.split())
                if len(summary) < 20 or summary.lower() == keyword.lower():
                    prompt = (
                        f"Define and explain: What is '{keyword}' and what is it used for? "
                        f"Answer in 1-2 clear sentences, based on this context: {context}"
                    )
                    response = summarizer(prompt, max_length=max_length, do_sample=False)
                    summary = response[0]['generated_text'].strip()
                    summary = ' '.join(summary.split())

                summaries[keyword] = summary
                
                summary_cache[keyword_cache_key] = {keyword: summary}

            except Exception as e:
                print(f"Error processing keyword '{keyword}': {e}")
                summaries[keyword] = f"Unable to analyze '{keyword}'"
                continue
        summary_cache[cache_key] = summaries
        
        print("Generated summaries:", summaries)  # Debug output
        return summaries

    except Exception as e:
        print(f"Error in summarize_keywords_in_context: {e}")
        return {keyword: f"Error: {str(e)}" for keyword in keywords}


@app.route('/phrase_list/<job_id>')
def phrase_list(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404
    text    = job['text']
    phrases = job.get('phrases', [])
    frequencies = Counter(phrases)
    cache_key   = generate_cache_key(text, phrases)
    summaries = summarize_keywords_in_context(text, phrases)
    addressed_phrases = session.get('addressed_phrases', [])
    return render_template(
        'list.html',
        job=job,
        text=text,
        phrases=phrases,
        frequencies=frequencies,
        cache_key=cache_key,
        summaries=summaries,
        addressed_phrases=addressed_phrases
    )
  
@app.route('/')
def index():
    industries = get_available_industries()
    selected_industry = request.args.get('industry', 'technology')
    if selected_industry not in industries:
        selected_industry = 'technology'
    news_items = get_news_for_industry(selected_industry)
    
    return render_template('upload.html', 
                          news_items=news_items, 
                          selected_industry=selected_industry,
                          industries=industries)


@app.route('/extract/<job_id>', methods=['POST'])
def extract_keywords(job_id):
    """Extracts keywords for a specific job and redirects to the results."""
    job = get_job(job_id)
    if not job:
        return "Job not found", 404
    
    phrases = extract_phrases(job['text'])
    job['phrases'] = phrases
    job['highlighted_text'] = highlight_phrases(job['text'], phrases)
    
    session['current_job_id'] = job_id
    session.modified = True
    
    return redirect(url_for('highlight_phrase', job_id=job_id))

@app.route('/compare_resume/<job_id>', methods=['GET', 'POST'])
def compare_resume(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404

    session['current_job_id'] = job_id

    if request.method == 'POST':
        if 'resume' not in request.files or request.files['resume'].filename == '':
            return "No resume file provided.", 400

        resume_file = request.files['resume']
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_filename = temp_file.name
            resume_file.save(temp_filename)

        matcher = ResumeJobMatcher()
        results = matcher.analyze_resume_for_job(temp_filename, job['text'])
        job['resume_analysis'] = results
        
        with pdfplumber.open(temp_filename) as pdf:
            raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        
        resume_key = str(uuid.uuid4())
        _resume_store[resume_key] = raw_text.split('\n')
        job['resume_key'] = resume_key
        
        os.unlink(temp_filename)
        session.modified = True
        
        return redirect(url_for('compare_resume', job_id=job_id))

    if not job.get('resume_analysis'):
        return redirect(url_for('upload_resume', job_id=job_id))

    return render_template('resume_compare.html', job=job, results=job['resume_analysis'])

@app.route('/visualization')
def visualize_phrases():
    text = request.args.get('text', '')
    phrases = request.args.getlist('phrases')
    frequencies = Counter(phrases)
    return render_template('cloud.html', phrases=phrases, text=text, frequencies=frequencies)

def get_current():
    data = session.get('last_extraction')
    if not data:
        return redirect(url_for('index'))
    return data

@app.route('/highlight_phrase/<job_id>')
def highlight_phrase(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404

    session['current_job_id'] = job_id

    # —————— KEYWORDS SCORING ——————
    if not job.get('keywords_visited', False):
        job['keywords_visited'] = True
        session.modified = True

    # full credit once visited
    keywords_score = 100

    return render_template(
        'results.html',
        job=job,
        keywords_score=keywords_score
    )


@app.route('/edit_phrases/<job_id>', methods=['GET', 'POST'])
def edit_phrases(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404

    if request.method == 'POST':
        updated_phrases = json.loads(request.form.get('phrases', '[]'))
        job['phrases'] = updated_phrases
        job['highlighted_text'] = highlight_phrases(job['text'], updated_phrases)
        session.modified = True
        return redirect(url_for('highlight_phrase', job_id=job_id))
    original_text = job['text']
    phrases       = job.get('phrases', [])
    words_info    = []
    for i, word in enumerate(original_text.split()):
        words_info.append({
            'word': word,
            'in_phrase': any(word in p.split() for p in phrases),
            'phrase': next((p for p in phrases if word in p.split()), None)
        })

    return render_template(
        'editor.html',
        job=job,
        original_text=original_text,
        phrases=phrases,
        words_info=words_info
    )

@app.route('/summarize_keywords', methods=['POST'])
def summarize_keywords():
    text = request.form.get('text', '')
    phrases = json.loads(request.form.get('phrases', '[]'))

    if not text or not phrases:
        return "Error: Missing text or keywords", 400

    cache_key = generate_cache_key(text, phrases)
    if cache_key in summary_cache:
        summaries = summary_cache[cache_key]
    else:
        summaries = {}
        for phrase in phrases:
            summaries[phrase] = summarize_keyword_in_context(text, phrase)
        summary_cache[cache_key] = summaries

    return render_template('summaries.html', text=text, summaries=summaries)

@app.route('/suggestions/<job_id>')
def suggestions(job_id):
    job = get_job(job_id)
    if not job or not job.get('resume_key') or not job.get('resume_analysis'):
        return "No resume analysis available for this job", 400

    # only regenerate once per job
    if job_id not in SUGGESTION_CACHE:
        # pull just the missing keywords
        results = job['resume_analysis']
        missing = results.get('missing_keywords', [])
        resume_text = "\n".join(_resume_store[job['resume_key']])

        job_cache = {}
        for phrase in missing:
            cat = llm_categorize(phrase)
            sug = llm_suggest(phrase, cat, resume_text)
            job_cache[phrase] = {"category": cat, "suggestion": sug}
        SUGGESTION_CACHE[job_id] = job_cache

    # pivot into { category: [ {phrase,suggestion}, … ], … }
    by_cat = {}
    for phrase, info in SUGGESTION_CACHE[job_id].items():
        by_cat.setdefault(info["category"], []).append({
            "phrase": phrase,
            "suggestion": info["suggestion"]
        })

    resume_text = "\n".join(_resume_store[job['resume_key']])
    return render_template(
      "suggestions.html",
      job=job,
      resume_text=resume_text,
      suggestions=by_cat
    )

@app.route('/clear_cache')
def clear_cache():
    global summary_cache
    summary_cache = {}
    return "Cache cleared", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
