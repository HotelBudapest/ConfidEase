from flask import Flask, render_template, request, redirect, url_for, send_file, session
import uuid
from extractor import extract_phrases
from wordcloud import WordCloud
from transformers import pipeline
from flask_session import Session
from collections import Counter
import re
import io
import html
import json
import matplotlib.pyplot as plt
from collections import Counter
import hashlib
import os
import tempfile
from resume_matcher import ResumeJobMatcher
from news_fetcher import get_news_for_industry, get_available_industries
import pdfplumber

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

@app.route('/jobs', methods=['GET', 'POST'])
def jobs():
    session.pop('current_job_id', None)
    jobs_list = session.setdefault('jobs', [])
    if request.method == 'POST':
        title = request.form.get('job_title', '').strip()
        text = request.form.get('job_text', '').strip()
        if title and text:
            new_job = {
                'id': str(uuid.uuid4()),
                'title': title,
                'text': text,
                'phrases': [],
                'highlighted_text': text,
                'resume_analysis': None,
                'resume_key': None
            }
            jobs_list.append(new_job)
            session['jobs'] = jobs_list
        return redirect(url_for('jobs'))
    return render_template('jobs.html', jobs=jobs_list)

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
    return render_template('results.html', job=job)


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

@app.route('/clear_cache')
def clear_cache():
    global summary_cache
    summary_cache = {}
    return "Cache cleared", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
