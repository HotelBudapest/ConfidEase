from flask import Flask, render_template, request, redirect, url_for, send_file, session
from extractor import extract_phrases
from wordcloud import WordCloud
from transformers import pipeline
from collections import Counter
import re
import io
import html
import json
import matplotlib.pyplot as plt
from collections import Counter
import hashlib

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' 
summarizer = pipeline("text2text-generation", model="google/flan-t5-large")

summary_cache = {}

def escape_js_string(text):
    return html.escape(text).replace('\n', '\\n').replace('\r', '').replace("'", "\\'").replace('"', '\\"')

app.jinja_env.filters['escapejs'] = escape_js_string

def generate_cache_key(text, phrases):
    """Generate a unique cache key based on text and phrases."""
    sorted_phrases = sorted(phrases)
    key_content = text + '|' + '|'.join(sorted_phrases)
    return hashlib.md5(key_content.encode()).hexdigest()

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

@app.route('/phrase_list', methods=['GET', 'POST'])
def phrase_list():
    print("Debug - Entering phrase_list route")  # Debug output
    
    if request.method == 'POST':
        addressed_phrases = request.form.getlist('addressed_phrases')
        addressed_phrases = set(addressed_phrases)
        return render_template(
            'list.html',
            text=request.args.get('text', ''),
            phrases=sorted(request.args.getlist('phrases')),
            frequencies=json.loads(request.args.get('frequencies', '{}')),
            addressed_phrases=addressed_phrases,
            summaries=json.loads(request.args.get('summaries', '{}'))
        )

    text = request.args.get('text', '')
    phrases = request.args.getlist('phrases')
    frequencies = Counter(phrases)

    cache_key = generate_cache_key(text, phrases)
    
    try:
        if cache_key in summary_cache:
            print(f"Debug - Using cached summaries with key: {cache_key}")
            summaries = summary_cache[cache_key]
        else:
            print(f"Debug - Generating new summaries for key: {cache_key}")
            summaries = summarize_keywords_in_context(text, phrases)
            summary_cache[cache_key] = summaries
    except Exception as e:
        print(f"Error in phrase_list: {e}")
        summaries = {phrase: f"Error generating summary" for phrase in phrases}
    return render_template(
        'list.html',
        text=text,
        phrases=sorted(frequencies.keys(), key=lambda x: frequencies[x], reverse=True),
        frequencies=frequencies,
        addressed_phrases=set(),
        summaries=summaries
    )
  
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/extract', methods=['POST'])
def extract_keywords():
    original_text = request.form.get('text', '')
    
    if not original_text.strip():
        return redirect(request.url)
    phrases = extract_phrases(original_text)
    highlighted_text = highlight_phrases(original_text, phrases)
    return render_template(
        'results.html', 
        highlighted_text=highlighted_text, 
        phrases=phrases, 
        original_text=original_text
    )

@app.route('/visualization')
def visualize_phrases():
    text = request.args.get('text', '')
    phrases = request.args.getlist('phrases')
    frequencies = Counter(phrases)
    return render_template('cloud.html', phrases=phrases, text=text, frequencies=frequencies)

@app.route('/highlight_phrase')
def highlight_phrase():
    original_text = request.args.get('original_text', '')
    phrases = request.args.get('phrases', '[]')
    try:
        phrases = json.loads(phrases) if isinstance(phrases, str) else phrases
    except json.JSONDecodeError:
        phrases = []
    selected_phrase = request.args.get('phrase')
    highlighted_text = highlight_phrases(original_text, phrases, selected_phrase)
    return render_template('results.html', highlighted_text=highlighted_text, phrases=phrases, original_text=original_text)

@app.route('/edit_phrases', methods=['GET', 'POST'])
def edit_phrases():
    if request.method == 'POST':
        original_text = request.form.get('original_text', '')
        phrases = request.form.get('phrases', '[]')
        phrases = eval(phrases)
        highlighted_text = highlight_phrases(original_text, phrases)
        return render_template('results.html', highlighted_text=highlighted_text, phrases=phrases, original_text=original_text)
    
    original_text = request.args.get('original_text', '')
    phrases = request.args.getlist('phrases')
    word_positions = get_word_positions(original_text)
    words_info = []
    for pos, word in word_positions.items():
        word_info = {
            'word': word,
            'position': pos,
            'in_phrase': False,
            'phrase': None
        }
        for phrase in phrases:
            if ' ' in phrase:  
                if word in phrase.split():
                    word_info['in_phrase'] = True
                    word_info['phrase'] = phrase
            elif phrase == word:  
                word_info['in_phrase'] = True
                word_info['phrase'] = phrase
                
        words_info.append(word_info)

    return render_template('editor.html', 
                         original_text=original_text, 
                         phrases=phrases,
                         words_info=words_info)

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
    app.run(dubug=True)
    # app.run(host='0.0.0.0', port=5001)
