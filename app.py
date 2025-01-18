from flask import Flask, render_template, request, redirect, url_for, send_file
from extractor import extract_phrases
from wordcloud import WordCloud
from transformers import pipeline  # Hugging Face Summarization
from collections import Counter
import re
import io
import html
import json
import matplotlib.pyplot as plt
from collections import Counter

app = Flask(__name__)
summarizer = pipeline("text2text-generation", model="google/flan-t5-large")

def escape_js_string(text):
    return html.escape(text).replace('\n', '\\n').replace('\r', '').replace("'", "\\'").replace('"', '\\"')

app.jinja_env.filters['escapejs'] = escape_js_string

def highlight_phrases(text, phrases, selected_phrase=None):
    phrases = sorted(phrases, key=len, reverse=True)
    if selected_phrase:
        pattern = r'\b' + re.escape(selected_phrase) + r'\b'
        repl = rf'<span class="highlight">{selected_phrase}</span>'
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text
    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        repl = rf'<span class="highlight">{phrase}</span>'
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text

def summarize_keywords_in_context(text, keywords, max_length=512):
    try:
        def chunk_text(text, max_length):
            words = text.split()
            chunks = []
            chunk = ""
            for word in words:
                if len(chunk.split()) + len(word.split()) <= max_length:
                    chunk += " " + word
                else:
                    chunks.append(chunk.strip())
                    chunk = word
            if chunk:
                chunks.append(chunk.strip())
            return chunks

        chunks = chunk_text(text, max_length)
        summaries = {}
        
        for chunk in chunks:
            prompt = f"Summarize the roles of the following keywords in the text:\nKeywords: {', '.join(keywords)}\nText: {chunk}\nFormat your response as:\n"
            prompt += "\n".join([f"- {keyword}:" for keyword in keywords])

            # Run the model on the chunked text
            response = summarizer(prompt, max_length=max_length, do_sample=False)
            
            for line in response[0].get('generated_text', '').split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    summaries[key.strip("- ")] = val.strip()

        return summaries

    except Exception as e:
        print(f"Error summarizing keywords: {e}")
        return {keyword: "Error generating summary" for keyword in keywords}

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/extract', methods=['POST'])
def extract_keywords():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    original_text = file.read().decode('utf-8')
    phrases = extract_phrases(original_text)
    highlighted_text = highlight_phrases(original_text, phrases)
    return render_template('results.html', highlighted_text=highlighted_text, phrases=phrases, original_text=original_text)

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

@app.route('/phrase_list', methods=['GET', 'POST'])
def phrase_list():
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

    try:
        summaries = summarize_keywords_in_context(text, phrases)
        print(f"Summaries: {summaries}")  # Debugging line to check summaries
    except Exception as e:
        print(f"Error in phrase_list: {e}")
        summaries = {phrase: "Error generating summary" for phrase in phrases}

    return render_template(
        'list.html',
        text=text,
        phrases=sorted(frequencies.keys(), key=lambda x: frequencies[x], reverse=True),
        frequencies=frequencies,
        addressed_phrases=set(),
        summaries=summaries
    )

@app.route('/summarize_keywords', methods=['POST'])
def summarize_keywords():
    text = request.form.get('text', '')
    phrases = json.loads(request.form.get('phrases', '[]'))

    if not text or not phrases:
        return "Error: Missing text or keywords", 400

    summaries = {}
    for phrase in phrases:
        summaries[phrase] = summarize_keyword_in_context(text, phrase)

    return render_template('summaries.html', text=text, summaries=summaries)

if __name__ == '__main__':
    app.run(debug=True)
