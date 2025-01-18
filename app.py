from flask import Flask, render_template, request, redirect, url_for, send_file
from extractor import extract_phrases
from wordcloud import WordCloud
import re
import io
import html
import json
import matplotlib.pyplot as plt

app = Flask(__name__)

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

def get_word_positions(text):
    """Generate a map of words and their positions in multi-word phrases"""
    words_in_phrases = {}
    words = text.split()
    for i, word in enumerate(words):
        words_in_phrases[i] = word
    return words_in_phrases

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
    from collections import Counter
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
            addressed_phrases=addressed_phrases
        )
    text = request.args.get('text', '')
    phrases = request.args.getlist('phrases')
    from collections import Counter
    frequencies = Counter(phrases)

    return render_template(
        'list.html', 
        text=text,
        phrases=sorted(frequencies.keys(), key=lambda x: frequencies[x], reverse=True),
        frequencies=frequencies,
        addressed_phrases=set()
    )

if __name__ == '__main__':
    app.run(debug=True)
