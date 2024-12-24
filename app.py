from flask import Flask, render_template, request, redirect, url_for
from extractor import extract_phrases
import re

app = Flask(__name__)

def highlight_phrases(text, phrases):
    phrases = sorted(phrases, key=len, reverse=True)

    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        repl = rf'<span class="highlight">{phrase}</span>'
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    return text

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
    text = file.read().decode('utf-8')
    phrases = extract_phrases(text, keep_single_word=False)
    highlighted_text = highlight_phrases(text, phrases)
    return render_template('results.html', highlighted_text=highlighted_text)

if __name__ == '__main__':
    app.run(debug=True)
