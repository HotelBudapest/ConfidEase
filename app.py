from flask import Flask, render_template, request, redirect, url_for
from extractor import extract_keywords_from_text

app = Flask(__name__)

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
    keywords = extract_keywords_from_text(text)

    return render_template('results.html', keywords=keywords)

if __name__ == '__main__':
    app.run(debug=True)

