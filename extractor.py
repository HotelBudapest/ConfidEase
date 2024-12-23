import sys
import spacy

def extract_keywords_from_text(text, nlp_model):
    doc = nlp_model(text)
    interesting_pos = {"NOUN", "PROPN", "ADJ"}

    keywords = []

    for token in doc:
        if not token.is_stop and token.is_alpha and token.pos_ in interesting_pos:
            keywords.append(token.lemma_.lower())

    return set(keywords)

def main():
    if len(sys.argv) < 2:
        print("Usage: python keyword_extractor.py <text_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    nlp = spacy.load("en_core_web_sm")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text_data = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    keywords = extract_keywords_from_text(text_data, nlp)
    print("Extracted Keywords:")
    for kw in sorted(keywords):
        print(f"- {kw}")

if __name__ == "__main__":
    main()

