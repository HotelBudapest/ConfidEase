import sys
import spacy

def extract_keywords_from_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    interesting_pos = {"NOUN", "PROPN", "ADJ"}

    keywords = []

    for token in doc:
        if not token.is_stop and token.is_alpha and token.pos_ in interesting_pos:
            keywords.append(token.lemma_.lower())

    return set(keywords)
