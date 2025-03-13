import spacy

def extract_phrases(
    text,
    model="en_core_web_sm",
    keep_single_word=True,
    top_n=None
):
    nlp = spacy.load(model)
    doc = nlp(text)

    phrases = []
    accum_tokens = []
    
    ALLOWED_POS = {"PROPN", "NOUN", "ADJ", "VERB", "X"}

    def is_significant_single_word(token):
        if token.ent_type_:
            return True
        if len(token.text) >= 4 and (token.text[0].isupper() or token.text.isupper()):
            return True
        return False

    def valid_multi_word_chunk(tokens):
        if not any(t.pos_ in ("NOUN","PROPN") for t in tokens):
            return False
        if len(tokens) == 2:
            pos_tags = [t.pos_ for t in tokens]
            all_lower = all(t.text.islower() for t in tokens)
            if pos_tags == ["ADJ","NOUN"] and all_lower:
                return False
        return True

    def flush_acc():
        if not accum_tokens:
            return
        if len(accum_tokens) > 1:
            if valid_multi_word_chunk(accum_tokens):
                phrase = " ".join(t.text for t in accum_tokens)
                phrases.append(phrase)
        else:
            token = accum_tokens[0]
            if keep_single_word and is_significant_single_word(token):
                phrases.append(token.text)
        accum_tokens.clear()
    for token in doc:
        if token.pos_ in ALLOWED_POS and not token.is_stop:
            accum_tokens.append(token)
        else:
            flush_acc()
    flush_acc()

    unique_phrases = sorted(set(phrases))
    print(unique_phrases)
    return unique_phrases
