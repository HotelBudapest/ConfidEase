import spacy

def customize_stopwords(nlp):
    custom_stops = {"looking", "making", "work", "people"}
    for w in custom_stops:
        nlp.vocab[w].is_stop = True

def extract_phrases(text, model="en_core_web_sm", keep_single_word=False):
    nlp = spacy.load(model)
    customize_stopwords(nlp)
    doc = nlp(text)

    phrases = []
    accum = []

    def flush_accumulator(acc):
        if not acc:
            return
        if len(acc) > 1:
            phrase = " ".join(acc)
            phrases.append(phrase.lower())
        elif len(acc) == 1 and keep_single_word:
            phrases.append(acc[0].lower())
        acc.clear()

    for token in doc:
        if (
            token.pos_ in {"PROPN", "NOUN", "ADJ"} 
            and not token.is_stop 
            and token.is_alpha
        ):
            accum.append(token.text)
        else:
            flush_accumulator(accum)
    
    flush_accumulator(accum)

    unique_phrases = sorted(set(phrases))
    return unique_phrases
