import os
import re
import spacy
from pypdf import PdfReader
from collections import Counter

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
        if len(tokens) >= 2:
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
    return unique_phrases

class ResumeJobMatcher:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.model = model
    
    def extract_text_from_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_resume_phrases(self, resume_text):
        preprocessed_text = self.preprocess_text(resume_text)
        resume_phrases = extract_phrases(
            preprocessed_text,
            model=self.model,
            keep_single_word=True
        )
        return resume_phrases
    
    def match_keywords(self, resume_text, job_keywords):
        resume_phrases = self.extract_resume_phrases(resume_text)
        resume_phrases_lower = [phrase.lower() for phrase in resume_phrases]
        job_keywords_lower = [keyword.lower() for keyword in job_keywords]
        matches = []
        missing = []
        
        for keyword in job_keywords_lower:
            if keyword in resume_phrases_lower:
                matches.append(keyword)
            else:
                found = False
                for resume_phrase in resume_phrases_lower:
                    if keyword in resume_phrase or resume_phrase in keyword:
                        matches.append(keyword)
                        found = True
                        break
                
                if not found:
                    keyword_words = keyword.split()
                    if len(keyword_words) > 1:
                        word_matches = sum(1 for word in keyword_words if any(word in phrase for phrase in resume_phrases_lower))
                        if word_matches >= len(keyword_words) // 2:  # At least half the words match
                            matches.append(keyword)
                            found = True
                
                if not found:
                    missing.append(keyword)
        original_case_matches = []
        for match in matches:
            original_idx = [i for i, k in enumerate(job_keywords_lower) if k == match]
            if original_idx:
                original_case_matches.append(job_keywords[original_idx[0]])
            else:
                original_case_matches.append(match)
        
        original_case_missing = []
        for miss in missing:
            original_idx = [i for i, k in enumerate(job_keywords_lower) if k == miss]
            if original_idx:
                original_case_missing.append(job_keywords[original_idx[0]])
            else:
                original_case_missing.append(miss)
        match_percentage = (len(matches) / len(job_keywords)) * 100 if job_keywords else 0
        
        return {
            "matched_keywords": original_case_matches,
            "missing_keywords": original_case_missing,
            "match_percentage": round(match_percentage, 2),
            "total_keywords": len(job_keywords),
            "matched_count": len(matches),
            "resume_keywords": resume_phrases
        }
    
    def analyze_resume_for_job(self, resume_pdf_path, job_description_text):
        try:
            job_keywords = extract_phrases(
                job_description_text,
                model=self.model,
                keep_single_word=True
            )
            resume_text = self.extract_text_from_pdf(resume_pdf_path)
            match_results = self.match_keywords(resume_text, job_keywords)
            
            return match_results
            
        except Exception as e:
            print(f"Error analyzing resume against job description: {str(e)}")
            return None

# if __name__ == "__main__":
#     matcher = ResumeJobMatcher()
#     job_description = """
#     We are looking for a Python Developer with experience in Machine Learning and Natural Language Processing.
#     The ideal candidate will have skills in:
#     - Python programming
#     - TensorFlow or PyTorch
#     - Data analysis
#     - SQL databases
#     - RESTful APIs
#     Familiarity with cloud platforms like AWS is a plus.
#     """
#     resume_path = "path/to/your/resume.pdf"
#     results = matcher.analyze_resume_for_job(resume_path, job_description)
#     if results:
#         print("\n===== RESUME TO JOB MATCH RESULTS =====")
#         print(f"Match percentage: {results['match_percentage']}%")
#         print(f"Matched keywords ({results['matched_count']}/{results['total_keywords']}):")
#         for keyword in results['matched_keywords']:
#             print(f"✓ {keyword}")
#
#         print("\nMissing keywords:")
#         for keyword in results['missing_keywords']:
#             print(f"✗ {keyword}")
