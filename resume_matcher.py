import os
import re
import spacy
from pypdf import PdfReader
from collections import Counter
from extractor import extract_phrases

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
