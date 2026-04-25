import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class ClauseExtractor:
    """Extracts clauses from legal documents"""
    
    # Legal document clause patterns
    CLAUSE_PATTERNS = [
        r'(?:Clause|Section|Article)\s+(\d+(?:\.\d+)?)[:\s]+([^.]+[.])',
        r'(\d+\.\d+|\d+)\s+([A-Z][^.!?]*[.!?])',
        r'([A-Z][A-Z\s]+):\s*([^.!?]+[.!?])',  # ALL CAPS HEADERS
    ]
    
    def extract_clauses(self, text: str, document_name: str) -> List[Dict]:
        """Extract clauses from document text"""
        clauses = []
        
        # Clean text
        text = self._clean_text(text)
        
        # Try pattern-based extraction first
        extracted = self._extract_by_patterns(text, document_name)
        
        if len(extracted) < 3:  # Fallback to sentence-based extraction
            extracted = self._extract_by_sentences(text, document_name)
        
        # Add clause IDs
        for idx, clause in enumerate(extracted):
            clause['id'] = f"{document_name}_clause_{idx+1}"
            clause['clause_number'] = str(idx + 1)
        
        return extracted
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page breaks markers
        text = re.sub(r'[-=]+\s*page\s*\d+\s*[-=]+', ' ', text, flags=re.I)
        return text.strip()
    
    def _extract_by_patterns(self, text: str, doc_name: str) -> List[Dict]:
        """Extract clauses using legal document patterns"""
        clauses = []
        
        for pattern in self.CLAUSE_PATTERNS:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    clause_num = match.group(1)
                    clause_text = match.group(2) if len(match.groups()) > 1 else match.group(0)
                    
                    clauses.append({
                        'document_name': doc_name,
                        'text': clause_text.strip(),
                        'clause_number': clause_num
                    })
        
        # Remove duplicates while preserving order
        seen_texts = set()
        unique_clauses = []
        for clause in clauses:
            text_key = clause['text'][:100]  # Use first 100 chars as key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_clauses.append(clause)
        
        return unique_clauses
    
    def _extract_by_sentences(self, text: str, doc_name: str) -> List[Dict]:
        """Fallback: Extract clauses by sentence boundaries"""
        sentences = sent_tokenize(text)
        clauses = []
        
        # Group sentences into meaningful clauses (max 3 sentences per clause)
        current_clause = []
        for sentence in sentences:
            current_clause.append(sentence)
            if len(sentence) > 100 or len(current_clause) >= 3:
                clauses.append({
                    'document_name': doc_name,
                    'text': ' '.join(current_clause),
                    'clause_number': f"clause_{len(clauses)+1}"
                })
                current_clause = []
        
        # Add remaining sentences
        if current_clause:
            clauses.append({
                'document_name': doc_name,
                'text': ' '.join(current_clause),
                'clause_number': f"clause_{len(clauses)+1}"
            })
        
        return clauses