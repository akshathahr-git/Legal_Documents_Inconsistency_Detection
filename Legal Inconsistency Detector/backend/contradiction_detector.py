import re
from typing import List, Dict, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContradictionDetector:
    """Detects contradictions between legal clauses"""
    
    # Contradiction indicators
    NEGATION_WORDS = {'not', 'no', 'never', 'without', 'except', 'unless', 'excluding'}
    OPPOSITE_VERBS = {
        ('allow', 'prohibit'), ('permit', 'forbid'), ('include', 'exclude'),
        ('increase', 'decrease'), ('raise', 'lower'), ('offer', 'withdraw'),
        ('accept', 'reject'), ('authorize', 'deny'), ('approve', 'disapprove')
    }
    
    # Numerical patterns
    NUM_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:%|dollars?|USD|euros?|years?|months?|days?)\b', re.I)
    
    # Legal term mappings
    CONTRADICTORY_TERMS = {
        ('mandatory', 'optional'), ('required', 'optional'), ('must', 'may'),
        ('shall', 'may not'), ('guarantee', 'no guarantee'), ('warranty', 'as-is')
    }
    
    def detect_inconsistencies(self, clauses: List[Dict], embeddings: np.ndarray, 
                                similarity_threshold: float = 0.65) -> List[Dict]:
        """Detect inconsistencies between all clause pairs"""
        results = []
        n_clauses = len(clauses)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Compare similar clauses (above threshold)
        for i in range(n_clauses):
            for j in range(i+1, n_clauses):
                sim_score = similarities[i][j]
                
                if sim_score >= similarity_threshold:
                    # Check for contradictions
                    contradiction_result = self._check_contradiction(
                        clauses[i]['text'], 
                        clauses[j]['text'],
                        sim_score
                    )
                    
                    if contradiction_result['is_inconsistent']:
                        results.append({
                            'clause_1': clauses[i],
                            'clause_2': clauses[j],
                            'similarity_score': sim_score,
                            'is_inconsistent': True,
                            'inconsistency_type': contradiction_result['type'],
                            'confidence': contradiction_result['confidence'],
                            'explanation': contradiction_result['explanation']
                        })
                    else:
                        # Consistent clauses
                        results.append({
                            'clause_1': clauses[i],
                            'clause_2': clauses[j],
                            'similarity_score': sim_score,
                            'is_inconsistent': False,
                            'inconsistency_type': None,
                            'confidence': 0.85,
                            'explanation': "Clauses are semantically consistent"
                        })
        
        return results
    
    def _check_contradiction(self, text1: str, text2: str, similarity: float) -> Dict:
        """Check if two texts contradict each other"""
        
        # Check for negation patterns
        negation_score = self._check_negation_contradiction(text1, text2)
        if negation_score > 0.7:
            return {
                'is_inconsistent': True,
                'type': 'direct_contradiction',
                'confidence': negation_score,
                'explanation': "One clause contains negation of the other"
            }
        
        # Check for numerical conflicts
        num_conflict = self._check_numerical_conflict(text1, text2)
        if num_conflict:
            return {
                'is_inconsistent': True,
                'type': 'numerical_conflict',
                'confidence': 0.9,
                'explanation': num_conflict
            }
        
        # Check for opposite verb pairs
        verb_conflict = self._check_opposite_verbs(text1, text2)
        if verb_conflict:
            return {
                'is_inconsistent': True,
                'type': 'semantic_mismatch',
                'confidence': 0.75,
                'explanation': verb_conflict
            }
        
        # Check for contradictory legal terms
        term_conflict = self._check_contradictory_terms(text1, text2)
        if term_conflict:
            return {
                'is_inconsistent': True,
                'type': 'term_contradiction',
                'confidence': 0.8,
                'explanation': term_conflict
            }
        
        return {
            'is_inconsistent': False,
            'type': None,
            'confidence': 1 - similarity,
            'explanation': "No contradictions detected"
        }
    
    def _check_negation_contradiction(self, text1: str, text2: str) -> float:
        """Check if one text negates the other"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Check if core content matches but one has negation
        core_similarity = len(words1 & words2) / max(len(words1 | words2), 1)
        
        if core_similarity > 0.5:
            negations1 = len([w for w in words1 if w in self.NEGATION_WORDS])
            negations2 = len([w for w in words2 if w in self.NEGATION_WORDS])
            
            if (negations1 > 0 and negations2 == 0) or (negations2 > 0 and negations1 == 0):
                return 0.85
        
        return 0.0
    
    def _check_numerical_conflict(self, text1: str, text2: str) -> str:
        """Check for numerical conflicts (e.g., 30 days vs 60 days)"""
        nums1 = self.NUM_PATTERN.findall(text1)
        nums2 = self.NUM_PATTERN.findall(text2)
        
        if nums1 and nums2:
            # Compare numerical values if mentioning same metric
            for n1 in nums1:
                for n2 in nums2:
                    if abs(float(n1) - float(n2)) > 0 and self._same_context(text1, text2):
                        return f"Numerical conflict: {n1} vs {n2}"
        return None
    
    def _check_opposite_verbs(self, text1: str, text2: str) -> str:
        """Check for opposite verb pairs"""
        for verb1, verb2 in self.OPPOSITE_VERBS:
            if verb1 in text1.lower() and verb2 in text2.lower():
                return f"Contract states '{verb1}' and '{verb2}' which are contradictory"
            if verb2 in text1.lower() and verb1 in text2.lower():
                return f"Contract states '{verb2}' and '{verb1}' which are contradictory"
        return None
    
    def _check_contradictory_terms(self, text1: str, text2: str) -> str:
        """Check for contradictory legal terms"""
        for term1, term2 in self.CONTRADICTORY_TERMS:
            if term1 in text1.lower() and term2 in text2.lower():
                return f"Contradictory terms: '{term1}' vs '{term2}'"
            if term2 in text1.lower() and term1 in text2.lower():
                return f"Contradictory terms: '{term2}' vs '{term1}'"
        return None
    
    def _same_context(self, text1: str, text2: str) -> bool:
        """Check if two texts discuss the same topic"""
        # Simple overlap check
        common_words = set(text1.lower().split()) & set(text2.lower().split())
        return len(common_words) > 5