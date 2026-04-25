import re
from typing import List, Dict
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
    CONTRADICTORY_TERMS = {
        ('mandatory', 'optional'), ('required', 'optional'), ('must', 'may'),
        ('shall', 'may not'), ('guarantee', 'no guarantee'), ('warranty', 'as-is')
    }

    def detect_inconsistencies(
        self,
        clauses: List[Dict],
        embeddings: np.ndarray,
        similarity_threshold: float = 0.65
    ) -> List[Dict]:
        """Detect inconsistencies between all clause pairs"""
        results = []
        similarities = cosine_similarity(embeddings)

        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                sim_score = similarities[i][j]
                if sim_score >= similarity_threshold:
                    contradiction_result = self._check_contradiction(
                        clauses[i]['text'], clauses[j]['text'], sim_score
                    )
                    results.append({
                        'clause_1': clauses[i],
                        'clause_2': clauses[j],
                        'similarity_score': sim_score,
                        'is_inconsistent': contradiction_result['is_inconsistent'],
                        'inconsistency_type': contradiction_result['type'],
                        'confidence': contradiction_result['confidence'],
                        'explanation': contradiction_result['explanation']
                    })
        return results

    def _check_contradiction(self, text1: str, text2: str, similarity: float) -> Dict:
        """Check if two texts contradict each other"""
        if self._check_negation_contradiction(text1, text2) > 0.7:
            return {'is_inconsistent': True, 'type': 'direct_contradiction',
                    'confidence': 0.85, 'explanation': "One clause negates the other"}

        num_conflict = self._check_numerical_conflict(text1, text2)
        if num_conflict:
            return {'is_inconsistent': True, 'type': 'numerical_conflict',
                    'confidence': 0.9, 'explanation': num_conflict}

        verb_conflict = self._check_opposite_verbs(text1, text2)
        if verb_conflict:
            return {'is_inconsistent': True, 'type': 'semantic_mismatch',
                    'confidence': 0.75, 'explanation': verb_conflict}

        term_conflict = self._check_contradictory_terms(text1, text2)
        if term_conflict:
            return {'is_inconsistent': True, 'type': 'term_contradiction',
                    'confidence': 0.8, 'explanation': term_conflict}

        return {'is_inconsistent': False, 'type': None,
                'confidence': 1 - similarity, 'explanation': "No contradictions detected"}

    def _check_negation_contradiction(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        core_similarity = len(words1 & words2) / max(len(words1 | words2), 1)
        if core_similarity > 0.5:
            neg1 = any(w in self.NEGATION_WORDS for w in words1)
            neg2 = any(w in self.NEGATION_WORDS for w in words2)
            if neg1 ^ neg2:  # one has negation, the other doesn’t
                return 0.85
        return 0.0

    def _check_numerical_conflict(self, text1: str, text2: str) -> str:
        nums1 = re.findall(r'[₹$]?\s*[\d,]+(?:\.\d+)?%?', text1)
        nums2 = re.findall(r'[₹$]?\s*[\d,]+(?:\.\d+)?%?', text2)
        for n1 in nums1:
            for n2 in nums2:
                try:
                    val1 = float(n1.replace(",", "").replace("₹", "").replace("$", "").replace("%", ""))
                    val2 = float(n2.replace(",", "").replace("₹", "").replace("$", "").replace("%", ""))
                except ValueError:
                    continue
                if abs(val1 - val2) >= 1 and self._same_context(text1, text2):
                    return f"Numerical conflict detected: {val1} vs {val2}"
        return None

    def _check_opposite_verbs(self, text1: str, text2: str) -> str:
        for v1, v2 in self.OPPOSITE_VERBS:
            if v1 in text1.lower() and v2 in text2.lower():
                return f"Contract states '{v1}' and '{v2}' which are contradictory"
            if v2 in text1.lower() and v1 in text2.lower():
                return f"Contract states '{v2}' and '{v1}' which are contradictory"
        return None

    def _check_contradictory_terms(self, text1: str, text2: str) -> str:
        for t1, t2 in self.CONTRADICTORY_TERMS:
            if t1 in text1.lower() and t2 in text2.lower():
                return f"Contradictory terms: '{t1}' vs '{t2}'"
            if t2 in text1.lower() and t1 in text2.lower():
                return f"Contradictory terms: '{t2}' vs '{t1}'"
        return None

    def _same_context(self, text1: str, text2: str) -> bool:
        common_words = set(text1.lower().split()) & set(text2.lower().split())
        return len(common_words) > 2
