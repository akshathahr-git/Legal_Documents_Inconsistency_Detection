from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from enum import Enum

class DocumentType(str, Enum):
    TEXT = "text"
    PDF = "pdf"

class Clause(BaseModel):
    id: str
    text: str
    document_name: str
    page_number: Optional[int] = None
    clause_number: Optional[str] = None

class ComparisonResult(BaseModel):
    clause_1: Clause
    clause_2: Clause
    similarity_score: float
    is_inconsistent: bool
    inconsistency_type: Optional[str] = None  # direct_contradiction, semantic_mismatch, numerical_conflict
    confidence: float
    explanation: str

class DocumentAnalysisRequest(BaseModel):
    documents: List[Dict[str, str]]  # [{"name": "doc1.pdf", "content": "text or base64"}]

class DocumentAnalysisResponse(BaseModel):
    total_clauses_extracted: int
    comparisons_made: int
    results: List[ComparisonResult]
    summary: Dict[str, Union[int, float]]
