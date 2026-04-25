from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import PyPDF2
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

from models import DocumentAnalysisRequest, DocumentAnalysisResponse, ComparisonResult, Clause
from clause_extractor import ClauseExtractor
from embedding_engine import EmbeddingEngine
from contradiction_detector import ContradictionDetector

app = FastAPI(title="Legal Document Inconsistency Detector")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
clause_extractor = ClauseExtractor()
embedding_engine = EmbeddingEngine()
contradiction_detector = ContradictionDetector()
executor = ThreadPoolExecutor(max_workers=4)

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_documents(files: List[UploadFile] = File(...)):
    """Analyze multiple documents for inconsistencies"""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least 2 documents")
    
    # Extract text from all documents
    documents = []
    for file in files:
        content = await file.read()
        
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8')
        
        documents.append({
            'name': file.filename,
            'text': text
        })
    
    # Run analysis in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        executor, 
        process_documents, 
        documents
    )
    
    return results

def process_documents(documents: List[Dict]) -> DocumentAnalysisResponse:
    """Process documents and detect inconsistencies"""
    all_clauses = []
    
    # Extract clauses from each document
    for doc in documents:
        clauses = clause_extractor.extract_clauses(doc['text'], doc['name'])
        all_clauses.extend(clauses)
    
    if len(all_clauses) < 2:
        return DocumentAnalysisResponse(
            total_clauses_extracted=len(all_clauses),
            comparisons_made=0,
            results=[],
            summary={"error": "Not enough clauses extracted"}
        )
    
    # Generate embeddings
    clause_texts = [c['text'] for c in all_clauses]
    embeddings = embedding_engine.generate_embeddings(clause_texts)
    
    # Detect inconsistencies
    inconsistency_results = contradiction_detector.detect_inconsistencies(
        all_clauses, embeddings, similarity_threshold=0.65
    )
    
    # Convert to response format
    comparison_results = []
    for res in inconsistency_results:
        comparison_results.append(ComparisonResult(
        clause_1=Clause(**res['clause_1']),
        clause_2=Clause(**res['clause_2']),
        similarity_score=float(res['similarity_score']),   # convert NumPy float
        is_inconsistent=bool(res['is_inconsistent']),      # convert NumPy bool
        inconsistency_type=res['inconsistency_type'],
        confidence=float(res['confidence']),               # convert NumPy float
        explanation=res['explanation']
        ))
    
    # Generate summary
    summary = {
        "total_comparisons": len(comparison_results),
        "inconsistencies_found": sum(1 for r in comparison_results if r.is_inconsistent),
        "consistent_pairs": sum(1 for r in comparison_results if not r.is_inconsistent),
        "avg_confidence": sum(r.confidence for r in comparison_results) / len(comparison_results) if comparison_results else 0
    }
    
    return DocumentAnalysisResponse(
        total_clauses_extracted=len(all_clauses),
        comparisons_made=len(comparison_results),
        results=comparison_results,
        summary=summary
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}