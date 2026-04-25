from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from functools import lru_cache
import torch

class EmbeddingEngine:
    """Handles embedding generation and similarity computation"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize embedding model"""
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        # Batch processing for efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Normalize for cosine similarity
        norm1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        norm2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        similarity = np.dot(norm1, norm2)
        return float(np.clip(similarity, 0, 1))
    
    @lru_cache(maxsize=1000)
    def get_cached_embedding(self, text: str) -> bytes:
        """Cache embeddings for frequently seen texts"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tobytes()