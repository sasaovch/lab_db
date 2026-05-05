# rag/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('intfloat/multilingual-e5-base')

def embed_text(text: str) -> list[float]:
    # E5 требует префикс для passage/query
    return model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

def embed_query(query: str) -> list[float]:
    return model.encode(f"query: {query}", normalize_embeddings=True).tolist()