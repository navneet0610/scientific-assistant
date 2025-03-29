import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("faiss_index.bin")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def search_arxiv(query, top_k=5):
    """Searches FAISS and returns papers with detailed metadata."""

    query_embedding = text_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results
