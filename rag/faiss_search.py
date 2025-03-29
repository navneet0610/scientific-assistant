import os
import json
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index"
METADATA_PATH = "metadata.json"


def search_faiss_index(query, top_k=5):
    """Searches the FAISS index for the most relevant scientific papers."""

    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("FAISS index or metadata file not found. Please run create_faiss_index() first.")

    # Load FAISS index
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(INDEX_PATH, embedding_model)

    # Perform similarity search
    results = vector_db.similarity_search(query, k=top_k)

    # Load metadata efficiently
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Convert metadata to a dictionary for fast lookup using the abstract as the key
    metadata_dict = {paper["abstract"]: paper for paper in metadata}

    # Retrieve additional details from metadata
    response = []
    for result in results:
        paper = metadata_dict.get(result.page_content)  # Fast dictionary lookup
        if paper:
            response.append({
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "category": paper["categories"],
                "pdf_url": paper["pdf_url"],
                "abstract": paper["abstract"]
            })

    return response
