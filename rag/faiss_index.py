import os
import json
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

INDEX_DIR = "indexes/"
METADATA_PATH = "metadata.json"

def create_faiss_index():
    """Creates FAISS index from arXiv dataset and stores metadata properly."""

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    # Load metadata
    df = pd.read_json(METADATA_PATH)

    # Prepare data for FAISS indexing
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "title": row["title"],
            "authors": row["authors"],
            "year": row["year"],
            "category": row["categories"],
            "pdf_url": row["pdf_url"]
        }
        doc = Document(page_content=row["abstract"], metadata=metadata)
        documents.append(doc)

    # Create FAISS index with LangChain
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Save FAISS index
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index created and stored in {INDEX_DIR}")
