import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from core import settings

INDEX_DIR = os.path.join(settings.BASE_DIR, "faiss_index")
METADATA_PATH = "metadata.json"

def create_faiss_index():
    """Creates FAISS index from metadata.json and stores embeddings."""

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    print("Loading metadata.json...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    documents = [
        Document(page_content=paper["abstract"], metadata={
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "category": paper["categories"],
            "pdf_url": paper["pdf_url"]
        })
        for paper in metadata
    ]

    print("Generating embeddings and creating FAISS index...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(INDEX_DIR)

    print(f"FAISS index created and stored in {INDEX_DIR}")

if __name__ == "__main__":
    create_faiss_index()
