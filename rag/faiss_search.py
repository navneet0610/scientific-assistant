import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from core import settings

INDEX_DIR = os.path.join(settings.BASE_DIR, "faiss_index")

def search_faiss_index(query, top_k=5):
    """Search FAISS index using LangChain, retrieving full metadata for relevant papers."""

    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError("FAISS index not found. Run create_faiss_index() first.")

    # Load FAISS index
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)  #vectorstore - .index file

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=top_k)

    # Format results correctly with metadata
    response = []
    for result in results:
        metadata = result.metadata
        response.append({
            "title": metadata.get("title", "Unknown Title"),
            "authors": metadata.get("authors", "Unknown Authors"),
            "year": metadata.get("year", "Unknown Year"),
            "category": metadata.get("category", "Unknown Category"),
            "pdf_url": metadata.get("pdf_url", "N/A"),
            "abstract": result.page_content
        })

    return response
