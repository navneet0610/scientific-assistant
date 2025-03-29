from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
import pandas as pd

def create_faiss_index():
    """Creates FAISS index for ArXiv papers."""
    print("Loading processed dataset...")
    df = pd.read_json("processed_arxiv.json")

    print("Generating text embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embeddings = embedding_model.embed_documents(df["abstract"].tolist())

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings, dtype=np.float32))

    print("Saving FAISS index...")
    faiss.write_index(index, "faiss_index.bin")

    print("Saving metadata...")
    df.to_json("metadata.json", orient="records")

    print("FAISS index creation complete!")
