import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from adapters.kaggle import fetch_arxiv_data, process_arxiv_data

text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_faiss_index():
    """Fetches arXiv data from Kaggle, generates embeddings, and creates FAISS index."""

    raw_df = fetch_arxiv_data()
    df = process_arxiv_data(raw_df)

    abstracts = df['abstract'].tolist()
    embeddings = text_model.encode(abstracts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "faiss_index.bin")
    df.to_json("metadata.json", orient="records")

    print(f"FAISS index created with {len(df)} papers!")


# Django does not automatically run faiss_index.py when you start the server.

# FAISS indexing takes time, so we don't want it running every time the server starts.

# Instead, you can manually trigger it only when needed - in the following cases
# 1. When first setting up the project (to generate faiss_index.bin).
# 2. Whenever new papers are added to the dataset.
# 3. If you delete or modify the FAISS index."""

# Run it manually:
# python manage.py shell
# from rag.faiss_index import create_faiss_index
# create_faiss_index()
