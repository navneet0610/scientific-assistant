import json
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

METADATA_PATH = "metadata.json"
INDEX_DIR = "../faiss_index"
BATCH_SIZE = 50000  # Process in batches of 50,000 documents

# Ensure metadata exists
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

print("Loading metadata.json in chunks...")

# Load JSON file line by line to reduce memory usage
def load_metadata_in_chunks(file_path, batch_size):
    with open(file_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for i in range(0, len(metadata), batch_size):
        yield metadata[i : i + batch_size]

# Initialize embedding model once
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index in batches
faiss_index = None

for batch_num, metadata_batch in enumerate(load_metadata_in_chunks(METADATA_PATH, BATCH_SIZE)):
    print(f"Processing batch {batch_num + 1} with {len(metadata_batch)} records...")

    documents = [
        Document(
            page_content=paper["abstract"],
            metadata={
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "category": paper["categories"],
                "pdf_url": paper["pdf_url"]
            }
        )
        for paper in metadata_batch
    ]

    if faiss_index is None:
        faiss_index = FAISS.from_documents(documents, embedding_model)
    else:
        new_index = FAISS.from_documents(documents, embedding_model)
        faiss_index.merge_from(new_index)  # Merge new index into existing one

# Save the final FAISS index
faiss_index.save_local(INDEX_DIR)
print(f"FAISS index created and saved in {INDEX_DIR}")
