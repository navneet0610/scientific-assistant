import os
import json
from typing import List
from langchain_core.embeddings import Embeddings
import gc
from langchain_community.vectorstores import FAISS
import torch
from transformers import CLIPProcessor, CLIPModel

# Directories
merged_index_dir = "/content/drive/My Drive/merged_index"
index_dir = "/content/drive/My Drive/faiss_index"
filtered_cs_ml_category_arxiv_papers = os.path.join(index_dir, "filtered_arxiv_cs_statml.json")

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_model.eval()

# Embedding Wrapper
class ClipEmbeddings(Embeddings):
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

clip_embedder = ClipEmbeddings(model=clip_model, processor=clip_processor, device=device)

# Load Existing Index
existing_faiss_index = FAISS.load_local(index_dir, embeddings=clip_embedder, allow_dangerous_deserialization=True)

# Helpers
def load_papers_in_chunks(file_path, batch_size):
    with open(file_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for i in range(0, len(metadata), batch_size):
        yield metadata[i : i + batch_size]

def get_clip_text_embedding(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)  # Shape: (batch_size, 512)
    return outputs

# Batch Processing
BATCH_SIZE = 1000

for batch_num, batch in enumerate(load_papers_in_chunks(filtered_cs_ml_category_arxiv_papers, BATCH_SIZE)):
    print(f"Processing batch {batch_num + 1} with {len(batch)} records...")

    text_contents = []
    text_embs = []
    metadatas = []

    for paper in batch:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text_content = f"{title} {abstract}"
        text_emb = get_clip_text_embedding([text_content])

        text_contents.append(text_content)
        text_embs.append(text_emb.squeeze(0).cpu().numpy())

        metadatas.append({
            "arxiv_id": paper.get("arxivid", ""),
            "title": title,
            "abstract": abstract,
            "authors": paper.get("authors", ""),
            "journal": paper.get("journal", ""),
            "license": paper.get("license", ""),
            "categories": paper.get("categories", "")
        })



    # Build one FAISS index per batch
    batch_index = FAISS.from_embeddings(
        [(text_contents[i], text_embs[i]) for i in range(len(text_contents))],
        embedding=clip_embedder,
        metadatas=metadatas
    )

    existing_faiss_index.merge_from(batch_index)

    # Clean up
    del batch, text_contents, text_embs, batch_index, metadatas
    torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection

# Save final index
existing_faiss_index.save_local(merged_index_dir)
print(f"FAISS index created and saved in {merged_index_dir}")
