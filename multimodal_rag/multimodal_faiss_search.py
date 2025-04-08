import os
import torch
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Paths
BASE_DIR = r"D:\Tampere\scientific_assistant"
index_dir = os.path.normpath(os.path.join(BASE_DIR, "multimodal_rag", "faiss_index"))
index_path = os.path.join(index_dir, "index.faiss")
pkl_path = os.path.join(index_dir, "index.pkl")
image_dir = os.path.normpath(os.path.join(BASE_DIR, "multimodal_rag", "images", "images"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP
clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_model.eval()

# Custom embedding class
class ClipEmbeddings(Embeddings):
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def embed_documents(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

clip_embedder = ClipEmbeddings(model=clip_model, processor=clip_processor, device=device)

# Load FAISS index
faiss_index = FAISS.load_local(index_dir, clip_embedder, allow_dangerous_deserialization=True)

# Search functions
def print_results(results):
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(json.dumps(result.metadata, indent=4, ensure_ascii=False))

def search_text(query: str, k=5):
    results = faiss_index.similarity_search(query, k=k)
    all_results = []
    for i, result in enumerate(results):
        all_results.append(result.metadata)
    print(all_results)
    return all_results

def search_image(image_path: str, k=5):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs).cpu().numpy().tolist()[0]
    results = faiss_index.similarity_search_by_vector(img_emb, k=k)
    all_results = []
    for i, result in enumerate(results):
        all_results.append(result.metadata)
    return all_results

# Example usage
if __name__ == "__main__":
    # ---- Text query ----
    query_text = "quantum physics"
    print("\n--- Text Search Results ---")
    search_text(query_text, k=5)

#     # ---- Image query ----
#     query_image_path = "D:\Tampere\scientific_assistant\multimodal_rag\images\cond-mat0006165_3.jpg"
#     print("\n--- Image Search Results ---")
#     search_image(query_image_path, k=5)


