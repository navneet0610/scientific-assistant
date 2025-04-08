import os
import datasets
import requests
from typing import List
import json
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import re

# Directories
data_dir = "/content/drive/My Drive/arxiv_data"
index_dir = "/content/drive/My Drive/faiss_index"
image_dir = os.path.join(index_dir, "images")
os.makedirs(image_dir, exist_ok=True)
metadata_path = os.path.join(index_dir, "metadata.json")

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_model.eval()

class ClipEmbeddings(Embeddings):
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
clip_embedder = ClipEmbeddings(model=clip_model, processor=clip_processor, device=device)

file_names_to_load = [
    "arXiv_src_0001_001.parquet",
    "arXiv_src_0002_001.parquet",
    "arXiv_src_0101_001.parquet",
    "arXiv_src_0108_001.parquet",
    "arXiv_src_0203_001.parquet",
    "arXiv_src_0204_001.parquet",
    "arXiv_src_0201_001.parquet",
    "arXiv_src_0308_001.parquet",
    "arXiv_src_0309_001.parquet",
    "arXiv_src_0301_001.parquet",
    "arXiv_src_0407_001.parquet",
    "arXiv_src_0409_001.parquet",
    "arXiv_src_0502_001.parquet",
    "arXiv_src_0503_001.parquet",
    "arXiv_src_0712_001.parquet",
    "arXiv_src_0804_001.parquet",
    "arXiv_src_0807_001.parquet",
    "arXiv_src_0809_004.parquet",
    "arXiv_src_0801_001.parquet",
    "arXiv_src_0803_001.parquet",
    "arXiv_src_1202_007.parquet",
    "arXiv_src_0701_001.parquet",
    "arXiv_src_1110_001.parquet",
    "arXiv_src_1204_001.parquet",
    "arXiv_src_0801_001.parquet",
    "arXiv_src_0801_002.parquet",
    "arXiv_src_0801_003.parquet",
    "arXiv_src_0901_001.parquet",
    "arXiv_src_1103_007.parquet",
    "arXiv_src_1104_006.parquet"
]




# Load dataset
base_url = "https://huggingface.co/datasets/MMInstruction/ArxivCap/resolve/main/data/"
data_files = {"train": [base_url + file for file in file_names_to_load]}
hf_dataset = datasets.load_dataset("parquet", data_files=data_files, cache_dir=data_dir, split="train")
hf_iter = iter(hf_dataset)

# FAISS Indexing
batch_size = 30
faiss_index = None
tqdm_iter = tqdm(range(0, len(hf_dataset), batch_size))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\\\\", " ").replace("\\", " ") # Remove LaTeX line breaks like '\\'
    text = re.sub(r"\$.*?\$", "", text)  # Remove LaTeX math mode like $...$
    text = re.sub(r"\\[a-zA-Z]+(\{.*?\})?", "", text)  # Remove LaTeX commands like \vskip, \rightline, etc.
    text = re.sub(r"[{}]", "", text)  # Remove curly braces left behind
    text = re.sub(r"\s+", " ", text)   # Normalize whitespace
    return text.strip()

def get_clip_text_embedding(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)  # Shape: (batch_size, 512)
    return outputs

def get_clip_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)  # Shape: (batch_size, 512)
    return outputs


for i in tqdm_iter:
    batch = [next(hf_iter) for _ in range(min(batch_size, len(hf_dataset) - i))]

    for row in batch:
        title = clean_text(row["title"])
        abstract = clean_text(row["abstract"])
        text_content = f"{title} {abstract}"
        text_emb = get_clip_text_embedding(text_content)

        metadata_entry = {
            "arxiv_id": row.get("arxiv_id", ""),
            "title": row["title"],
            "abstract": row["abstract"],
            "journal": row['meta']['meta_from_kaggle'].get('journey', ''),
            "license": row['meta']['meta_from_kaggle'].get('license', ''),
            "categories": row['meta']['meta_from_kaggle'].get('categories', ''),
            "citationCount": row['meta']['meta_from_s2'].get('citationCount', 0),
            "publicationTypes": row['meta']['meta_from_s2'].get('publicationTypes', []),
            "caption": row.get("caption_images", [{}])[0].get("caption", ""),
            "images": []
        }
        image_texts = []
        img_embs = []
        for img_data in row.get("caption_images", []):
            for pair in img_data.get("cil_pairs", []):
                if isinstance(pair["image"], Image.Image) and "image_file" in pair:
                    image_filename = pair["image_file"].split("/")[-1]
                    image_path = os.path.join(image_dir, image_filename)
                    pair["image"].save(image_path)
                    caption = pair.get("sub_caption", "")
                    image_text = f"{text_content} {caption}" if caption else text_content
                    img_emb = get_clip_image_embedding(pair["image"])
                    img_embs.append(img_emb)
                    image_texts.append(image_text)

                    image = {
                        "image_name": image_filename,
                        "caption": caption
                    }
                    metadata_entry["images"].append(image)


        embedding_pair_text = [(text_content, text_emb.squeeze(0).cpu().numpy())]
        embedding_pair_image = list(zip(image_texts, [e.squeeze(0).cpu().numpy() for e in img_embs]))

        # creating indices
        faiss_index_text = FAISS.from_embeddings(embedding_pair_text, clip_embedder, metadatas=[metadata_entry])
        faiss_index_image = FAISS.from_embeddings(embedding_pair_image, clip_embedder, metadatas=[metadata_entry] * len(image_texts))
        new_index = faiss_index_text
        new_index.merge_from(faiss_index_image)

        if faiss_index is None:
            faiss_index = new_index
        else:
            faiss_index.merge_from(new_index)

        del image_text, text_content, embedding_pair_text, embedding_pair_image, text_emb, img_embs, image_texts

    torch.cuda.empty_cache()
    tqdm_iter.set_description(f"Processed {i + batch_size} records")

faiss_index.save_local(index_dir)


print("FAISS index updated successfully!")
