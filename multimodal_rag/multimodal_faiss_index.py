import os
import datasets
import numpy as np
import pickle
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Directories
data_dir = "/content/drive/My Drive/arxiv_data"
index_dir = "/content/drive/My Drive/faiss_index"
image_dir = os.path.join(index_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# Models
text_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
clip_model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to("cuda" if torch.cuda.is_available() else "cpu")


# Projection to 384-dim
def project_to_common_dim(embedding, target_dim=384):
    if embedding.shape[-1] > target_dim:
        return embedding[:, :target_dim]
    elif embedding.shape[-1] < target_dim:
        padding = torch.zeros((embedding.shape[0], target_dim - embedding.shape[-1]), device=embedding.device)
        return torch.cat([embedding, padding], dim=-1)
    return embedding


# Load dataset
base_url = "https://huggingface.co/datasets/MMInstruction/ArxivCap/resolve/main/data/"
data_files = {"train": [base_url + f"arXiv_src_{i:04d}_001.parquet" for i in range(1, 13)]}
hf_dataset = datasets.load_dataset("parquet", data_files=data_files, cache_dir=data_dir, split="train")
hf_iter = iter(hf_dataset)

# FAISS Indexing
batch_size = 30
faiss_index = None
metadata_store = []
tqdm_iter = tqdm(range(0, len(hf_dataset), batch_size))

for i in tqdm_iter:
    batch = [next(hf_iter) for _ in range(min(batch_size, len(hf_dataset) - i))]
    text_embeddings = []
    image_embeddings = []
    text_data = []

    for row in batch:
        text_content = f"{row['title']} {row['abstract']} {row['meta']['meta_from_kaggle'].get('journey', '')} {row['meta']['meta_from_kaggle'].get('license', '')} {row['meta']['meta_from_kaggle'].get('categories', '')} {row['meta']['meta_from_s2'].get('citationCount', 0)} {row['meta']['meta_from_s2'].get('publicationTypes', [])}"
        text_emb = torch.tensor(text_embedding_model.encode(text_content, convert_to_numpy=True), device="cuda")

        text_data.append(text_content)
        text_embeddings.append(text_emb)
        metadata_store.append({"type": "text", "content": text_content})

        for img_data in row.get("caption_images", []):
            for pair in img_data.get("cil_pairs", []):
                if isinstance(pair["image"], Image.Image) and "image_file" in pair:
                    image_filename = pair["image_file"].split("/")[-1]
                    image_path = os.path.join(image_dir, image_filename)
                    pair["image"].save(image_path)

                    caption = pair.get("caption", "")
                    image_text = f"{text_content} {caption}" if caption else text_content

                    caption_emb = torch.tensor(text_embedding_model.encode(image_text, convert_to_numpy=True),
                                               device="cuda")
                    text_embeddings.append(caption_emb)
                    text_data.append(image_text)

                    inputs = processor(images=pair["image"], return_tensors="pt").to("cuda")
                    img_emb = clip_model.get_image_features(**inputs).detach()
                    img_emb = project_to_common_dim(img_emb)
                    image_embeddings.append(img_emb)

                    metadata_store.append({"type": "image", "caption": caption, "image_path": image_path})

    text_embeddings = torch.stack(text_embeddings).cpu().numpy()
    image_embeddings = torch.cat(image_embeddings, dim=0).detach().cpu().numpy() if image_embeddings else np.empty(
        (0, text_embeddings.shape[1]))

    all_embeddings = np.vstack([text_embeddings, image_embeddings])
    embedding_pairs = list(zip(text_data, all_embeddings))

    if faiss_index is None:
        faiss_index = FAISS.from_embeddings(embedding_pairs, text_embedding_model)
    else:
        new_index = FAISS.from_embeddings(embedding_pairs, text_embedding_model)
        faiss_index.merge_from(new_index)

    del text_embeddings, image_embeddings, text_data, embedding_pairs, all_embeddings
    torch.cuda.empty_cache()
    tqdm_iter.set_description(f"Processed {i + batch_size} records")

faiss_index.save_local(index_dir)

# Save metadata to index.pkl
with open(os.path.join(index_dir, "index.pkl"), "wb") as f:
    pickle.dump(metadata_store, f)

print("FAISS index updated successfully!")
