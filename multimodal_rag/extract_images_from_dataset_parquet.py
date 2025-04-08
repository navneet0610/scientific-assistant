import os
from datasets import Dataset
import requests


# Constants
BASE_DIR = r"D:\Tampere\scientific_assistant"
BASE_URL = "https://huggingface.co/datasets/MMInstruction/ArxivCap/resolve/main/data/"
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "multimodal_rag", "static", "images"))

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

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def download_image(image, image_filename):
    """
    Saves the image object to the specified path.
    """
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    image.save(image_path)
    print(f"Saved image: {image_filename}")


def process_parquet_file(parquet_file_path):
    """
    Loads the Parquet file and processes each row to extract and save images.
    """
    # Load the dataset from the Parquet file
    dataset = Dataset.from_parquet(parquet_file_path)

    # Process each row in the dataset
    for row in dataset:
        for img_data in row.get("caption_images", []):
            for pair in img_data.get("cil_pairs", []):
                if "image" in pair and "image_file" in pair:
                    # Extract the image filename
                    image_filename = pair["image_file"].split("/")[-1]

                    # Get the image (PIL Image object)
                    image = pair["image"]

                    # Save the image to the disk
                    download_image(image, image_filename)


def delete_parquet_file(parquet_file_path):
    """
    Deletes the specified Parquet file from the local file system.
    """
    if os.path.exists(parquet_file_path):
        os.remove(parquet_file_path)
        print(f"Deleted {parquet_file_path} to free up memory.")


def download_and_process_parquet_file(parquet_file):
    """
    Downloads a Parquet file, processes it, and deletes it after processing.
    """
    # Download the Parquet file to a temporary location
    parquet_file_url = BASE_URL + parquet_file
    temp_parquet_file_path = os.path.join("temp", parquet_file)

    if not os.path.exists("temp"):
        os.makedirs("temp")

    response = requests.get(parquet_file_url, stream=True)
    with open(temp_parquet_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    print(f"Downloaded {parquet_file}...")

    # Process the Parquet file
    process_parquet_file(temp_parquet_file_path)

    # Delete the Parquet file to free memory
    delete_parquet_file(temp_parquet_file_path)


# Process each Parquet file in batch
for parquet_file in file_names_to_load:
    print(f"Processing {parquet_file}...")
    download_and_process_parquet_file(parquet_file)

print("All images have been processed and Parquet files deleted.")
print("All images have been processed.")
