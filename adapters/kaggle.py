import os
import json
import pandas as pd

DATASET_PATH = "datasets/arxiv-metadata-oai-snapshot.json"
TARGET_SIZE_GB = 1  # Load approximately 1GB of data

def fetch_arxiv_data():
    """Loads ~1GB of ArXiv dataset efficiently from a local JSON file."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please run Kaggle API download first.")

    print("Reading dataset in chunks until ~1GB of data is loaded...")

    data = []
    total_size = 0

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                data.append(record)

                # Estimate memory size (each JSON object as string)
                total_size += len(line.encode("utf-8"))

                # Stop when ~1GB of data is reached
                if total_size >= TARGET_SIZE_GB * 1024 * 1024 * 1024:
                    break
            except json.JSONDecodeError:
                continue  # Skip corrupted lines

    df = pd.DataFrame(data)  # Convert to Pandas DataFrame

    print(f"Loaded {len(df)} records (~{total_size / (1024**3):.2f} GB).")
    return df

def process_arxiv_data(df):
    """Prepares Kaggle arXiv dataset for indexing."""

    df = df[['id', 'title', 'abstract', 'authors', 'categories', 'versions']].copy()  # Create a new DataFrame

    df.loc[:, 'year'] = df['versions'].apply(lambda v: v[0]['created'][:4])
    df.loc[:, 'pdf_url'] = df['id'].apply(lambda x: f"https://arxiv.org/pdf/{x}.pdf")

    df = df.drop(columns=['versions'])

    return df.dropna()
