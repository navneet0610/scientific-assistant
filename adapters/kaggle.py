import os
import json
import pandas as pd
from datetime import datetime

DATASET_PATH = "datasets/arxiv-metadata-oai-snapshot.json"
METADATA_PATH = "metadata.json"
TARGET_SIZE_GB = 1  # Load approx. 1GB of data


def fetch_arxiv_data():
    """Loads ~1GB of ArXiv dataset efficiently from a local JSON file."""
    print("Current Working Directory:", os.getcwd())

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please download it first.")

    print("Reading dataset in chunks until ~1GB of data is loaded...")

    data = []
    total_size = 0

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                data.append(record)

                total_size += len(line.encode("utf-8"))  # Estimate memory usage

                if total_size >= TARGET_SIZE_GB * 1024 * 1024 * 1024:
                    break  # Stop when ~1GB is loaded
            except json.JSONDecodeError:
                continue  # Skip corrupted lines

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records (~{total_size / (1024 ** 3):.2f} GB).")
    return df


def extract_year(versions):
    """Safely extracts the year from the 'versions' field."""
    if isinstance(versions, list) and len(versions) > 0 and "created" in versions[0]:
        try:
            # Convert 'created' string to datetime object and extract the year
            return datetime.strptime(versions[0]["created"], "%a, %d %b %Y %H:%M:%S %Z").year
        except ValueError:
            return "Unknown"  # Return "Unknown" if parsing fails
    return "Unknown"  # Default if year is missing


def process_arxiv_data(df):
    """Processes Kaggle arXiv dataset and saves it as metadata.json."""

    # Select relevant fields
    df = df[['id', 'title', 'abstract', 'authors', 'categories', 'versions']].dropna()

    # Extract year safely
    df.loc[:, 'year'] = df['versions'].apply(extract_year)

    # Generate PDF URL
    df.loc[:, 'pdf_url'] = df['id'].apply(lambda x: f"https://arxiv.org/pdf/{x}.pdf")

    # Remove unnecessary columns
    df = df.drop(columns=['versions']).dropna()

    # Convert to JSON format and save
    metadata = df.to_dict(orient="records")

    # Validate JSON structure before saving
    try:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {METADATA_PATH}.")
    except Exception as e:
        print(f"Error saving metadata: {e}")


if __name__ == "__main__":
    df = fetch_arxiv_data()
    process_arxiv_data(df)
