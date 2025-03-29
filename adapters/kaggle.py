import kagglehub
import pandas as pd


def fetch_arxiv_data():
    """Fetches arXiv dataset from Kaggle dynamically."""

    df = kagglehub.load_dataset(
        kagglehub.KaggleDatasetAdapter.PANDAS,
        "Cornell-University/arxiv",
        "arxiv-metadata-oai-snapshot.json"
    )

    return pd.DataFrame(df)


def process_arxiv_data(df):
    """Prepares Kaggle arXiv dataset for indexing."""

    df = df[['id', 'title', 'abstract', 'authors', 'categories', 'versions']]

    df['year'] = df['versions'].apply(lambda v: v[0]['created'][:4])

    df['pdf_url'] = df['id'].apply(lambda x: f"https://arxiv.org/pdf/{x}.pdf")

    df = df.drop(columns=['versions'])

    return df.dropna()
