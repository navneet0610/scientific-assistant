# Voila! - Scientific Assistant - Django Project

This is a Django-based project for an AI based scientific assistant capable of retrieving text and graphical data for the textual and image query. 

Dataset - ArXivCap for multimodal retrieval

Framework - Langchain with FAISS 

This README provides steps for setting up the project on your local machine.


## Prerequisites

Before setting up the project, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)
- Git
- Django (and other dependencies listed in further steps to install)

## Getting Started

### 1. Clone the Repository

To get started, clone the project repository to your local machine using Git:




`git clone https://github.com/navneet0610/scientific-assistant.git`

`cd scientific-assitant`

## Install virtualenv if you don't have it
`pip install virtualenv`

## Create a virtual environment (you can name it as you like)
`virtualenv venv`

## Activate the virtual environment
### On Windows:
`venv\Scripts\activate`
### On macOS/Linux:
`source venv/bin/activate`

## Install Dependencies

`pip install datasets numpy tqdm langchain langchain_community faiss-cpu torch requests django`

## Set paths in project to point to your local directories
Inside `multimodal_faiss_search.py` set `BASE_DIR = r"your_dir\scientific-assistant"` - your_dir to point at the folder containing scientific_assistant

rest all paths will be set automatically.

## Run Server - start app

`python manage.py runserver`

# NOTES -

- FAISS Index - vector store is best created on a GPU with cuda through - `multimodal_faiss_index_colab.py`
- Downloaded index - `index.faiss` and pickle file with metadata - `index.pkl` to be placed inside dir `multimodal_rag/faiss_index/`
- `index.faiss` and `index.pkl` are already present inside dir `multimodal_rag/faiss_index/`
- `index.faiss` represent a vector store for text and image embeddings for 30 GBs of ArxivCap Data for multimodal retrieval.
- `index.pkl` contains metadata corresponding to the indexes in the vector store.
- Images extraction for showing in results fail even on cloud GPUs due to script running for long duration & memory shortages/overheads, images can be extracted through `extract_images_from_dataset_parquet.py` in `multimodal/static/images`



