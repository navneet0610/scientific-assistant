# Voila! - Scientific Assistant - Django Project

This is a Django-based project for an AI based scientific assistant capable of retrieving text and graphical data for the textual and image query. 

Dataset - ArXivCap for multimodal retrieval and Kaggle Arxiv Scientific literature

Framework - Langchain with FAISS 

[![Watch the video]](https://github.com/navneet0610/scientific-assistant/blob/master/demo-with-neuron-image-and-query.mkv)

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

`pip install -r requirements.txt`

## Set paths in project to point to your local directories
Inside `multimodal_faiss_search.py` set `BASE_DIR = r"your_dir\scientific-assistant"` - your_dir to point at the folder containing scientific_assistant

rest all paths will be set automatically.

## Download and place the index.faiss and index.pkl - vectorstore/index file & the metadata file in the `multimodal_rag/faiss_index/`
- download [index.faiss](https://drive.google.com/file/d/1-1dYtlYJACsiTVZgLotU9l5b4IZ7vTzZ/view?usp=drive_link)
- download [index.pkl](https://drive.google.com/file/d/1-EIrRYdIh4yMLmcQKPOiYSidpH-ijiAb/view?usp=drive_link)
- it couldn't be pushed with Git LFS either due to no quota available

## Run Server - start app for backend

`python manage.py runserver`

## Run Streamlit Front-End UI

`streamlit run app.py`
- in your browser UI is accessible at - http://localhost:8501

# NOTES -

- FAISS Index is sourced through two datasets - [ArxivCap](https://huggingface.co/datasets/MMInstruction/ArxivCap) and [Kaggle Arxiv Scientific Literature](https://www.kaggle.com/datasets/Cornell-University/arxiv).
- ArxivCap index is best created on a GPU with cuda through - `multimodal_faiss_index_colab.py`
- to filter cs and ml category papers from the kaggle's arxiv literature -`adapters/filter_arxiv_cs_ml_category.py`
- Kaggle index is merged onto the previous one using - `multimodal_rag/new_index.py` 
- Downloaded index - `index.faiss` and pickle file with metadata - `index.pkl` to be placed inside dir `multimodal_rag/faiss_index/`
- `index.faiss` represent a vector store built over two datasets for 30 GBs of ArxivCap Data and 1.2GBs of kaggle Arxiv Scientific Literature data.
- `index.pkl` contains metadata corresponding to the indexes in the vector store.
- Images extraction - to download them locally to show in results. Images can be extracted through `extract_images_from_dataset_parquet.py` in `multimodal/static/images`



