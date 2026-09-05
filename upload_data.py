import os
import pandas as pd
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
INDEX_NAME = "list"
CSV_FILE_PATH = "Noise.csv"

def create_index():
    es = Elasticsearch([ELASTICSEARCH_URL], basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD))

    mapping = {
        "mappings": {
            "properties": {
                "Title": {"type": "text"},
                "Rating": {"type": "keyword"},
                "Price after Discount": {"type": "keyword"},
                "MRP": {"type": "keyword"},
                "Delivery By": {"type": "keyword"},
                "embedding": {"type": "dense_vector", "dims": 384}
            }
        }
    }

    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=mapping)
        print(f"Index '{INDEX_NAME}' created successfully!")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def upload_csv_to_elasticsearch():
    es = Elasticsearch([ELASTICSEARCH_URL], basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD))
    data = pd.read_csv(CSV_FILE_PATH, encoding='ISO-8859-1')
    data = data.fillna("")

    cols = ["a-link-normal href", "s-image src", "s-label-popover href", "a-color-secondary", "a-color-base",
            "a-popover-preload", "a-link-normal href 2", "a-link-normal", "a-link-normal href 3", "a-popover-trigger href",
            "a-link-normal href 4", "a-size-base", "a-size-base href", "a-offscreen", "a-price-symbol", "a-row",
            "a-row 2", "a-row 3", "a-badge-text"]
    data = data.drop(columns=cols)

    data = data.rename(columns={'a-size-medium': 'Title',
                                'a-icon-alt': 'Rating',
                                'a-price-whole': 'Price after Discount',
                                'a-offscreen 2': 'MRP',
                                'a-text-bold': 'Delivery By'})

    data['MRP'] = data['MRP'].str.replace('ýýý', '')

    data["embedding"] = data["Title"].apply(lambda x: model.encode(x).tolist())

    for i, row in data.iterrows():
        doc = row.to_dict()
        es.index(index=INDEX_NAME, id=i, document=doc)

    print("CSV data uploaded successfully!")

if __name__ == "__main__":
    upload_csv_to_elasticsearch()