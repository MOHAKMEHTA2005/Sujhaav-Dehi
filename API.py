import os
from flask import Flask, request, render_template
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import requests

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
INDEX_NAME = "list"
EMBEDDING_API_URL = os.getenv(
    "EMBEDDING_API_URL",
    "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

es = Elasticsearch([ELASTICSEARCH_URL], basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD))
app = Flask(__name__)


def create_embedding(text):
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    response = requests.post(
        EMBEDDING_API_URL,
        headers=headers,
        json={"inputs": text},
        timeout=60,
    )
    response.raise_for_status()
    embedding = response.json()
    if embedding and isinstance(embedding[0], list):
        embedding = embedding[0]
    if not isinstance(embedding, list) or len(embedding) != 384:
        raise RuntimeError("Embedding service returned an unexpected vector")
    return embedding

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    if request.method == 'POST':
        user_input = request.form['input']
        user_embedding = create_embedding(user_input)

        query = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": user_embedding}
                    }
                }
            }
        }

        response = es.search(index=INDEX_NAME, body=query)
        results = [hit["_source"] for hit in response["hits"]["hits"]]

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)