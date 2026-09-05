import os
from flask import Flask, request, render_template
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
INDEX_NAME = "list"

es = Elasticsearch([ELASTICSEARCH_URL], basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD))
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    if request.method == 'POST':
        user_input = request.form['input']
        user_embedding = model.encode(user_input).tolist()

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