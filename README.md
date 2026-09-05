# Sujhaav Dehi

Sujhaav Dehi is a Flask web application for semantic product search. It creates vector embeddings for product titles with `sentence-transformers/all-MiniLM-L6-v2`, stores them in Elasticsearch, and returns the five most relevant products for a search phrase.

## How It Works

1. `upload_data.py` creates the `list` Elasticsearch index and its product mapping.
2. Product titles from `Noise.csv` are converted to 384-dimensional embeddings.
3. Product records and embeddings are indexed in Elasticsearch.
4. `API.py` embeds each submitted search phrase and ranks products with cosine similarity.
5. `templates/index.html` displays the matching products and prices.

## Requirements

- Python 3.10 or newer
- An Elasticsearch deployment with credentials
- Internet access the first time the SentenceTransformer model is downloaded

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file from `.env.example` and set the Elasticsearch connection values:

```text
ELASTICSEARCH_URL=https://your-deployment.region.gcp.elastic-cloud.com:443
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=replace-with-your-password
```

Never commit `.env` or share its password. The file is ignored by Git.

## Load the Data

With Elasticsearch reachable and `.env` configured, create the index and upload the CSV data:

```powershell
python upload_data.py
```

The script expects `Noise.csv` in the project root. It creates an index named `list` with a 384-dimensional `dense_vector` field. Running it again uploads the records again, so remove or recreate the index first when a clean reload is needed.

## Run the Web App

Start Flask:

```powershell
python API.py
```

Open <http://127.0.0.1:5000/> in a browser and enter a product description. The development server runs with Flask debug mode enabled; use a production WSGI server and deployment configuration before exposing the application publicly.

## Project Structure

```text
.
├── API.py                 # Flask search application
├── upload_data.py         # Elasticsearch index and CSV loader
├── Noise.csv              # Product source data
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── templates/index.html   # Search page
```

## Notes

- The embedding model produces 384-dimensional vectors, matching the Elasticsearch mapping in `upload_data.py`.
- The CSV loader uses the `ISO-8859-1` encoding and removes scraper-specific columns before indexing.
- The application currently expects the CSV's renamed fields: `Title`, `Rating`, `Price after Discount`, `MRP`, and `Delivery By`.
