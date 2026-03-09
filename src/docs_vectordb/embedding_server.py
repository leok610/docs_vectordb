import json
import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from waitress import serve

import rich_click as click

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("embedding-server")

logger.info("Loading SentenceTransformer model 'all-mpnet-base-v2'...")
model = SentenceTransformer("C:/git-repositories/forks/all-mpnet-base-v2")
logger.info("Model loaded successfully.")

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json()
    if not data or 'queries' not in data:
        return jsonify({"error": "Missing 'queries' in request body"}), 400
    
    queries = data['queries']
    batch_size = len(queries)
    if queries and queries[0] != "primer":
        logger.info(f"Received batch of {batch_size} queries.")
    embeddings = model.encode(queries, show_progress_bar=False).tolist()
    return jsonify({"embeddings": embeddings})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@click.command()
@click.option("--port", default=5000, type=int, help="Port to run the server on.")
def main(port):
    logger.info(f"Starting embedding server on port {port}")
    serve(app, host="127.0.0.1", port=port, threads=4)

if __name__ == "__main__":
    main()
