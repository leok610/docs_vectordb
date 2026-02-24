import json
import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from waitress import serve

import rich_click as click

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-server")

# Lazy load model
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model 'all-mpnet-base-v2'...")
        _model = SentenceTransformer("all-mpnet-base-v2")
        logger.info("Model loaded successfully.")
    return _model

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json()
    if not data or 'queries' not in data:
        return jsonify({"error": "Missing 'queries' in request body"}), 400
    
    queries = data['queries']
    model = get_model()
    embeddings = model.encode(queries).tolist()
    return jsonify({"embeddings": embeddings})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@click.command()
@click.option("--port", default=5000, type=int, help="Port to run the server on.")
def main(port):
    # Standard Flask dev server for manual testing
    # Windows Service will call serve() directly
    logger.info(f"Starting embedding server on port {port}...")
    # Use threads=1 to ensure each process handles only one request at a time,
    # preventing internal resource contention. We scale via multiple processes instead.
    serve(app, host="127.0.0.1", port=port, threads=1)

if __name__ == "__main__":
    main()
