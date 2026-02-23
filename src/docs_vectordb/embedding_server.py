import json
import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from waitress import serve

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-server")

# Load model globally (once)
logger.info("Loading SentenceTransformer model 'all-mpnet-base-v2'...")
model = SentenceTransformer("all-mpnet-base-v2")
logger.info("Model loaded successfully.")

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json()
    if not data or 'queries' not in data:
        return jsonify({"error": "Missing 'queries' in request body"}), 400
    
    queries = data['queries']
    embeddings = model.encode(queries).tolist()
    return jsonify({"embeddings": embeddings})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    # Standard Flask dev server for manual testing
    # Windows Service will call serve() directly
    serve(app, host="127.0.0.1", port=5000)
