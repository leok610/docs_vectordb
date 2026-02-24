import json
import lancedb
import os
from pathlib import Path
import rich_click as click
import numpy as np
from google import genai
from google.genai import types

project_root = Path(__file__).parent.parent.parent
URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"
GEMINI_TABLE = "gemini_reference_docs"
PYTORCH_TABLE = "reference_docs"

def normalize_l2(vector):
    """Normalizes a single vector to unit length."""
    v = np.array(vector)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()

def get_gemini_embedding(query, model="models/gemini-embedding-001"):
    client = genai.Client()
    config_args = {
        "task_type": "RETRIEVAL_QUERY",
        "output_dimensionality": 3072
    }
        
    response = client.models.embed_content(
        model=model,
        contents=[query],
        config=types.EmbedContentConfig(**config_args)
    )
    # The API returns a list of ContentEmbedding objects. We want the first one's values.
    vector = response.embeddings[0].values
    return vector

def get_pytorch_embedding(query):
    import requests
    import sys
    try:
        response = requests.post(
            "http://127.0.0.1:5000/encode",
            json={"queries": [query]},
            timeout=2 # Faster timeout for interactive search
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except Exception as e:
        # Report as warning in JSON for interpretation, but exit with error
        print(json.dumps({"warning": f"Embedding server connection failed: {e}. Start server with 'service-wrapper'." }))
        sys.exit(1)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS, help="""
    Performs a vector search across the documentation database.
    
    This tool takes a query string, generates an embedding using the specified backend, 
    and returns the most relevant documentation chunks in JSON format.
    """)
@click.argument("query")
@click.option("-n", default=5, help="Number of results to return")
@click.option("--embedder", type=click.Choice(["gemini", "pytorch"]), default="pytorch")
def main(query, n, embedder):
    db = lancedb.connect(uri=URI)
    response = db.list_tables()
    # Handle both list and ListTablesResponse object
    available_tables = response.tables if hasattr(response, 'tables') else response
    
    # Logic to select the best table based on requested embedder and availability
    selected_table = None
    if embedder == "gemini":
        if GEMINI_TABLE in available_tables:
            selected_table = GEMINI_TABLE
        elif PYTORCH_TABLE in available_tables:
            selected_table = PYTORCH_TABLE
    else:
        if PYTORCH_TABLE in available_tables:
            selected_table = PYTORCH_TABLE
            
    if not selected_table:
        print(json.dumps({"error": f"No suitable table found for {embedder}. Available: {available_tables}"}))
        return

    table = db.open_table(selected_table)
    
    # Get the correct query vector
    try:
        if embedder == "gemini":
            query_vector = get_gemini_embedding(query)
        else:
            query_vector = get_pytorch_embedding(query)

        # Search
        results = table.search(query_vector).limit(n).to_list()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return

    # Format for JSON output
    output = []
    for r in results:
        output.append({
            "distance": r.get("_distance", 0),
            "source": r.get("source_doc"),
            "chunk_id": r.get("id"),
            "text": r.get("text")
        })

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
