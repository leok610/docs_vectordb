import json
import os
import logging
import lancedb
import rich_click as click
from rich.traceback import install as trace_install

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

trace_install()

URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"
TABLE_NAME = "glossary"

@click.command()
@click.argument("query", type=str)
@click.option(
    "-n",
    "--top-n",
    type=int,
    default=3,
    show_default=True,
    help="Number of chunks to return.",
)
def main(query: str, top_n: int):
    """Searches the LanceDB vector database for the given query silently."""
    
    try:
        db = lancedb.connect(uri=URI)
        table = db.open_table(TABLE_NAME)
    except Exception as e:
        print(json.dumps({"error": f"Error connecting to LanceDB or opening table '{TABLE_NAME}': {e}"}))
        raise click.Abort()

    model = SentenceTransformer("all-mpnet-base-v2")
    query_vector = model.encode(query).tolist()
    
    results = table.search(query_vector).limit(top_n).select(["id", "source_doc", "text", "_distance"]).to_polars()
        
    if len(results) == 0:
        print(json.dumps([]))
        return
        
    output_results = []
    for i, row in enumerate(results.iter_rows(named=True)):
        output_results.append({
            "distance": row.get("_distance", 0.0),
            "source": row.get("source_doc", "Unknown"),
            "chunk_id": row.get("id", "Unknown"),
            "text": row.get("text", "").strip()
        })
        
    print(json.dumps(output_results, indent=2))

if __name__ == "__main__":
    main()
