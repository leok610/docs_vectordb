import json
import os
import logging
import lancedb
import time
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.traceback import install as trace_install
from docs_vectordb.chunking_utils import get_timestamp, setup_shared_logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

trace_install()

project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
embedding_log = logs_dir / "embed_pytorch.log"
setup_shared_logging(embedding_log)

def print_log(message: str):
    """Logs a message to the shared embedding log."""
    logging.info(f"[PYTORCH] {message}")

URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"
TABLE_NAME = "reference_docs"

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS, help="""
    Reads a JSON file containing chunk file paths, embeds as vectors using PyTorch (local model), 
    and stores in LanceDB.
    """)
@click.argument("file_paths_json", type=click.Path(path_type=Path, exists=True))
def main(file_paths_json):
    # Ensure sys is available for stdout
    import sys
    
    print_log(f"=== Starting PyTorch Embedding Run: {get_timestamp()} ===")
    
    # Load targets
    try:
        with file_paths_json.open("r", encoding="utf-8-sig") as f:
            file_paths = [Path(p) for p in json.load(f)]
    except Exception as e:
        print_log(f"Failed to load chunk list from {file_paths_json}: {e}")
        sys.exit(1)
        
    if not file_paths:
        print_log("No files to process.")
        sys.stdout.write(json.dumps({"embedder": "pytorch", "vectors_stored": 0, "duration": 0}) + "\n")
        return
        
    db = lancedb.connect(uri=URI)
    
    # Pre-fetch existing source documents to allow resuming
    existing_sources = set()
    try:
        if TABLE_NAME in db.list_tables():
            tbl = db.open_table(TABLE_NAME)
            # Safely get existing sources using Polars
            df = tbl.to_polars().select(["source_doc"]).unique()
            existing_sources = set(df["source_doc"].to_list())
            print_log(f"Found {len(existing_sources)} already processed documents.")
    except Exception as e:
        print_log(f"Warning: Could not check existing documents: {e}")

    table_exists = TABLE_NAME in db.list_tables()
    
    import requests
    server_url = "http://127.0.0.1:5000/encode"
    use_server = False
    
    try:
        requests.get("http://127.0.0.1:5000/health", timeout=2).raise_for_status()
        print_log("Connected to local embedding server.")
        use_server = True
    except Exception as e:
        print_log(f"Server not available: {e}. Falling back to local SentenceTransformer.")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")
    
    total_inserted = 0
    start_time = time.time()
    
    for chunk_file in file_paths:
        if not chunk_file.exists():
            continue

        try:
            with chunk_file.open("r", encoding="utf-8") as f:
                chunk_data = json.load(f)
        except Exception as e:
            print_log(f"Failed to read {chunk_file}: {e}")
            continue
            
        # Handle formats
        if isinstance(chunk_data, list):
            chunks = chunk_data
            source_doc = chunk_file.stem.replace("_chunks", "").replace("_rst_chunks", "").replace("_md_chunks", "").replace("_txt_chunks", "")
            program = None
        else:
            chunks = chunk_data.get("chunks", [])
            source_doc = chunk_data.get("source_doc", "unknown")
            program = chunk_data.get("program")
            
        if not chunks:
            continue

        if source_doc in existing_sources:
            continue
            
        embeddings = None
        if use_server:
            try:
                response = requests.post(server_url, json={"queries": chunks}, timeout=120)
                response.raise_for_status()
                embeddings = response.json()["embeddings"]
            except Exception as e:
                print_log(f"Error calling embedding server for {source_doc}: {e}. Skipping server for this file.")
                # Fallback to local if server fails mid-run
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-mpnet-base-v2")
                use_server = False
                embeddings = model.encode(chunks)
        else:
            embeddings = model.encode(chunks)
        
        if embeddings is None:
            continue

        data = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            # If embedding is a numpy array (from local model), convert to list
            if hasattr(embedding, "tolist"):
                vector = embedding.tolist()
            else:
                vector = embedding

            row = {
                "id": f"{source_doc}_{i:04d}",
                "source_doc": source_doc,
                "text": chunk_text,
                "vector": vector
            }
            if program:
                row["program"] = program
            data.append(row)
            
        try:
            if table_exists:
                table = db.open_table(TABLE_NAME)
                try:
                    table.add(data)
                except Exception as e:
                    if "program" in str(e) and any("program" in r for r in data):
                        print_log("Schema mismatch on 'program' column. Retrying without it.")
                        for r in data: r.pop("program", None)
                        table.add(data)
                    else:
                        raise e
            else:
                db.create_table(TABLE_NAME, data=data)
                table_exists = True
            
            total_inserted += len(data)
            print_log(f"Processed {source_doc}: {len(data)} vectors.")
        except Exception as e:
            print_log(f"Critical error storing data for {source_doc}: {e}")
            # We don't exit here to try to finish other files, but we log it
        
    duration = time.time() - start_time
    
    stats = {
        "embedder": "pytorch",
        "vectors_stored": total_inserted,
        "duration": duration,
        "tokens_sent": 0,
        "billable_chars": 0
    }
    sys.stdout.write(json.dumps(stats) + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
