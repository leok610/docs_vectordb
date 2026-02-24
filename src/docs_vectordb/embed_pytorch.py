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
@click.option("--force", is_flag=True, help="Force rebuild by skipping existing document checks.")
@click.option("--batch-size", default=1000, help="Number of chunks to batch before embedding/storing.")
def main(file_paths_json, force, batch_size):
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
        sys.stdout.write(json.dumps({"embedder": "pytorch", "vectors_stored": 0, "duration": 0, "embedding_time": 0, "storage_time": 0}) + "\n")
        return
        
    db = lancedb.connect(uri=URI)
    
    # Pre-fetch existing source documents to allow resuming
    existing_sources = set()
    if not force:
        try:
            if TABLE_NAME in db.list_tables():
                tbl = db.open_table(TABLE_NAME)
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
    embedding_time = 0.0
    storage_time = 0.0
    
    # We use stderr=True and force_terminal so progress output doesn't interfere with the JSON stdout
    console = Console(stderr=True, force_terminal=True)
    
    # Pre-scan and group into batches
    work_items = []
    total_chunks = 0
    
    with console.status("[cyan]Scanning documentation files...[/cyan]") as status:
        for chunk_file in file_paths:
            if not chunk_file.exists():
                continue
            try:
                with chunk_file.open("r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
            except Exception as e:
                print_log(f"Failed to read {chunk_file}: {e}")
                continue

            if isinstance(chunk_data, list):
                chunks = chunk_data
                source_doc = chunk_file.stem.replace("_chunks", "").replace("_rst_chunks", "").replace("_md_chunks", "").replace("_txt_chunks", "")
                program = None
            else:
                chunks = chunk_data.get("chunks", [])
                source_doc = chunk_data.get("source_doc", "unknown")
                program = chunk_data.get("program")

            if chunks and (force or source_doc not in existing_sources):
                work_items.append({
                    "chunks": chunks,
                    "source_doc": source_doc,
                    "program": program
                })
                total_chunks += len(chunks)

    if total_chunks == 0:
        print_log("Nothing new to process.")
        sys.stdout.write(json.dumps({"embedder": "pytorch", "vectors_stored": 0, "duration": 0, "embedding_time": 0, "storage_time": 0}) + "\n")
        return

    processed_chunks = 0
    
    # Multi-file batching logic
    pending_data = []
    
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
    
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[dim]{task.completed}/{task.total}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console
    )

    with progress:
        task_id = progress.add_task("Processing...", total=total_chunks)
        for item in work_items:
            current_chunks = item["chunks"]
            source_doc = item["source_doc"]
            program = item["program"]

            # 1. Embedding
            progress.update(task_id, description=f"Embedding {source_doc}")
            
            t_emb0 = time.time()
            embeddings = None
            if use_server:
                try:
                    response = requests.post(server_url, json={"queries": current_chunks}, timeout=120)
                    response.raise_for_status()
                    embeddings = response.json()["embeddings"]
                except Exception as e:
                    print_log(f"Server error for {source_doc}: {e}. Falling back to local.")
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-mpnet-base-v2")
                    use_server = False
                    embeddings = model.encode(current_chunks)
            else:
                embeddings = model.encode(current_chunks)
            embedding_time += (time.time() - t_emb0)
            
            if embeddings is None: continue

            # 2. Alignment
            for i, (chunk_text, embedding) in enumerate(zip(current_chunks, embeddings)):
                vector = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                row = {
                    "id": f"{source_doc}_{i:04d}",
                    "source_doc": source_doc,
                    "text": chunk_text,
                    "vector": vector
                }
                if program: row["program"] = program
                pending_data.append(row)

            # 3. Batch Store
            if len(pending_data) >= batch_size or item == work_items[-1]:
                progress.update(task_id, description=f"Storing {source_doc}")
                t_store0 = time.time()
                try:
                    if table_exists:
                        table = db.open_table(TABLE_NAME)
                        try:
                            table.add(pending_data)
                        except Exception as e:
                            if "program" in str(e):
                                print_log("Schema mismatch. Stripping 'program' column.")
                                for r in pending_data: r.pop("program", None)
                                table.add(pending_data)
                            else: raise e
                    else:
                        db.create_table(TABLE_NAME, data=pending_data)
                        table_exists = True
                    
                    total_inserted += len(pending_data)
                    print_log(f"Committed batch: {len(pending_data)} vectors.")
                    pending_data = [] # Reset batch
                except Exception as e:
                    print_log(f"Critical error during batch store: {e}")
                    pending_data = []
                storage_time += (time.time() - t_store0)

            # Update completion status
            progress.update(task_id, completed=processed_chunks + len(current_chunks))
            processed_chunks += len(current_chunks)

    duration = time.time() - start_time
    
    stats = {
        "embedder": "pytorch",
        "vectors_stored": total_inserted,
        "duration": duration,
        "embedding_time": embedding_time,
        "storage_time": storage_time
    }
    sys.stdout.write(json.dumps(stats) + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
