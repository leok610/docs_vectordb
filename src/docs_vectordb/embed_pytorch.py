import json
import os
import logging
import lancedb
import time
import sys
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, cast, Union, Tuple
import rich_click as click
from rich.console import Console
from rich.traceback import install as trace_install
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, SpinnerColumn, TimeRemainingColumn
from rich.live import Live
from rich.console import Group
from rich.text import Text
from rich.theme import Theme
from rich.panel import Panel

custom_theme = Theme({
    "markdown": "color(151)",
    "markdown.code": "color(33)",
    "markdown.emph": "italic color(179)",
    "markdown.strong": "color(35)",
    "markdown.header": "color(179)",
    "markdown.h1": "bold color(35)",
    "markdown.h2": "color(179)",
    "markdown.h3": "color(179)",
    "markdown.link": "color(179)",
    "status.cyan": "color(39)",
    "status.magenta": "color(72)",
    "dim": "color(103)",
    "info": "color(103)",
    "warning": "color(179)",
    "error": "bold color(131)",
    "header": "color(151)",
    "query": "bold color(103)"
})
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

def fetch_embeddings(url: str, chunks_batch: List[str], max_retries: int = 3) -> Tuple[Optional[List[List[float]]], float]:
    """Helper to call the single server and return results with timing and retries."""
    last_error = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            # Increased timeout to 300s to handle heavy load/large chunks
            res = requests.post(url, json={"queries": chunks_batch}, timeout=300)
            res.raise_for_status()
            return res.json()["embeddings"], time.time() - t0
        except (requests.exceptions.RequestException, Exception) as e:
            last_error = e
            print_log(f"Attempt {attempt + 1}/{max_retries} failed on {url}: {e}")
            time.sleep(1) # Brief pause before retry
    
    print_log(f"All {max_retries} attempts failed for a batch. Last error: {last_error}")
    return None, 0.0

def embed_and_store_pytorch(
    file_paths: List[Path], 
    force: bool, 
    batch_size: int, 
    port: int
) -> Dict[str, Any]:
    """Core pipeline for PyTorch embedding and storage using single worker with massive batching."""
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
    
    # Setup Server
    server_url = f"http://127.0.0.1:{port}/encode"
    health_url = server_url.replace("/encode", "/health")
    
    server_active = False
    try:
        requests.get(health_url, timeout=2).raise_for_status()
        server_active = True
        print_log(f"Connected to active server at {server_url}.")
    except:
        print_log(f"Server at {server_url} is unreachable.")
    
    model = None
    if not server_active:
        print_log("No active server found. Falling back to local SentenceTransformer.")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")

    total_inserted = 0
    start_time = time.time()
    embedding_time = 0.0
    storage_time = 0.0
    
    console = Console(theme=custom_theme, stderr=True, force_terminal=True, force_interactive=True, legacy_windows=False)
    
    # Flatten all chunks and track their metadata
    global_chunks: List[str] = []
    global_metadata: List[Dict[str, Any]] = []
    total_chunks = 0
    
    with console.status("[cyan]Scanning documentation files...[/cyan]"):
        for chunk_file in file_paths:
            if not chunk_file.exists(): continue
            try:
                with chunk_file.open("r", encoding="utf-8") as f_in:
                    chunk_data = json.load(f_in)
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
                for i, chunk in enumerate(chunks):
                    global_chunks.append(chunk)
                    global_metadata.append({
                        "source_doc": source_doc,
                        "program": program,
                        "index": i
                    })
                total_chunks += len(chunks)

    if total_chunks == 0:
        return {"embedder": "pytorch", "vectors_stored": 0, "duration": 0, "embedding_time": 0.0, "storage_time": 0.0}

    processed_chunks = 0
    pending_data: List[Dict[str, Any]] = []
    
    status_text = Text("Initializing...", style="status.cyan") 
    requests_completed = 0
    total_latency = 0.0
    telemetry_text = Text("Batch: 0 | RPS: 0.0 | Latency: 0ms", style="status.magenta")
    
    progress = Progress(
        SpinnerColumn(style="warning"),
        TextColumn("[header]Vectorizing"),
        BarColumn(bar_width=None, complete_style="markdown.strong", finished_style="markdown.header"),
        TextColumn("[dim]{task.completed}/{task.total}"),
        TaskProgressColumn(style="status.cyan"),
        TimeRemainingColumn(),
        console=console
    )
    ui_group = Group(status_text, telemetry_text, progress)
    panel = Panel(ui_group, title="[bold color(39)]PyTorch Embedding Pipeline[/bold color(39)]", border_style="color(103)")
    
    # Massive batching to maximize single-worker VRAM (e.g., 90% utilization)
    MAX_CHUNKS_PER_REQ = 2048
    
    max_batch_size = 0
    total_batch_size = 0
    batch_count = 0
    current_batch_size = 0

    with Live(panel, console=console, refresh_per_second=4, screen=True):
        task_id = progress.add_task("", total=total_chunks)
        start_telemetry_time = time.time()
        
        for batch_start in range(0, len(global_chunks), MAX_CHUNKS_PER_REQ):
            batch_end = min(batch_start + MAX_CHUNKS_PER_REQ, len(global_chunks))
            batch_texts = global_chunks[batch_start:batch_end]
            batch_meta = global_metadata[batch_start:batch_end]
            
            current_batch_size = len(batch_texts)
            max_batch_size = max(max_batch_size, current_batch_size)
            total_batch_size += current_batch_size
            batch_count += 1
            
            status_text.plain = f"Vectorizing batch of {current_batch_size} chunks..."
            
            embs = None
            t_spent = 0.0
            
            if server_active:
                embs, t_spent = fetch_embeddings(server_url, batch_texts)
            elif model:
                t0 = time.time()
                emb_res = model.encode(batch_texts, show_progress_bar=False)
                t_spent = time.time() - t0
                embs = emb_res.tolist()
            
            if embs:
                requests_completed += 1
                total_latency += t_spent
                embedding_time += t_spent
                
                # Reconstruct and queue for storage
                for i, vec in enumerate(embs):
                    meta = batch_meta[i]
                    s_doc = meta["source_doc"]
                    prog = meta["program"]
                    idx = meta["index"]
                    txt = batch_texts[i]
                    
                    row = {"id": f"{s_doc}_{idx:04d}", "source_doc": s_doc, "text": txt, "vector": vec}
                    if prog: row["program"] = prog
                    pending_data.append(row)
                
                processed_chunks += len(batch_texts)
                progress.update(task_id, completed=processed_chunks)
            else:
                print_log(f"Failure for a batch starting at index {batch_start}")
                continue
                
            # Database Store
            if len(pending_data) >= batch_size or (batch_end == len(global_chunks) and pending_data):
                status_text.plain = f"Storing {len(pending_data)} vectors..."
                t_store0 = time.time()
                try:
                    if table_exists:
                        table = db.open_table(TABLE_NAME)
                        table.add(pending_data)
                    else:
                        db.create_table(TABLE_NAME, data=pending_data)
                        table_exists = True
                    total_inserted += len(pending_data)
                    pending_data = []
                except Exception as e:
                    print_log(f"Database error: {e}")
                    pending_data = []
                storage_time += (time.time() - t_store0)

            # Telemetry Refresh
            if time.time() - start_telemetry_time >= 1.0:
                runtime = time.time() - start_time
                rps = requests_completed / runtime if runtime > 0 else 0
                avg_lat = (total_latency / requests_completed * 1000) if requests_completed > 0 else 0
                telemetry_text.plain = f"Batch: {batch_count} | Size: {current_batch_size} | RPS: {rps:.1f} | Latency: {int(avg_lat)}ms"
                start_telemetry_time = time.time()

    duration = time.time() - start_time
    avg_batch_size = total_batch_size / batch_count if batch_count > 0 else 0
    return {
        "embedder": "pytorch",
        "vectors_stored": total_inserted,
        "duration": duration,
        "embedding_time": embedding_time,
        "storage_time": storage_time,
        "max_batch_size": max_batch_size,
        "avg_batch_size": round(avg_batch_size, 1)
    }

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS, help="""
    Reads a JSON file containing chunk file paths, embeds as vectors using PyTorch (local model), 
    and stores in LanceDB.
    """)
@click.argument("file_paths_json", type=click.Path(path_type=Path, exists=True))
@click.option("--force", is_flag=True, help="Force rebuild by skipping existing document checks.")
@click.option("--batch-size", default=1000, help="Number of chunks to batch before storing to DB.")
@click.option("--port", default=5000, type=int, help="Port of the single embedding server.")
def main(file_paths_json: Path, force: bool, batch_size: int, port: int):
    print_log(f"=== Starting PyTorch Embedding Run: {get_timestamp()} ===")
    try:
        with file_paths_json.open("r", encoding="utf-8-sig") as f_in:
            file_paths = [Path(p) for p in json.load(f_in)]
    except Exception as e:
        print_log(f"Failed to load chunk list from {file_paths_json}: {e}")
        sys.exit(1)
        
    if not file_paths:
        print_log("No files to process.")
        sys.stdout.write(json.dumps({"embedder": "pytorch", "vectors_stored": 0, "duration": 0, "embedding_time": 0.0, "storage_time": 0.0}) + "\n")
        return

    stats = embed_and_store_pytorch(file_paths, force, batch_size, port)
    sys.stdout.write(json.dumps(stats) + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
