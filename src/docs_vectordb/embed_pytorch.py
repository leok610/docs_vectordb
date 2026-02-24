import json
import os
import logging
import lancedb
import time
import sys
import requests
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, cast, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from itertools import cycle
import rich_click as click
from rich.console import Console
from rich.traceback import install as trace_install
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, SpinnerColumn, TimeRemainingColumn
from rich.live import Live
from rich.console import Group
from rich.text import Text
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

def fetch_embeddings(url_pool_iterator: Iterator[str], chunks_batch: List[str], max_retries: int = 3) -> Tuple[Optional[List[List[float]]], float]:
    """Helper to call a server from the pool and return results with timing and retries."""
    last_error = None
    for attempt in range(max_retries):
        url = next(url_pool_iterator)
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
    ports: str
) -> Dict[str, Any]:
    """Core pipeline for PyTorch embedding and storage."""
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
    
    # Setup Server Pool
    port_list = [p.strip() for p in ports.split(",")]
    server_urls = [f"http://127.0.0.1:{p}/encode" for p in port_list]
    
    # Verify at least one server is up, or fallback
    active_urls = []
    for url in server_urls:
        health_url = url.replace("/encode", "/health")
        try:
            requests.get(health_url, timeout=2).raise_for_status()
            active_urls.append(url)
        except:
            print_log(f"Server at {url} is unreachable.")
    
    use_pool = len(active_urls) > 0
    url_pool_iter = cycle(active_urls) if use_pool else None
    
    model = None
    if not use_pool:
        print_log("No active servers found. Falling back to local SentenceTransformer.")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")
    else:
        print_log(f"Connected to {len(active_urls)} active servers.")

    total_inserted = 0
    start_time = time.time()
    embedding_time = 0.0
    storage_time = 0.0
    
    console = Console(stderr=True, force_terminal=True)
    work_items: List[Dict[str, Any]] = []
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
                work_items.append({
                    "chunks": chunks,
                    "source_doc": source_doc,
                    "program": program
                })
                total_chunks += len(chunks)

    if total_chunks == 0:
        return {"embedder": "pytorch", "vectors_stored": 0, "duration": 0, "embedding_time": 0.0, "storage_time": 0.0}

    processed_chunks = 0
    pending_data: List[Dict[str, Any]] = []
    
    status_text = Text("Initializing...", style="#00afff") 
    requests_completed = 0
    total_latency = 0.0
    telemetry_text = Text("Workers: 0 | RPS: 0.0 | Latency: 0ms", style="#d7af5f")
    
    progress = Progress(
        SpinnerColumn(style="#d7af5f"),
        TextColumn("[#afd7af]Vectorizing"),
        BarColumn(bar_width=None, complete_style="#00875f", finished_style="#afd7af"),
        TextColumn("[dim]{task.completed}/{task.total}"),
        TaskProgressColumn(style="#00afff"),
        TimeRemainingColumn(),
        console=console
    )
    ui_group = Group(status_text, telemetry_text, progress)
    MAX_CHUNKS_PER_REQ = 256

    with Live(ui_group, console=console, refresh_per_second=4, vertical_overflow="crop"):
        task_id = progress.add_task("", total=total_chunks)
        num_threads = len(active_urls) if use_pool else 1
        
        # Sliding Window / Throttle logic
        MAX_IN_FLIGHT = num_threads * 2 
        in_flight_futures: Dict[Future, Dict[str, Any]] = {}
        
        # Prepare all batches
        all_batches = []
        for item in work_items:
            s_doc = cast(str, item["source_doc"])
            prog = cast(Optional[str], item["program"])
            d_chunks = cast(List[str], item["chunks"])
            chunk_batches = [d_chunks[i:i + MAX_CHUNKS_PER_REQ] for i in range(0, len(d_chunks), MAX_CHUNKS_PER_REQ)]
            for cb in chunk_batches:
                all_batches.append({"source_doc": s_doc, "program": prog, "chunks": cb})

        batch_idx = 0
        doc_buffers: Dict[str, List[List[float]]] = {}
        doc_completion: Dict[str, int] = {}
        start_telemetry_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            while batch_idx < len(all_batches) or in_flight_futures:
                # 1. Fill the window
                while batch_idx < len(all_batches) and len(in_flight_futures) < MAX_IN_FLIGHT:
                    b = all_batches[batch_idx]
                    if use_pool and url_pool_iter:
                        f_net = executor.submit(fetch_embeddings, url_pool_iter, cast(List[str], b["chunks"]))
                        in_flight_futures[f_net] = b
                    elif model:
                        t0 = time.time()
                        # encode handles a list of strings
                        emb = model.encode(cast(List[str], b["chunks"]))
                        f_sync: Future = Future()
                        f_sync.set_result((emb.tolist(), time.time() - t0))
                        in_flight_futures[f_sync] = b
                    batch_idx += 1

                # 2. Collect finished results
                done_futures = [f for f in in_flight_futures if f.done()]
                for fut in done_futures:
                    meta = in_flight_futures.pop(fut)
                    s_doc = meta["source_doc"]
                    prog = meta["program"]
                    b_chunks = cast(List[str], meta["chunks"])
                    
                    try:
                        embs, t_spent = fut.result()
                        if embs:
                            if s_doc not in doc_buffers: doc_buffers[s_doc] = []
                            doc_buffers[s_doc].extend(embs)
                            doc_completion[s_doc] = doc_completion.get(s_doc, 0) + len(b_chunks)
                            
                            requests_completed += 1
                            total_latency += t_spent
                            embedding_time += t_spent
                            
                            # Find expected chunk count for this doc
                            expected = 0
                            for item in work_items:
                                if item["source_doc"] == s_doc:
                                    expected = len(item["chunks"])
                                    break
                            
                            if doc_completion[s_doc] >= expected:
                                status_text.plain = f"Vectorized: {s_doc}"
                                current_embs = doc_buffers.pop(s_doc)
                                original_chunks: List[str] = []
                                for item in work_items:
                                    if item["source_doc"] == s_doc:
                                        original_chunks = item["chunks"]
                                        break
                                
                                for i, (txt, vec) in enumerate(zip(original_chunks, current_embs)):
                                    row = {"id": f"{s_doc}_{i:04d}", "source_doc": s_doc, "text": txt, "vector": vec}
                                    if prog: row["program"] = prog
                                    pending_data.append(row)
                                
                                processed_chunks += expected
                                progress.update(task_id, completed=processed_chunks)
                                del doc_completion[s_doc]
                        else:
                            print_log(f"Failure for a batch of {s_doc}")
                    except Exception as e:
                        print_log(f"Error: {e}")

                # 3. Database Store (Serial)
                if len(pending_data) >= batch_size or (batch_idx >= len(all_batches) and not in_flight_futures and pending_data):
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

                # 4. Telemetry Refresh
                if time.time() - start_telemetry_time >= 1.0:
                    runtime = time.time() - start_time
                    rps = requests_completed / runtime if runtime > 0 else 0
                    avg_lat = (total_latency / requests_completed * 1000) if requests_completed > 0 else 0
                    telemetry_text.plain = f"Workers: {len(in_flight_futures)} in-flight | RPS: {rps:.1f} | Latency: {int(avg_lat)}ms"
                    start_telemetry_time = time.time()

                if not done_futures:
                    time.sleep(0.05)

    duration = time.time() - start_time
    return {
        "embedder": "pytorch",
        "vectors_stored": total_inserted,
        "duration": duration,
        "embedding_time": embedding_time,
        "storage_time": storage_time
    }

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS, help="""
    Reads a JSON file containing chunk file paths, embeds as vectors using PyTorch (local model), 
    and stores in LanceDB.
    """)
@click.argument("file_paths_json", type=click.Path(path_type=Path, exists=True))
@click.option("--force", is_flag=True, help="Force rebuild by skipping existing document checks.")
@click.option("--batch-size", default=1000, help="Number of chunks to batch before storing.")
@click.option("--ports", default="5000", help="Comma-separated list of ports for embedding servers.")
def main(file_paths_json: Path, force: bool, batch_size: int, ports: str):
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

    stats = embed_and_store_pytorch(file_paths, force, batch_size, ports)
    sys.stdout.write(json.dumps(stats) + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
