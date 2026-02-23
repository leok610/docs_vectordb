import asyncio
import json
import os
import time
import lancedb
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Union, cast
import rich_click as click
from rich.console import Console
from rich.traceback import install as trace_install
from google import genai
from google.genai import types
import logging

# We use a try-except to support both relative and absolute imports
try:
    from .chunking_utils import get_timestamp, setup_shared_logging
except (ImportError, ValueError):
    from chunking_utils import get_timestamp, setup_shared_logging

trace_install()

project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
embedding_log = logs_dir / "embedding.log"
setup_shared_logging(embedding_log)

def print_log(message: str):
    """Logs a message to the shared embedding log."""
    logging.info(f"[GEMINI] {message}")

URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"
TABLE_NAME = "gemini_reference_docs"

def normalize_l2(embeddings: np.ndarray) -> List[List[float]]:
    """Normalizes embeddings to unit length."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    result = (embeddings / norms).tolist()
    return cast(List[List[float]], result)

class RateLimiter:
    def __init__(self, tpm_limit: int):
        self.tpm_limit = tpm_limit
        self.tokens_this_minute = 0
        self.minute_start = time.time()

    async def wait_if_needed(self, estimated_tokens: int):
        now = time.time()
        if now - self.minute_start > 60:
            self.minute_start = now
            self.tokens_this_minute = 0
            
        if self.tokens_this_minute + estimated_tokens > self.tpm_limit * 0.9:
            sleep_time = 60 - (now - self.minute_start) + 1
            if sleep_time > 0:
                print_log(f"Rate limit approaching. Sleeping for {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                self.minute_start = time.time()
                self.tokens_this_minute = 0
        
        self.tokens_this_minute += estimated_tokens

async def process_batch(
    client: genai.Client, 
    model: str, 
    batch_chunks: List[str], 
    config_args: Dict[str, Any], 
    rate_limiter: RateLimiter
) -> Optional[types.EmbedContentResponse]:
    """
    Sends a batch of text chunks to Gemini for embedding with exponential backoff.
    """
    estimated_tokens = sum(len(c) for c in batch_chunks) // 4 + 1
    
    max_retries = 5
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            await rate_limiter.wait_if_needed(estimated_tokens)
            
            response = await client.aio.models.embed_content(
                model=model,
                contents=cast(Any, batch_chunks),
                config=types.EmbedContentConfig(**config_args)
            )
            return response
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                delay = base_delay * (2 ** attempt)
                print_log(f"Quota exceeded (429). Retry {attempt+1}/{max_retries} in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print_log(f"Permanent error in process_batch: {e}")
                raise e
    
    print_log(f"Failed to process batch after {max_retries} attempts.")
    return None

async def embed_and_store_gemini(
    file_paths: List[Path], 
    model: str, 
    dimension: int, 
    tpm_limit: int
) -> Dict[str, Any]:
    """
    Core pipeline to read files, request embeddings, and store them in LanceDB.
    """
    client = genai.Client()
    db = lancedb.connect(uri=URI)
    
    # Pre-fetch existing source documents to allow resuming
    existing_sources = set()
    if TABLE_NAME in db.list_tables():
        tbl = db.open_table(TABLE_NAME)
        # We only need the source_doc column to check for existing files
        df = tbl.search().select(["source_doc"]).to_polars()
        existing_sources = set(df["source_doc"].unique())
        print_log(f"Found {len(existing_sources)} already processed documents.")

    rate_limiter = RateLimiter(tpm_limit)
    total_inserted = 0
    total_billable_chars = 0
    start_time = time.time()
    
    config_args: Dict[str, Any] = {
        "task_type": "RETRIEVAL_DOCUMENT",
        "title": "Documentation Chunk"
    }
    if dimension:
        config_args["output_dimensionality"] = dimension

    for file_path in file_paths:
        path_obj = Path(file_path)
        if not path_obj.exists():
            continue
            
        source_name = path_obj.stem.replace("_chunks", "").replace("_rst_chunks", "").replace("_md_chunks", "").replace("_txt_chunks", "")
        
        if source_name in existing_sources:
            # We don't have accurate char/token counts for skipped files here,
            # but we skip them to save time and quota.
            continue

        with path_obj.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        if not chunks:
            continue
            
        batch_size = 100
        tasks = []
        for i in range(0, len(chunks), batch_size):
            tasks.append(process_batch(client, model, chunks[i:i+batch_size], config_args, rate_limiter))
            
        responses = await asyncio.gather(*tasks)
        
        data = []
        file_chars = sum(len(c) for c in chunks)
        total_billable_chars += file_chars
        
        vector_count = 0
        for response in responses:
            if not response or not response.embeddings:
                continue
            
            vectors: List[List[float]] = [cast(List[float], emb.values) for emb in response.embeddings]
            if dimension and dimension < 3072:
                vectors = normalize_l2(np.array(vectors))
                
            for vec in vectors:
                data.append({
                    "id": f"{source_name}_{vector_count:04d}",
                    "source_doc": source_name,
                    "text": chunks[vector_count],
                    "vector": vec
                })
                vector_count += 1
                
        if data:
            if TABLE_NAME in db.list_tables():
                table = db.open_table(TABLE_NAME)
                table.add(data)
            else:
                try:
                    db.create_table(TABLE_NAME, data=data)
                except ValueError:
                    table = db.open_table(TABLE_NAME)
                    table.add(data)
            
            total_inserted += len(data)
            print_log(f"Processed {source_name}: {len(data)} vectors.")

    duration = time.time() - start_time
    return {
        "embedder": "gemini",
        "vectors_stored": total_inserted,
        "duration": duration,
        "billable_chars": total_billable_chars,
        "tokens_sent": int(total_billable_chars / 4)
    }

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("file_paths", nargs=-1, type=str)
@click.option("--model", default="models/gemini-embedding-001", help="Gemini model to use.")
@click.option("--dim", default=3072, type=int, help="Output dimensionality for embeddings.")
@click.option("--tpm", default=1000000, type=int, help="Tokens Per Minute rate limit.")
def main(file_paths: Sequence[str], model: str, dim: int, tpm: int):
    if not file_paths:
        return
        
    print_log(f"=== Starting Gemini Embedding Run: {get_timestamp()} ===")
    
    if len(file_paths) == 1 and file_paths[0].endswith(".json"):
        with open(file_paths[0], "r", encoding="utf-8-sig") as f:
            paths = [Path(p) for p in json.load(f)]
    else:
        paths = [Path(p) for p in file_paths]
        
    stats = asyncio.run(embed_and_store_gemini(paths, model, dim, tpm))
    
    import sys
    sys.stdout.write(json.dumps(stats))

if __name__ == "__main__":
    main()
