import json
import os
import logging
import lancedb
import time
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.traceback import install as trace_install
from .chunking_utils import get_timestamp, setup_shared_logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

trace_install()

project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
embedding_log = logs_dir / "embedding.log"
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
    print_log(f"=== Starting PyTorch Embedding Run: {get_timestamp()} ===")
    
    with file_paths_json.open("r", encoding="utf-8-sig") as f:
        file_paths = [Path(p) for p in json.load(f)]
        
    if not file_paths:
        return
        
    db = lancedb.connect(uri=URI)
    table_exists = TABLE_NAME in db.list_tables()
    
    model = SentenceTransformer("all-mpnet-base-v2")
    
    total_inserted = 0
    start_time = time.time()
    
    for chunk_file in file_paths:
        if not chunk_file.exists():
            continue

        with chunk_file.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        if not chunks:
            continue
            
        embeddings = model.encode(chunks)
        
        data = []
        source_name = chunk_file.stem.replace("_chunks", "").replace("_rst_chunks", "").replace("_md_chunks", "").replace("_txt_chunks", "")
        
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            data.append({
                "id": f"{source_name}_{i:04d}",
                "source_doc": source_name,
                "text": chunk_text,
                "vector": embedding.tolist()
            })
            
        if table_exists:
            table = db.open_table(TABLE_NAME)
            table.add(data)
        else:
            db.create_table(TABLE_NAME, data=data)
            table_exists = True
            
        total_inserted += len(data)
        print_log(f"Processed {source_name}: {len(data)} vectors.")
        
    duration = time.time() - start_time
    
    if total_inserted == 0:
        raise RuntimeError("No vectors were successfully embedded or stored.")

    stats = {
        "embedder": "pytorch",
        "vectors_stored": total_inserted,
        "duration": duration,
        "tokens_sent": 0,
        "billable_chars": 0
    }
    import sys
    sys.stdout.write(json.dumps(stats))

if __name__ == "__main__":
    main()
