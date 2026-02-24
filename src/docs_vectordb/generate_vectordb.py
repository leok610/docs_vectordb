import json
import shutil
import lancedb
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.traceback import install as trace_install
import rich_click as click

trace_install()
console = Console()
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / "src" / "docs_vectordb"

def run_script(script_name, *args):
    """Runs a python script as a subprocess. Captures stdout."""
    script_path = src_dir / script_name
    cmd = [sys.executable, str(script_path)] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Error running {script_name}:[/red]\n{result.stderr}")
        return None
    return result.stdout.strip()

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS, help="""
    Orchestrates the generation of a vector database from documentation files.
    
    This script performs three main phases:
    1. Assembly: Recursively finds all target .rst, .md, and .txt files.
    2. Chunking: Parses documents into overlapping semantic units based on structure.
    3. Embedding: Generates vectors using Gemini (3072-dim) or PyTorch (768-dim) and stores them in LanceDB.
    
    The resulting database is stored in the project's database/ directory.
    """)
@click.option("--embedder", type=click.Choice(["gemini", "pytorch"]), default="gemini", help="Backend to use for embeddings.")
def main(embedder):
    start_run = time.time()
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    master_log = logs_dir / "orchestrator.log"
    
    with master_log.open("a", encoding="utf-8") as f:
        f.write(f"\n=== Run Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def log_master(msg):
        with master_log.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    console.print(f"[bold blue]Starting Vector DB Generation (Embedder: {embedder})[/bold blue]")
    log_master(f"Starting run with embedder: {embedder}")
    
    # 0. Clean up previous runs
    chunks_dir = project_root / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(exist_ok=True)
    
    URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"
    TABLE_NAME = "gemini_reference_docs" if embedder == "gemini" else "reference_docs"
    
    try:
        db = lancedb.connect(uri=URI)
        response = db.list_tables()
        available_tables = response.tables if hasattr(response, 'tables') else response
        if TABLE_NAME in available_tables:
            if embedder == "gemini":
                console.print(f"[cyan]Table '{TABLE_NAME}' exists. Resuming ingestion...[/cyan]")
            else:
                db.drop_table(TABLE_NAME)
                console.print(f"[yellow]Dropped existing table: '{TABLE_NAME}'[/yellow]")
    except Exception as e:
        console.print(f"[red]Could not handle table initialization:[/red] {e}")

    # 1. Assemble Document List
    t0 = time.time()
    with console.status("[cyan]Assembling doclist...[/cyan]"):
        doclist_json = run_script("assemble_doclist.py", "rst", "md", "txt")
        if not doclist_json:
            raise RuntimeError("Phase 1 Failed: Failed to assemble document list.")
        target_files = json.loads(doclist_json)
    
    if not target_files:
        raise RuntimeError("Phase 1 Failed: No target documents found.")

    t_assemble = time.time() - t0
    console.print(f"[green]Found {len(target_files)} files to process.[/green]")
    log_master(f"Assembled {len(target_files)} files in {t_assemble:.2f}s")
    
    # 2. Chunking
    t0 = time.time()
    rst_files, md_files, txt_files, glossary_files = [], [], [], []
    for f in target_files:
        p = Path(f)
        if p.name == "glossary.rst": glossary_files.append(f)
        elif p.suffix == ".rst": rst_files.append(f)
        elif p.suffix == ".md": md_files.append(f)
        elif p.suffix == ".txt": txt_files.append(f)

    temp_targets_dir = project_root / "temp_targets"
    temp_targets_dir.mkdir(exist_ok=True)
    
    target_groups = {
        "chunk_by_indents.py": ("glossary_targets.json", glossary_files),
        "chunk_by_rst.py": ("rst_targets.json", rst_files),
        "chunk_by_md.py": ("md_targets.json", md_files),
        "chunk_by_txt.py": ("txt_targets.json", txt_files),
    }

    for script, (jname, flist) in target_groups.items():
        if not flist: continue
        tpath = temp_targets_dir / jname
        with tpath.open("w", encoding="utf-8-sig") as f: json.dump(flist, f)
        with console.status(f"[cyan]Chunking {len(flist)} files with {script}...[/cyan]"):
            chunk_output = run_script(script, str(tpath), "--async-mode")
            if chunk_output is None:
                raise RuntimeError(f"Phase 2 Failed: Chunking script {script} failed.")
    
    t_chunk = time.time() - t0
    shutil.rmtree(temp_targets_dir)
    log_master(f"Chunking completed in {t_chunk:.2f}s")

    # 3. Embedding & Storing
    t0 = time.time()
    chunk_files = list(chunks_dir.glob("*_chunks.json"))
    if not chunk_files:
        raise RuntimeError("Phase 2 Failed: No chunk files were generated.")

    console.print(f"[green]Generated {len(chunk_files)} chunk files.[/green]")
    
    # Create a JSON list of chunk files to avoid shell argument length limits
    list_path = chunks_dir / "all_chunk_files.json"
    with list_path.open("w", encoding="utf-8-sig") as f:
        json.dump([str(p) for p in chunk_files], f)

    with console.status(f"[cyan]Embedding and storing into LanceDB...[/cyan]"):
        if embedder == "gemini":
            raw_stats = run_script("embed_gemini.py", str(list_path))
        else:
            # PyTorch now also uses the JSON list
            raw_stats = run_script("embed_pytorch.py", str(list_path))
            
    if not raw_stats:
        raise RuntimeError("Phase 3 Failed: Embedding script failed to produce stats output.")

    t_embed = time.time() - t0
    log_master(f"Embedding completed in {t_embed:.2f}s")

    # 3.5 Indexing
    t0 = time.time()
    with console.status("[cyan]Creating vector indexes for high-speed search...[/cyan]"):
        db = lancedb.connect(URI)
        table_to_index = "gemini_reference_docs" if embedder == "gemini" else "reference_docs"
        
        response = db.list_tables()
        available_tables = response.tables if hasattr(response, 'tables') else response
        
        if table_to_index in available_tables:
            table = db.open_table(table_to_index)
            # Use cosine for Gemini/MPNET, partitions ~ sqrt(N)
            table.create_index(
                metric="cosine", 
                num_partitions=64, 
                num_sub_vectors=96 if embedder == "gemini" else 32
            )
            log_master(f"Index created for {table_to_index}")
        else:
            log_master(f"Skipping indexing: table {table_to_index} not found in {available_tables}")
        
    t_index = time.time() - t0
    log_master(f"Indexing completed in {t_index:.2f}s")

    # 4. Summary
    total_time = time.time() - start_run
    
    stats = {}
    try:
        # Find the last JSON block in case there's logging output before it
        start_idx = raw_stats.rfind('{')
        if start_idx != -1:
            stats = json.loads(raw_stats[start_idx:])
        else:
            stats = json.loads(raw_stats)
    except Exception as e:
        console.print(f"[red]Error parsing embedding stats:[/red] {e}")
        console.print(f"Raw output: {raw_stats}")
        raise RuntimeError(f"Phase 3 Failed: Could not parse embedding results: {e}")
    
    if not stats or stats.get("vectors_stored", 0) == 0:
        raise RuntimeError("Phase 3 Failed: No vectors were stored in the database.")

    summary = Table(title="Run Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="yellow")
    summary.add_row("Total Files", str(len(target_files)))
    summary.add_row("Total Chunks", str(len(chunk_files)))
    summary.add_row("Vectors Stored", str(stats.get("vectors_stored", 0)))
    summary.add_row("Tokens Sent (est)", str(stats.get("tokens_sent", 0)))
    summary.add_row("Billable Chars", str(stats.get("billable_chars", 0)))
    summary.add_row("Phase: Assembly", f"{t_assemble:.2f}s")
    summary.add_row("Phase: Chunking", f"{t_chunk:.2f}s")
    summary.add_row("Phase: Embedding", f"{t_embed:.2f}s")
    summary.add_row("Phase: Indexing", f"{t_index:.2f}s")
    summary.add_row("Total Runtime", f"{total_time:.2f}s")
    
    console.print(summary)
    log_master(f"TOTAL RUNTIME: {total_time:.2f}s. Vectors: {stats.get('vectors_stored')}")
    console.print("[bold green]Success! View logs/orchestrator.log for details.[/bold green]")

if __name__ == "__main__":
    main()
