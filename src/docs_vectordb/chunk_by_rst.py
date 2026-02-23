import asyncio
import time
import logging
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.traceback import install as trace_install
from chunking_utils import (
    split_long_unit, 
    write_chunks_to_json, 
    process_files_async, 
    process_files_sync, 
    load_targets,
    get_timestamp,
    setup_shared_logging
)

trace_install()

project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
chunking_log = logs_dir / "chunking.log"
setup_shared_logging(chunking_log)

def print_log(message: str):
    """Logs a message to the shared chunking log."""
    logging.info(f"[RST] {message}")

def is_underline(line: str) -> bool:
    """Checks if a line is a Sphinx header underline."""
    line = line.strip()
    if len(line) < 3:
        return False
    return len(set(line)) == 1 and line[0] in "=-~^*+#\"'."

def process_single_rst(file_path: Path, output_dir: Path) -> int:
    """
    Worker function to process a single .rst file.
    
    Side-effects:
        - Reads from `file_path`.
        - Writes to `{output_dir}/{file_path.stem}_rst_chunks.json` using `write_chunks_to_json`.
        
    Args:
        file_path (Path): Path to the source .rst document.
        output_dir (Path): Path to the destination directory for chunks.
        
    Returns:
        int: Total number of chunks written.
    """
    with file_path.open("r", encoding="utf-8", errors="ignore") as t:
        lines = [line.rstrip() for line in t]

    units = []
    current_unit: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        start_new_unit = False
        
        if line.lstrip().startswith(".."):
            start_new_unit = True
        elif i + 1 < len(lines) and is_underline(lines[i+1]):
            if line.strip():
                start_new_unit = True

        if start_new_unit:
            if current_unit and any(l.strip() for l in current_unit):
                units.append(current_unit)
            current_unit = [line]
        else:
            current_unit.append(line)
        i += 1

    if current_unit and any(l.strip() for l in current_unit):
        units.append(current_unit)

    MAX_LINES = 20
    OVERLAP = 5
    chunks_lists = []
    current_chunk_lines: list[str] = []
    
    processed_units = []
    for unit in units:
        if len(unit) > MAX_LINES:
            processed_units.extend(split_long_unit(unit, MAX_LINES, OVERLAP))
        else:
            processed_units.append(unit)
            
    for unit in processed_units:
        unit_length = len(unit)
        if len(current_chunk_lines) + unit_length > MAX_LINES and current_chunk_lines:
            chunks_lists.append(current_chunk_lines)
            current_chunk_lines = []
            
        current_chunk_lines.extend(unit)
        
    if current_chunk_lines:
        chunks_lists.append(current_chunk_lines)

    final_chunks = ["\n".join(chunk_lines).strip() for chunk_lines in chunks_lists]
    final_chunks = [c for c in final_chunks if c]

    output_path = output_dir / f"{file_path.stem}_rst_chunks.json"
    return write_chunks_to_json(final_chunks, output_path)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("file_paths", nargs=-1, type=str)
@click.option("--async-mode/--sync-mode", default=False, help="Run processing asynchronously")
def main(file_paths, async_mode):
    """
    Parses RST files into chunks. 
    Accepts a single .json list of paths or multiple raw paths.
    Outputs JSON logs to chunks_dir.
    """
    paths = load_targets(file_paths)
    if not paths:
        print_log("No files provided.")
        return

    print_log(f"=== Starting RST Chunking Run: {get_timestamp()} ===")
    print_log(f"Processing {len(paths)} files (Async={async_mode})...")
    
    chunks_dir = project_root / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    if async_mode:
        total_chunks = asyncio.run(process_files_async(process_single_rst, paths, chunks_dir))
    else:
        total_chunks = process_files_sync(process_single_rst, paths, chunks_dir)
        
    duration = time.time() - start_time
    print_log(f"Finished! Processed {len(paths)} files into {total_chunks} chunks in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
