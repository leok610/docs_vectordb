import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.traceback import install as trace_install
import time

trace_install()

project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
console = Console(file=open(logs_dir / "chunk_by_rst_async.log", "a", encoding="utf-8"))
def print(*args, **kwargs):
    console.print(*args, **kwargs)

def is_underline(line: str) -> bool:
    """Checks if a line is a Sphinx header underline."""
    line = line.strip()
    if len(line) < 3:
        return False
    return len(set(line)) == 1 and line[0] in "=-~^*+#\"'."

def split_long_unit(unit: list[str], max_lines: int, overlap: int = 5) -> list[list[str]]:
    """Splits a single long semantic unit into multiple overlapping units."""
    if len(unit) <= max_lines:
        return [unit]
    
    split_units = []
    start = 0
    while start < len(unit):
        end = start + max_lines
        split_units.append(unit[start:end])
        if end >= len(unit):
            break
        start += max_lines - overlap
    return split_units

# --- This function does the heavy CPU work for a SINGLE file ---
def process_single_file(file_path_str: str, chunks_dir_str: str) -> int:
    target = Path(file_path_str)
    chunks_dir = Path(chunks_dir_str)
    
    with target.open("r", encoding="utf-8") as t:
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

    if not final_chunks:
        return 0

    output_filename = f"{target.stem}_rst_chunks.json"
    output_path = chunks_dir / output_filename
    
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(final_chunks, out, indent=2)
        
    return len(final_chunks)

# --- This is our Async orchestrator for the chunks ---
async def process_all_files(file_paths: list[Path]):
    chunks_dir = project_root / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    loop = asyncio.get_running_loop()
    
    # We create a pool of worker processes (usually equal to your CPU cores)
    with ProcessPoolExecutor() as pool:
        tasks = []
        for path in file_paths:
            # We schedule the work on the background pool. 
            # This instantly returns a "Future" object.
            future = loop.run_in_executor(
                pool, 
                process_single_file, 
                str(path), 
                str(chunks_dir)
            )
            tasks.append(future)
            
        # We 'await' the gather function. This pauses process_all_files 
        # until ALL the futures in the list have completed and returned their results.
        results = await asyncio.gather(*tasks)
        
    return sum(results)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("file_paths", nargs=-1, type=click.Path(path_type=Path))
def main(file_paths):
    """Parses multiple RST files into chunks asynchronously. 
    Accepts a single .json file containing a list of paths, or multiple raw paths."""
    
    if not file_paths:
        click.echo("No files provided.")
        return

    # Check if the user passed a single JSON file containing a list of files
    if len(file_paths) == 1 and file_paths[0].suffix == ".json":
        with file_paths[0].open("r", encoding="utf-8-sig") as f:
            paths = [Path(p) for p in json.load(f)]
    else:
        paths = list(file_paths)

    click.echo(f"Starting async chunking for {len(paths)} files...")
    
    start_time = time.time()
    
    total_chunks = asyncio.run(process_all_files(paths))
    
    duration = time.time() - start_time
    click.echo(f"Finished! Processed {len(paths)} files into {total_chunks} chunks in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
