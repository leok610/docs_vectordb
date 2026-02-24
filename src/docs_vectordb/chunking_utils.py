import asyncio
import datetime
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, List, Tuple

def get_timestamp() -> str:
    """Returns a formatted timestamp string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def setup_shared_logging(log_file: Path):
    """Configures logging to write to a specific file with timestamps."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def split_long_unit(unit: List[str], max_lines: int, overlap: int = 5) -> List[List[str]]:
    """
    Splits a single long semantic unit into multiple overlapping units.
    
    This is a pure function with deterministic output.
    
    Args:
        unit (List[str]): A list of string lines representing a semantic unit.
        max_lines (int): The maximum number of lines per chunk.
        overlap (int): The number of lines to overlap between chunks.
        
    Returns:
        List[List[str]]: A list of chunked string lists.
    """
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

def write_chunks_to_json(chunks: List[str], output_path: Path, source_doc: str = "", program: str = "unknown") -> int:
    """
    Writes a list of chunk strings to a JSON file along with metadata.
    
    Side-effects:
        - Creates or overwrites the file at `output_path`.
        
    Args:
        chunks (List[str]): The text chunks to save.
        output_path (Path): The destination file path.
        source_doc (str): The name of the source document.
        program (str): The program name metadata.
        
    Returns:
        int: The number of chunks written. Returns 0 if the list is empty.
    """
    if not chunks:
        return 0
    data = {
        "source_doc": source_doc,
        "program": program,
        "chunks": chunks
    }
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(data, out, indent=2)
    return len(chunks)

async def process_files_async(worker_func: Callable[[Path, Path], int], file_paths: List[Path], output_dir: Path) -> int:
    """
    Executes a worker function across multiple files asynchronously using a ProcessPoolExecutor.
    
    Side-effects:
        - Spawns multiple CPU-bound processes.
        - Triggers the side-effects of `worker_func` (typically reading/writing files).
    
    Args:
        worker_func: A top-level function taking (file_path: Path, output_dir: Path) -> int.
        file_paths: A list of files to process.
        output_dir: The directory to pass to the worker function for output.
        
    Returns:
        int: The total sum of chunks generated across all files.
    """
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        tasks = [loop.run_in_executor(pool, worker_func, path, output_dir) for path in file_paths]
        results = await asyncio.gather(*tasks)
    return sum(results)

def process_files_sync(worker_func: Callable[[Path, Path], int], file_paths: List[Path], output_dir: Path) -> int:
    """
    Executes a worker function across multiple files synchronously (sequentially).
    
    Side-effects:
        - Triggers the side-effects of `worker_func` (typically reading/writing files).
    
    Args:
        worker_func: A top-level function taking (file_path: Path, output_dir: Path) -> int.
        file_paths: A list of files to process.
        output_dir: The directory to pass to the worker function for output.
        
    Returns:
        int: The total sum of chunks generated across all files.
    """
    total = 0
    for path in file_paths:
        total += worker_func(path, output_dir)
    return total

def load_targets(file_paths: Tuple[str, ...]) -> List[Path]:
    """
    Parses CLI path arguments into a flat list of Path objects.
    If a single JSON file is provided, reads the paths from it.
    
    Side-effects:
        - Reads from disk if a JSON file is provided.
        
    Args:
        file_paths (Tuple[str, ...]): A tuple of path strings from the CLI.
        
    Returns:
        List[Path]: A list of target file paths.
    """
    if not file_paths:
        return []
        
    if len(file_paths) == 1 and Path(file_paths[0]).suffix == ".json":
        with Path(file_paths[0]).open("r", encoding="utf-8-sig") as f:
            return [Path(p) for p in json.load(f)]
            
    return [Path(p) for p in file_paths]