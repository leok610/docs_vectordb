import json
from pathlib import Path
import rich_click as click
from rich import print
from rich.traceback import install as trace_install

trace_install()

def get_indent(line: str) -> int:
    """Returns the number of leading spaces on a line."""
    return len(line) - len(line.lstrip(" "))

def split_long_unit(unit: list[str], max_lines: int, overlap: int = 5) -> list[list[str]]:
    """Splits a single long glossary entry into multiple overlapping units."""
    if len(unit) <= max_lines:
        return [unit]
    
    split_units = []
    start = 0
    while start < len(unit):
        end = start + max_lines
        split_units.append(unit[start:end])
        if end >= len(unit):
            break
        # Advance by max_lines minus the overlap amount
        start += max_lines - overlap
    return split_units

@click.command()
@click.argument("file_path", type=click.Path(path_type=Path))
def main(file_path: Path):
    """Parses an RST glossary file into chunks and saves them to a JSON file."""
    
    target = file_path.absolute()
    
    if not target.exists():
        print(f"[red]Error: Target file not found at {target}[/red]")
        raise click.Abort()

    print(f"Reading target file: [cyan]{target}[/cyan]")

    # --- PASS 1: Group lines into glossary units ---
    glossary_units = []
    current_unit = []
    
    with target.open("r", encoding="utf-8") as t:
        for line in t:
            clean_text = line.strip()
            
            # Skip empty lines and RST directives/comments
            if not clean_text or clean_text.startswith(".."):
                continue
                
            indent = get_indent(line)
            
            if indent == 3:
                # It's exactly 3 spaces: A new Glossary Term
                if current_unit:
                    glossary_units.append(current_unit)
                current_unit = [clean_text] # Start a new unit with the term
                
            elif indent >= 6:
                # It's 6 or more spaces: A definition line belonging to the current term
                if current_unit is not None and len(current_unit) > 0:
                    current_unit.append(clean_text)
            
            # If indent is 0, we just ignore it (e.g., headers or main title)

        # Append the very last unit
        if current_unit:
            glossary_units.append(current_unit)

    print(f"Parsed [green]{len(glossary_units)}[/green] total glossary terms from the file.")

    # --- PASS 2: Pack units into chunks of ~20 lines max ---
    MAX_LINES = 20
    OVERLAP = 5
    chunks_lists = []
    current_chunk_lines = []
    
    # Pre-process glossary_units to split any that are individually too long
    processed_units = []
    for unit in glossary_units:
        if len(unit) > MAX_LINES:
            processed_units.extend(split_long_unit(unit, MAX_LINES, OVERLAP))
        else:
            processed_units.append(unit)
            
    for unit in processed_units:
        unit_length = len(unit)
        
        # If adding this unit exceeds the line limit AND the current chunk isn't empty:
        if len(current_chunk_lines) + unit_length > MAX_LINES and current_chunk_lines:
            chunks_lists.append(current_chunk_lines)
            current_chunk_lines = []
            
        current_chunk_lines.extend(unit)
        
    if current_chunk_lines:
        chunks_lists.append(current_chunk_lines)

    # Convert lists of lines into final joined strings
    final_chunks = [" ".join(chunk_lines) for chunk_lines in chunks_lists]

    print(f"Packed into [green]{len(final_chunks)}[/green] chunks (max {MAX_LINES} lines each).")

    # --- SAVE TO FILE ---
    # Create the chunks directory at the project root
    project_root = Path(__file__).parent.parent.parent
    chunks_dir = project_root / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    # Create an output filename based on the input file
    output_filename = f"{target.stem}_chunks.json"
    output_path = chunks_dir / output_filename
    
    print(f"Saving chunks to [cyan]{output_path.absolute()}[/cyan]...")
    
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(final_chunks, out, indent=2)
        
    print("[green]Success![/green]")

if __name__ == "__main__":
    main()
