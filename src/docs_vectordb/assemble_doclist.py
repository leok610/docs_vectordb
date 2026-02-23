import json
import sys
from pathlib import Path
import rich_click as click
from rich.console import Console

console = Console(stderr=True)  # Send logs to stderr so stdout is pure JSON

REF_DIR = Path("C:/git-repositories/leok610/ref/doc")

# Valid text formats found in the reference repository
VALID_EXTENSIONS = [
    "rst", "md", "txt", "py", "html", "conf", 
    "css", "mmd", "ps1", "yaml", "inc", "js", "json", "xml"
]

@click.command()
@click.argument("extensions", nargs=-1, type=click.Choice(VALID_EXTENSIONS, case_sensitive=False))
def main(extensions):
    """
    Finds documentation files in the reference directory matching the provided extensions.
    Outputs a JSON array of absolute file paths.
    
    Example usage:
    assemble-doclist rst md txt > targets.json
    """
    
    if not REF_DIR.exists() or not REF_DIR.is_dir():
        console.print(f"[red]Error: Reference directory not found at {REF_DIR}[/red]")
        sys.exit(1)

    if not extensions:
        console.print("[yellow]No extensions provided. Defaulting to: rst, md, txt[/yellow]")
        extensions = ["rst", "md", "txt"]

    target_files = []
    
    for ext in extensions:
        # Find all files with this extension recursively
        for file_path in REF_DIR.rglob(f"*.{ext}"):
            if file_path.is_file():
                # Store paths as standard strings and replace backward slashes with forward ones
                # to avoid escape sequence issues in JSON
                target_files.append(str(file_path.absolute()).replace("\\", "/"))
                
    console.print(f"[green]Found {len(target_files)} matching files.[/green]")
    
    # Print the JSON array to standard output
    # This allows it to be piped into another file or script directly
    sys.stdout.write(json.dumps(target_files, indent=2))
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
