import json
import lancedb
from pathlib import Path
import rich_click as click
from rich import print
from rich.traceback import install as trace_install
from sentence_transformers import SentenceTransformer

trace_install()

# Delay model loading until inside the main function to avoid slow startups
# if someone just wants to run --help
uri = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"
db = lancedb.connect(uri=uri)

@click.command()
@click.argument("file_path", type=click.Path(path_type=Path))
def main(file_path: Path):
    """Reads a JSON chunk file, embeds as vectors, and stores in LanceDB."""
    
    chunk_file = file_path.absolute()
    
    if not chunk_file.exists():
        print(f"[red]Error: Target file not found at {chunk_file}[/red]")
        raise click.Abort()

    print(f"Reading chunks from: [cyan]{chunk_file}[/cyan]")

    # Load chunks from JSON
    with chunk_file.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"Loaded [green]{len(chunks)}[/green] chunks. Loading embedding model...")
    
    # Initialize the model right before we need it
    model = SentenceTransformer("all-mpnet-base-v2")

    print(f"Generating embeddings using device: [yellow]{model.device}[/yellow]...")
    
    # We pass the entire list of chunks to encode() to process them efficiently in a batch
    embeddings = model.encode(chunks)
    
    print("Preparing data for LanceDB insertion...")
    
    data = []
    # Use the source filename to keep track of where these chunks came from
    source_name = chunk_file.stem.replace("_chunks", "")
    
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        data.append({
            "id": f"{source_name}_{i:04d}",
            "source_doc": source_name,
            "text": chunk_text,
            "vector": embedding.tolist() # Convert NumPy array to Python list
        })
        
    # Using the source name as the table name (e.g., 'glossary')
    table_name = source_name
    
    print(f"Storing {len(data)} records in LanceDB table: [cyan]'{table_name}'[/cyan]...")
    
    # Create or overwrite the table
    db.create_table(table_name, data=data, mode="overwrite")
    
    print("[green]Insertion complete![/green]")
    print(f"You can now query the [cyan]'{table_name}'[/cyan] table at [yellow]{uri}[/yellow]")

if __name__ == "__main__":
    main()
