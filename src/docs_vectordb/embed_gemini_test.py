import asyncio
import json
import os
import time
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.traceback import install as trace_install
from google import genai
from google.genai import types

from typing import Optional, Any

trace_install()

project_root = Path(__file__).parent.parent.parent
console = Console()
def print(*args, **kwargs):
    console.print(*args, **kwargs)

async def process_chunk_file(chunk_file: Path, model: str, dimension: Optional[int] = None):
    """
    Reads a single JSON chunk file and embeds using the Gemini API asynchronously.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[red]Error: GEMINI_API_KEY environment variable is not set.[/red]")
        print("[yellow]Please set it using: $env:GEMINI_API_KEY='your-key'[/yellow]")
        return
        
    client = genai.Client()

    print(f"Reading chunks from: [cyan]{chunk_file}[/cyan]")

    with chunk_file.open("r", encoding="utf-8") as f:
        try:
            chunks = json.load(f)
        except json.JSONDecodeError:
            print(f"[red]Error: Failed to parse JSON in {chunk_file}[/red]")
            return
            
    if not chunks:
        print(f"[yellow]No chunks found in {chunk_file}[/yellow]")
        return
        
    print(f"Loaded {len(chunks)} chunks. Generating embeddings via Gemini API...")
    
    # Calculate local stats for immediate feedback
    local_chars = sum(len(c) for c in chunks)
    local_tokens = local_chars // 4
    
    batch_size = 100
    all_embeddings_objects = []
    
    start_time = time.time()
    
    # Create tasks for all batches in this file
    tasks = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # Build config
        config_args: dict[str, Any] = {
            "task_type": "RETRIEVAL_DOCUMENT",
            "title": "Documentation Chunk"
        }
        if dimension:
            config_args["output_dimensionality"] = dimension
            
        # Use the asynchronous client (client.aio)
        task = client.aio.models.embed_content(
            model=model,
            contents=batch_chunks,
            config=types.EmbedContentConfig(**config_args)
        )
        tasks.append(task)
        
    # Execute all batches for this file concurrently
    try:
        responses = await asyncio.gather(*tasks)
    except Exception as e:
         print(f"[red]API Error during embedding:[/red] {e}")
         return

    # Aggregate results
    for response in responses:
        if response.embeddings:
            all_embeddings_objects.extend(response.embeddings)

    duration = time.time() - start_time
    
    # Extract the actual vectors from the objects
    vectors = [emb.values for emb in all_embeddings_objects]

    print("\n[bold green]Embedding Test Complete![/bold green]")
    print(f"Total Chunks Processed: {len(vectors)}")
    if vectors and vectors[0] is not None:
        print(f"Vector Dimensions:     [yellow]{len(vectors[0])}[/yellow]")
    
    print("\n[bold cyan]Metrics:[/bold cyan]")
    print(f"- Total Time:           [yellow]{duration:.2f} seconds[/yellow]")
    print(f"- Billable Characters:  [yellow]{local_chars}[/yellow] (Estimated locally)")
    print(f"- Tokens Sent:          [yellow]{local_tokens}[/yellow] (Estimated locally, 4 chars/token)")

@click.command()
@click.argument("chunk_file", type=click.Path(path_type=Path, exists=True))
@click.option("--model", default="models/gemini-embedding-001", help="The embedding model to use")
@click.option("--dim", type=int, default=None, help="Output dimensionality (e.g. 768, 1536). Defaults to API default (3072).")
def main(chunk_file, model, dim):
    """
    Test script for Gemini async embeddings.
    """
    asyncio.run(process_chunk_file(chunk_file, model, dim))

if __name__ == "__main__":
    main()
