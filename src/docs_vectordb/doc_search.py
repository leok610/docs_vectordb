import os
import json
import sys
import time
import subprocess
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.markdown import Markdown
from google import genai
from google.genai import types

# Constants
MODEL_ID = "gemini-3-flash-preview"
URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"

console = Console()

# The system prompt derived from the skill definition
SYSTEM_PROMPT = """You are a technical documentation assistant. 
Your goal is to answer questions based on the provided documentation context.
If the context is inadequte, help the user to adjust the query or to search
with ripgrep on the local help files, which are complete sets for different
programs.

## Instructions
1. Be concise and answer quickly.
2. Provide code examples if present in the context.
3. Cite your sources (file names provided in context).
4. If the answer isn't in the context, suggest a better query.
5. Your output is for human consumption in a CLI environment. Use Markdown.

## Formatting
Return your answer in clear, well-formatted Markdown.
"""

def get_context(query, top_n=5, embedder="pytorch"):
    """Calls doc-retrieval to get relevant chunks."""
    try:
        # We call doc-retrieval as a subprocess to leverage the existing CLI interface
        cmd = ["doc-retrieval", query, "-n", str(top_n), "--embedder", embedder]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        console.print(f"[red]Error retrieving context:[/red] {e}")
        return []

@click.command()
@click.argument("query")
@click.option("-n", "--top-n", default=5, help="Number of context chunks to retrieve.")
@click.option("--embedder", type=click.Choice(["gemini", "pytorch"]), default="pytorch", help="Embedding backend to use.")
@click.option("--raw", is_flag=True, help="Output raw markdown instead of rich-rendered text.")
def main(query, top_n, embedder, raw):
    """
    Search documentation and get an AI-generated answer using Gemini Flash.
    Fast alternative to full Gemini CLI skill interaction.
    """
    start_time = time.time()
    
    # 1. Retrieve Context
    with console.status("[cyan]Retrieving context...[/cyan]"):
        context_chunks = get_context(query, top_n, embedder)
    
    if not context_chunks:
        console.print("[yellow]No relevant documentation found.[/yellow]")
        return

    # 2. Prepare Prompt
    context_text = "\n\n".join([
        f"SOURCE: {c['source']}\nCHUNK_ID: {c['chunk_id']}\nCONTENT:\n{c['text']}"
        for c in context_chunks
    ])
    
    user_prompt = f"USER QUERY: {query}\n\nDOCUMENTATION CONTEXT:\n{context_text}"

    # 3. Call Gemini Flash
    try:
        with console.status("[magenta]Generating answer...[/magenta]"):
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                )
            )
    except Exception as e:
        console.print(f"[red]API Error:[/red] {e}")
        sys.exit(1)

    end_time = time.time()
    latency = end_time - start_time

    # 4. Output
    answer = response.text
    
    if raw:
        print(answer)
    else:
        console.print(Markdown(answer))
        console.print(f"\n[dim italic]Latency: {latency:.2f}s | Context Chunks: {len(context_chunks)}[/dim italic]")

if __name__ == "__main__":
    main()
