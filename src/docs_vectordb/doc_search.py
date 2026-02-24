import os
import json
import sys
import time
import subprocess
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from google import genai
from google.genai import types

# Constants
MODEL_ID = "gemini-2.5-flash-lite"
URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"

"""
COLOR REFERENCE TABLE
| Theme Key             | Index | Rich Name          | Hex Code |
|-----------------------|-------|--------------------|----------|
| markdown, header      | 151   | dark_sea_green_3d  | #afd7af  |
| markdown.h1, .strong  | 35    | spring_green_4     | #00875f  |
| markdown.h2, .h3      | 179   | light_goldenrod_3  | #d7af5f  |
| markdown.code         | 33    | dodger_blue_1      | #0087ff  |
| markdown.link         | 179   | light_goldenrod_3  | #d7af5f  |
| status.cyan           | 39    | deep_sky_blue_1    | #00afff  |
| status.magenta        | 72    | dark_cyan          | #5f8787  |
| dim, info, query      | 103   | slate_gray_3       | #8787af  |
| error                 | 131   | indian_red         | #af5f5f  |
"""
custom_theme = Theme({
    "markdown": "color(151)",
    "markdown.code": "color(33)",
    "markdown.emph": "italic color(179)",
    "markdown.strong": "color(35)",
    "markdown.header": "color(179)",
    "markdown.h1": "bold color(35)",
    "markdown.h2": "color(179)",
    "markdown.h3": "color(179)",
    "markdown.link": "color(179)",
    "status.cyan": "color(39)",
    "status.magenta": "color(72)",
    "dim": "color(103)",
    "info": "color(103)",
    "warning": "color(179)",
    "error": "bold color(131)",
    "header": "color(151)",
    "query": "bold color(103)"
})
console = Console(theme=custom_theme, width=80)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

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
    except subprocess.CalledProcessError as e:
        try:
            # Try to parse the JSON error/warning from stdout
            err_data = json.loads(e.stdout)
            msg = err_data.get("error") or err_data.get("warning") or e.stderr
            console.print(f"[bold color(131)]Retrieval Error:[/bold color(131)] {msg}")
        except:
            console.print(f"[bold color(131)]Retrieval Failed:[/bold color(131)] {e.stderr or str(e)}")
        return []
    except Exception as e:
        console.print(f"[bold color(131)]System Error:[/bold color(131)] {e}")
        return []

@click.command(context_settings=CONTEXT_SETTINGS)
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
        console.print(Markdown(answer, code_theme="sata-dark"))
        console.print(f"\n[dim italic]Latency: {latency:.2f}s | Context Chunks: {len(context_chunks)}[/dim italic]")

if __name__ == "__main__":
    main()
