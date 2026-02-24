import os
import json
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from enum import Enum
import rich_click as click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from google import genai
from google.genai import types
from rich.theme import Theme


# Constants
class GeminiModel(str, Enum):
    FLASH_25 = "gemini-2.5-flash"
    LITE_25 = "gemini-2.5-flash-lite"
    FLASH_3 = "gemini-3-flash-preview"
    PRO_3 = "gemini-3-pro-preview"

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
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHATS_DIR = PROJECT_ROOT / "chats"

SYSTEM_PROMPT = """You are an expert technical assistant. Your goal is to help the user understand their documentation and navigate the application effectively.

## Strategy:
1. **Primary Source**: Use the provided LOCAL DOCUMENTATION CONTEXT as the definitive source for facts about the specific project. Cite the file names.
2. **Supplemental Knowledge**: Use your general programming knowledge to explain broad concepts or fill in gaps, but clearly distinguish between "Project-Specific" (from context) and "General Knowledge".
3. **Self-Service Focus**: Teach the user HOW to find this information himself next time (e.g., naming relevant files or search terms).
4. **Interactive**: This is a conversation. Keep responses snappy and helpful.

## Formatting:
- Use Markdown.
- Keep it concise.
- Cite local source files.
"""


def get_context(query, top_n=5, embedder="gemini"):
    """Calls doc-retrieval to get relevant chunks."""
    try:
        cmd = ["doc-retrieval", query, "-n", str(top_n), "--embedder", embedder]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    # TODO: Specify exceptions
    except Exception as e:
        return e


def save_history(history):
    """Serializes and saves chat history to JSON."""
    CHATS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = CHATS_DIR / f"chat_{timestamp}.json"

    # Convert Content objects to serializable dicts
    serializable_history = []
    for entry in history:
        serializable_history.append({
            "role": entry.role,
            "text": entry.parts[0].text if entry.parts else ""
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2)

    console.print(f"[bold green]History saved to {filename}[/bold green]")
    return filename


def load_history(filepath):
    """Loads history from JSON and converts to types.Content list."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = []
    for entry in data:
        history.append(types.Content(
            role=entry["role"],
            parts=[types.Part(text=entry["text"])]
        ))
    return history


def run_chat(initial_query=None):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    current_model = GeminiModel.LITE_25

    def start_session(model_id, history=None):
        return client.chats.create(
            model=model_id,
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
            )
        )

    chat = start_session(current_model.value)
    retrieval_enabled = True

    console.set_window_title("Gemini Docs Chat")
    console.print("[bold color(35)]Documentation Conversation Started. Type 'exit' or 'quit' to stop.[/bold color(35)]")
    console.print("[dim]Commands: /model, /save, /resume, /toggle, exit[/dim]")
    console.print(f"[dim]Initial Model: {current_model.value} | Context: Local VectorDB[/dim]\n")

    # Handle initial query if provided
    first_run = True

    while True:
        try:
            # Display current status
            status_text = "ON" if retrieval_enabled else "OFF"

            if first_run and initial_query:
                query = initial_query
                first_run = False
            else:
                console.rule(style="color(35)")
                console.print(f"[dim]model: {current_model.value} | retrieval {status_text}[/dim]")
                query = console.input("[query]Query > [/query]")
                first_run = False

            if not query.strip():
                continue

            lowered_query = query.lower()

            if lowered_query in ["exit", "quit"]:
                break

            if lowered_query == "/toggle":
                retrieval_enabled = not retrieval_enabled
                console.print(f"[warning]Retrieval is now {'ENABLED' if retrieval_enabled else 'DISABLED'}[/warning]")
                continue

            if lowered_query == "/save":
                save_history(chat.get_history())
                continue

            if lowered_query == "/resume":
                files = sorted(list(CHATS_DIR.glob("chat_*.json")), reverse=True)
                if not files:
                    console.print("[warning]No saved chats found in chats/ directory.[/warning]")
                    continue

                table = Table(title="Saved Conversations", title_style="header")
                table.add_column("Key", style="query")
                table.add_column("Filename", style="markdown.header")
                table.add_column("Date", style="markdown.emph")

                for i, f in enumerate(files[:10]):  # Show last 10
                    date_str = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    table.add_row(str(i+1), f.name, date_str)
                console.print(table)

                choice = console.input("[warning]Select file number to resume > [/warning]")
                try:
                    idx = int(choice) - 1
                    history = load_history(files[idx])
                    chat = start_session(current_model.value, history=history)
                    console.print(f"[bold color(35)]Resumed conversation from {files[idx].name}.[/bold color(35)]")
                    # Show last message as context
                    if history:
                        last_msg = history[-1].parts[0].text
                        console.print(f"[dim]Last message: {last_msg[:100]}...[/dim]")
                except (ValueError, IndexError):
                    console.print("[error]Invalid selection.[/error]")
                continue

            if lowered_query == "/model":
                table = Table(title="Available Models", title_style="header")
                table.add_column("Key", style="query")
                table.add_column("Model ID", style="markdown.header")
                for i, m in enumerate(GeminiModel):
                    table.add_row(str(i+1), m.value)
                console.print(table)

                choice = console.input("[warning]Select model number > [/warning]")
                try:
                    idx = int(choice) - 1
                    selected_model = list(GeminiModel)[idx]

                    # Capture history to resume with new model
                    old_history = chat.get_history()
                    current_model = selected_model
                    chat = start_session(current_model.value, history=old_history)
                    console.print(f"[bold color(35)]Switched to {current_model.value}. Conversation resumed.[/bold color(35)]")
                except (ValueError, IndexError):
                    console.print("[error]Invalid selection. Staying with current model.[/error]")
                continue

            context_text = ""
            if retrieval_enabled:
                with console.status("[status.cyan]Retrieving context...[/status.cyan]"):
                    context_chunks = get_context(query)

                if context_chunks:
                    context_text = "\n\nLOCAL DOCUMENTATION CONTEXT:\n"
                    for context_item in context_chunks:
                        source = context_item['source']
                        text = context_item['text']
                        context_text += f"\nSOURCE: {source}\nCONTENT:\n{text}\n"

            # Send to chat
            full_prompt = f"{query}\n{context_text}" if retrieval_enabled else query

            with console.status("[status.magenta]Thinking...[/status.magenta]"):
                response = chat.send_message(full_prompt)

            console.print(Markdown(response.text, code_theme="stata-dark"))
            console.print("-" * 40)

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@click.command()
@click.argument("query", required=False)
def main(query):
    """
    Interactive conversation with your documentation.
    Optional: pass an initial query to start the chat immediately.
    """
    run_chat(initial_query=query)


if __name__ == "__main__":
    main()
