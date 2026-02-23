import lancedb
import json
import rich_click as click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.traceback import install as trace_install

trace_install()
console = Console()

URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"
GEMINI_TABLE = "gemini_reference_docs"
PYTORCH_TABLE = "reference_docs"

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

@click.command(context_settings=CONTEXT_SETTINGS, help="""
    Verifies the status and integrity of the documentation vector database.
    
    Checks for the existence of required tables (Gemini and PyTorch) and 
    reports row counts and schema information.
    """)
def main():
    db_path = Path(URI)
    if not db_path.exists():
        console.print(f"[red]Error: Database directory not found at {URI}[/red]")
        return

    try:
        db = lancedb.connect(uri=URI)
        response = db.list_tables()
        available_tables = response.tables if hasattr(response, 'tables') else response
    except Exception as e:
        console.print(f"[red]Error connecting to database:[/red] {e}")
        return

    table = Table(title="Database Health Status")
    table.add_column("Table Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Rows", style="yellow")
    table.add_column("Dimensions", style="magenta")

    for table_name in [GEMINI_TABLE, PYTORCH_TABLE]:
        if table_name in available_tables:
            try:
                t = db.open_table(table_name)
                count = t.count_rows()
                # Get dimensions from schema
                schema = t.schema
                dim = "Unknown"
                for field in schema:
                    if field.name == "vector":
                        # arrow fixed_size_list type has a list_size attribute
                        dim = str(field.type.list_size)
                
                table.add_row(table_name, "Healthy", str(count), dim)
            except Exception as e:
                table.add_row(table_name, "Error", "N/A", f"[red]{str(e)}[/red]")
        else:
            table.add_row(table_name, "Missing", "0", "N/A")

    console.print(table)

if __name__ == "__main__":
    main()
