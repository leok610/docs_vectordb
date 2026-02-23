"""
Database Healthcheck Script
===========================

A CLI tool for validating the state of the documentation vector database.
It uses the `lancedb_inspection` module to verify connectivity, table integrity,
and provide a summary of the stored data.

Usage:
------
uv run py src/docs_vectordb/database_healthcheck.py [--db-uri PATH]

Dependencies:
-------------
- rich: For beautiful console output.
- rich-click: For a polished CLI interface.
- lancedb: To interact with the vector database.
- polars: For data manipulation and preview.
"""

import os
import sys
import lancedb
from pathlib import Path
import rich_click as click
import rich.box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import polars as pl

# Import our inspection module
try:
    import lancedb_inspection as info
except ImportError:
    # Handle cases where the module isn't in the path
    sys.path.append(str(Path(__file__).parent))
    import lancedb_inspection as info

# Configuration
DEFAULT_URI = "C:/git-repositories/leok610/docs_vectordb/database/docs_lancedb"
console = Console()

@click.command()
@click.option("--db-uri", default=DEFAULT_URI, help="Path to the LanceDB database directory.")
@click.option("--verbose", is_flag=True, help="Show more detailed schema information.")
def healthcheck(db_uri, verbose):
    """
    Performs a baseline healthcheck and inspection of the documentation vector database.
    
    This script verifies that the database is accessible, lists all tables,
    reports row counts, and previews data to ensure everything is indexed correctly.
    """
    db_path = Path(db_uri)
    if not db_path.exists():
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] Database path does not exist: [yellow]{db_uri}[/yellow]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]LanceDB Diagnostic Tool[/bold cyan]\n"
        f"URI: [blue]{db_uri}[/blue]\n"
        f"Docs: https://lancedb.github.io/lancedb/",
        border_style="bright_blue"
    ))

    try:
        db = info.connect_db(db_uri)
    except Exception as e:
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] Database connection failed.\nDetails: {e}")
        sys.exit(1)

    try:
        tables = info.list_tables(db)
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] Could not retrieve table list: {e}")
        sys.exit(1)

    if not tables:
        console.print(Panel("[yellow]The database is initialized but contains no tables.[/yellow]", title="Status"))
        return

    # 1. Summary Table
    summary_table = Table(title="Inventory Summary", show_header=True, header_style="bold magenta", box=rich.box.ROUNDED)
    summary_table.add_column("Table Name", style="cyan")
    summary_table.add_column("Record Count", justify="right", style="green")
    summary_table.add_column("Dimensions", justify="center")

    for table_name in tables:
        # Ensure table_name is a string for rendering and API calls
        tname_str = str(table_name)
        try:
            details = info.get_table_details(db, tname_str)
            # Try to find vector dimension from schema
            dim = "N/A"
            for field in details['schema']:
                # Lance/Arrow fixed size list usually represents the vector
                if str(field.type).startswith("fixed_size_list"):
                    dim = str(field.type).split("[")[-1].split("]")[0]
            
            summary_table.add_row(
                tname_str,
                f"{details['count']:,}",
                dim
            )
        except Exception as e:
            summary_table.add_row(tname_str, "[red]Error[/red]", f"[red]{str(e)}[/red]")

    console.print(summary_table)

    # 2. Detailed Inspections
    for table_name in tables:
        tname_str = str(table_name)
        console.print(f"\n[bold inverse yellow] Table: {tname_str} [/bold inverse yellow]")
        
        try:
            # Schema
            fields = info.get_schema_summary(db, tname_str)
            if verbose:
                schema_table = Table(show_header=True, header_style="italic", title="Schema Details")
                schema_table.add_column("Field Name")
                schema_table.add_column("Data Type (Arrow)")
                for f in fields:
                    schema_table.add_row(f['name'], f['type'])
                console.print(schema_table)

            # Data Integrity Peek
            count = info.get_table_details(db, tname_str)['count']
            if count > 0:
                df = info.peek_rows(db, tname_str, limit=2)
                console.print(f"[dim]Sample records (Total: {count:,}):[/dim]")
                
                # Clean up for display
                display_cols = [c for c in df.columns if c != 'vector']
                console.print(df.select(display_cols))
                
                # Check for nulls in critical columns
                if 'text' in df.columns:
                    null_count = df.filter(pl.col("text").is_null()).height
                    if null_count > 0:
                        console.print(f"[bold red]WARNING:[/bold red] Found {null_count} nulls in 'text' column!")
                    else:
                        console.print("[green]✓ Text integrity verified (no nulls in sample).[/green]")
            else:
                console.print("[yellow]Table exists but is currently empty.[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]ERROR inspecting {tname_str}:[/bold red] {str(e)}")

    console.print("\n[bold green]Diagnostics complete. Database appears healthy.[/bold green]")

if __name__ == "__main__":
    healthcheck()
