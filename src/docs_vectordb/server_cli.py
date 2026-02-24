import os
import sys
import time
import signal
import threading
import subprocess
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout

from docs_vectordb.embedding_server import app
from waitress import serve

PID_FILE = Path(".pytorch_server.pid")

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help', '-?'])

def get_memory_usage(pid):
    try:
        output = subprocess.check_output(f'tasklist /FI "PID eq {pid}" /FO CSV /NH', shell=True, text=True)
        parts = output.strip().split('","')
        if len(parts) >= 5:
            mem_str = parts[4].replace('"', '')
            return f"{mem_str.strip()}"
    except Exception:
        pass
    return "Unknown"

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--stop", is_flag=True, help="Stop the running Pytorch Server.")
def main(stop):
    console = Console()
    if stop:
        if PID_FILE.exists():
            pid_str = PID_FILE.read_text().strip()
            if not pid_str.isdigit():
                console.print(f"[red]Invalid PID file content: {pid_str}[/red]")
                return
            pid = int(pid_str)
            try:
                # taskkill is more reliable on Windows to kill process trees if needed, 
                # but os.kill(pid, signal.SIGTERM) usually works for Python processes.
                os.kill(pid, signal.SIGTERM)
                console.print(f"[green]Stopped server with PID {pid}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to stop server (PID {pid}): {e}[/red]")
                # If it fails, we might still want to clean up the PID file if the process is actually gone
            finally:
                PID_FILE.unlink(missing_ok=True)
        else:
            console.print("[yellow]No PID file found. Is the server running?[/yellow]")
        return

    if PID_FILE.exists():
        console.print("[yellow]PID file already exists. Server might be running. If not, delete .pytorch_server.pid[/yellow]")
        return
        
    pid = os.getpid()
    PID_FILE.write_text(str(pid))
    
    server_thread = threading.Thread(target=serve, args=(app,), kwargs={"host": "127.0.0.1", "port": 5000, "clear_untrusted_proxy_headers": False}, daemon=True)
    server_thread.start()

    console.set_window_title("Pytorch Server 🚀")
    start_time = time.time()
    
    # Suppress background logging that interferes with the rich Live dashboard
    import logging
    logging.getLogger("waitress").setLevel(logging.ERROR)
    logging.getLogger("embedding-server").setLevel(logging.ERROR)
    
    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                uptime_seconds = int(time.time() - start_time)
                    m, s = divmod(uptime_seconds, 60)
                    h, m = divmod(m, 60)
                    uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
                    
                    mem_usage = get_memory_usage(pid)
                    
                    text = Text()
                    text.append("Pytorch Embedding Server 🚀\n\n", style="bold cyan")
                    text.append(f"Port:         5000\n", style="green")
                    text.append(f"Uptime:       {uptime_str}\n", style="yellow")
                    text.append(f"Memory Usage: {mem_usage}\n", style="magenta")
                    text.append(f"PID:          {pid}\n\n", style="dim white")
                    text.append("Press Ctrl+C to close the server.", style="bold red")
                    
                    panel = Panel(text, title="Server Status", border_style="cyan", width=50)
                    live.update(panel)
                    time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        PID_FILE.unlink(missing_ok=True)
        console.print("[green]Server stopped.[/green]")

if __name__ == "__main__":
    main()
