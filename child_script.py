
import sys, time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

console = Console(stderr=True, force_terminal=True, force_interactive=True)
with Live(Panel('Test Dashboard'), console=console, refresh_per_second=4, screen=True):
    for i in range(5):
        time.sleep(0.5)
