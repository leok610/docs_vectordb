from pathlib import Path
from rich import print
from rich.traceback import install as trace_install
from sentence_transformers import SentenceTransformer

trace_install()

model = SentenceTransformer("all-mpnet-base-v2")
