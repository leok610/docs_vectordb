from google import genai
from google.genai import types
import os
from rich import inspect, print
from rich.text import Text
from rich.traceback import install as trace_install

trace_install(show_locals=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

try:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=("What is the meaning of life?", "Does the soul exist?", "Does God exist?")
    )
except genai.ClientError as e:
    print(e)

# text1 = Text(response.embeddings)
# text1.stylize("blue")

print(response.embeddings)
