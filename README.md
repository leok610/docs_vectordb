# Docs VectorDB

This repository contains a CLI-native vector database system designed to ingest
local technical reference documentation and make it highly searchable from the
terminal. The system parses Markdown, RST, and plain text files, generates
vector embeddings, and stores them in LanceDB for rapid semantic retrieval.

## Backends

The system explicitly separates the task of embedding text into vectors from
the task of generating the final conversational answers. Crucially, the
embedding process happens twice: first when parsing the files to build the
database, and again every time a search query is submitted. The search query
must be vectorized using the exact same backend as the database to accurately
calculate similarity.

**Embedding Backends**
The pipeline supports two distinct models for generating vector embeddings:

* **Local PyTorch:** Uses the `all-mpnet-base-v2` SentenceTransformer model.
This executes entirely locally and is optimized for speed and zero-cost
scaling, making it the default for rapid CLI queries.
* **Gemini API:** Uses Google's remote embedding API to generate
higher-dimensional vectors. This relies on cloud processing and is subject to
network latency and API rate limits.

**Generative Backend**
Once the query is embedded and the relevant document chunks are retrieved from
the local database, the context is passed exclusively to the **Gemini 2.5 Flash
Lite** model. This model is utilized as the sole generative engine because of
its extreme speed and strict adherence to system prompts, quickly formatting
the raw documentation excerpts into concise Markdown answers directly in the
terminal.

## Pipeline Architecture

The data flow is managed by a central orchestrator script
(`generate_vectordb.py`) that pushes raw documentation through a multi-phase
pipeline.

```text
[Raw Docs] 
   |
   v
(Chunking Scripts)  --> Split text by headers, indents, or paragraphs
   |
   v
[JSON Chunks]       --> Temp storage in /chunks
   |
   v
(Embedding Scripts) --> Pass text to PyTorch or Gemini API to get vectors
   |
   v
[LanceDB]           --> Persistent vector database
   |
   v
(doc_search)        --> Terminal queries the DB, sends context to AI
   |
   v
[Terminal UI]       --> Formatted Markdown answers and CLI ripgrep fallbacks

```

The assembly phase scans the target directories for supported file types and
passes them to the chunking scripts. These chunkers parse the text based on its
format (such as separating RST by Sphinx headers or Markdown by hash marks) to
create overlapping semantic units. These units are temporarily saved as JSON
files in the `chunks` directory. The embedding phase then reads these JSON
files and sends the text to either a local PyTorch server or the remote Gemini
API to generate vector arrays. Finally, these arrays are committed to LanceDB.

Once the database is built, the terminal UI scripts query the database using
the same embedding logic and pass the retrieved context to an LLM to generate
formatted, actionable answers directly in the console.

## Directory Structure

The architecture isolates the persistent database, the execution logic, and the
temporary build files into dedicated directories.

* **`src/docs_vectordb/`**: This directory houses the core execution logic. It
is functionally divided into chunking processors, embedding generators, and the
terminal interaction tools.
* **`tests/`**: The pytest suite ensures that the chunking boundary logic
remains accurate and that the CLI output formatting stays consistent as the
tools are updated.
* **`database/`**: The persistent storage location where LanceDB maintains the
binary vector tables and search indices.
* **`logs/`**: Text logs generated during the heavy background processing steps
are written here to keep the main terminal output clean during database
generation.
* **`chats/`**: Serialized JSON dumps of previous interactive sessions from the
conversation script are saved here, allowing specific problem-solving contexts
to be reloaded in future terminal sessions.
* **`chunks/`**: A temporary directory generated during the database build
process to hold the intermediate JSON representations of the parsed documents
before they are embedded.

## Concurrency in the Pipeline

Because text parsing and vector math are heavily CPU-bound, relying on
standard, single-threaded Python would be a massive bottleneck due to the
Global Interpreter Lock (GIL). The pipeline bypasses this by implementing two
different concurrency strategies for chunking and embedding.

### 1. Async Document Chunking

When the orchestrator calls scripts like `chunk_by_rst.py`, it passes an
`--async-mode` flag. Instead of parsing hundreds of files sequentially, the
chunking utilities use a hybrid approach.

```text
[Async Event Loop] (Main Thread)
   |
   |-- loop.run_in_executor() creates a "Future" (an IOU) for each file
   |
   +---> [Process Pool Executor] (Bypasses the GIL)
             |
             |---> Worker 1 (CPU Core) -> process_single_file(doc_A)
             |---> Worker 2 (CPU Core) -> process_single_file(doc_B)
             |---> Worker 3 (CPU Core) -> process_single_file(doc_C)
             |
   | <-------+ (Workers finish and resolve their Futures)
   |
   |-- await asyncio.gather(*tasks) pauses the main thread until ALL are done
   v
[Return Total Chunks]

```

The synchronous work (reading a file, finding headers, splitting lines) is
handed off to a background process pool. The main thread's async event loop
simply manages the IOUs ("Futures") for these tasks, ensuring all CPU cores are
utilized simultaneously to crunch the text without blocking the orchestrator.

### 2. GPU-Optimized Local Embedding

Generating local PyTorch embeddings (`embed_pytorch.py`) is the most
computationally expensive step. Instead of spawning multiple workers (which duplicates the CUDA context tax and model weights in VRAM), the orchestrator uses a single-worker massive-batching strategy.

```text
[Orchestrator: generate_vectordb.py]
   |
   |-- Spawns a single massive-batch Flask server
   v
[Worker 5000] (Waitress/Flask + MPNet Model)
   ^
   |-- Sends massive sequential HTTP POST batches
   |
[embed_pytorch.py] (Reads all JSON chunks into flat list)

```

1. **Worker Spawning:** `generate_vectordb.py` uses `subprocess` to spin up
   a single instance of `embedding_server.py` on a local port. This reserves maximum VRAM (e.g., ~11 GB out of 12 GB) exclusively for processing data rather than overhead.
2. **Massive Batch Routing:** `embed_pytorch.py` flattens chunks across all documents into a massive global list. It batches these chunks (e.g., 2048 at a time) and sends them to the server sequentially, ensuring the GPU's CUDA cores are fully fed without crashing.
3. **Stateless Processing:** The single Flask server handles the requests statelessly. Because the entire pipeline sends massive batches sequentially, the GPU operates at peak efficiency.

## Installation

The project relies on the `uv` Python manager for fast dependency resolution. The
configuration includes a custom index to pull the correct CUDA 13.0 wheels for
PyTorch directly.

To install the dependencies and register the CLI tools directly into your
system environment, run the following in the project root:

```bash
uv pip install --system .

```
The scripts can also be run with `uv run <script_name>` if you prefer not to
install system-wide. The Python virtual environment will be used automatically.
It is also possible to activate the virtual environment with the activate
scripts and run Python commands without `uv`.

## CLI Tools

Installing the project via `uv` registers several command-line interfaces
directly into the environment path. These commands provide direct access to the
database and AI search features without needing to invoke the underlying Python
scripts manually.

* **`database-healthcheck`**: A diagnostic command that verifies the LanceDB
connection, checks table integrity, and ensures the vector indices are properly
initialized.
* **`doc-retrieval`**: The raw search interface. It takes a search query,
embeds it into a vector using the chosen backend, and returns the raw JSON
chunks retrieved from LanceDB. This command is heavily utilized under the hood
by the AI scripts but is exposed here for rapid debugging and manual vector
queries.
* **`doc-search`**: The primary one-shot AI query tool. It uses `doc-retrieval`
to fetch local context and immediately passes it to the Gemini 2.5 Flash Lite
model. It is optimized for speed and outputs concise, formatted Markdown. If
the database lacks the answer, it generates specific `ripgrep` fallback
commands.
* **`doc-search-conversation`**: The interactive, stateful chat mode. This
command launches a continuous console session with several built-in commands:
* `/toggle`: Turns local vector retrieval on or off, allowing the model to
switch between acting as a strict documentation assistant and a general coding
assistant.
* `/model`: Swaps the active Gemini model mid-conversation.
* `/save`: Serializes the current conversation history to a JSON file in the
`chats/` directory.
* `/resume`: Displays a table of recent chat sessions and reloads the history
to continue a previous troubleshooting session.


* **`pytorch-server`**: A standalone utility to spin up a local PyTorch
Waitress/Flask embedding server independently. When `doc-retrieval` needs to
embed a user's terminal query, loading the PyTorch model into memory from
scratch has significant cold-start latency. By keeping a local
server running in the background via `service-wrapper`, the CLI tools can
instantly POST the query to the active model, speeding up the
response time for `doc-search` and `doc-search-conversation`. Passing the
switch `--external-server` to the `generate-vectordb.py` when using Pytorch
will also utilize this server.
