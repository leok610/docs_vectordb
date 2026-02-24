# Docs VectorDB

A high-performance, scalable pipeline for vectorizing and searching programming documentation using Gemini and local PyTorch models.

## Overview

Docs VectorDB automates the process of transforming plain-text documentation (RST, Markdown, TXT) into a searchable vector database. It features structural semantic chunking, multi-worker scaling for local models, and stateful conversational search interfaces.

## Key Features

- **Scalable Local Inference:** Orchestrates multiple local PyTorch embedding servers in parallel to maximize GPU/CPU utilization.
- **Structural Chunking:** Specialized parsers for RST, Markdown, and TXT that respect document hierarchy and semantic boundaries.
- **Dual Backends:** Support for high-dimensional Gemini (3072-dim) and local `all-mpnet-base-v2` (768-dim) vectors.
- **Stateful Search:** Includes both single-shot AI search and a conversational chat interface with history tracking.
- **Resilient Pipeline:** Built-in resume logic, schema-mismatch tolerance, and VRAM protection via chunk-level batching.

## Quick Start

The project is managed by `uv`. Ensure you have Python >= 3.14 installed.

### 1. Generating the Vector Database
The orchestrator manages the entire lifecycle: assembly, chunking, and embedding.

```powershell
# Default: High-speed local rebuild with 4 parallel worker servers
uv run python src/docs_vectordb/generate_vectordb.py --embedder pytorch --workers 4 --rebuild

# Gemini: Cloud-based indexing with resume support (skips existing files)
uv run python src/docs_vectordb/generate_vectordb.py --embedder gemini
```

### 2. AI-Powered Search
Use the single-shot search tool for quick technical answers:
```powershell
# Formal AI search across all documentation
uv run doc-search "How do I use asyncio queues?"
```

### 3. Conversational Interface
Engage in a stateful dialogue with your documentation:
```powershell
# Start the interactive chat tool
uv run doc-chat
```
**Chat Commands:**
- `/model [1-4]`: Switch between Gemini 1.5 Flash, 1.5 Flash-Lite, 3.0 Flash, etc.
- `/toggle`: Enable or disable VectorDB retrieval for the next message.
- `/save`: Export the current conversation to `chats/`.
- `/resume`: Reconstruct conversation history from a saved JSON.

## Project Structure

- `src/docs_vectordb/`: Core source code.
  - `generate_vectordb.py`: Orchestrator (Phase 1-4).
  - `doc_search.py`: Single-shot AI search tool.
  - `doc_search_conversation.py`: Stateful chat interface.
  - `embedding_server.py`: Scalable PyTorch inference server.
  - `embed_pytorch.py`: Parallel client for local models.
  - `embed_gemini.py`: Asynchronous client for Google GenAI.
- `tests/`: Comprehensive suite (39+ tests) including unit, integration, and CLI consistency checks.
- `database/`: Local LanceDB vector storage.
- `chunks/`: Intermediate semantic fragments (gitignored).

## Maintenance & Verification

### Health Check
Verify database integrity and count vectors per source:
```powershell
database-healthcheck
```

### Testing & Linting
Run the unified test runner for static analysis (mypy) and unit tests:
```powershell
python run_tests.py
```

## Technical Specifications
- **Gemini Embeddings:** 3072 dimensions, 1M TPM rate limit.
- **PyTorch Embeddings:** 768 dimensions (`all-mpnet-base-v2`).
- **Batching:** 256 chunks per inference request (VRAM Protection), 1000 chunks per DB transaction.
- **Database:** LanceDB (Serverless).
