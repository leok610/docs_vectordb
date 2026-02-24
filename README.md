# Docs VectorDB

A high-performance pipeline for vectorizing and searching programming documentation using Gemini and local PyTorch models.

## Overview

Docs VectorDB automates the process of transforming plain-text documentation (RST, Markdown, TXT) into a searchable vector database. It features structural chunking, rate-limited embedding generation, and an efficient retrieval interface.

## System Commands

The project is managed by `uv` and provides the following interface commands:

- `doc-retrieval`: Performs vector search across the generated database.
- `database-healthcheck`: Verifies the status and integrity of the documentation vector database.

## Project Structure

- `src/docs_vectordb/`: Core source code.
  - `generate_vectordb.py`: Internal orchestrator logic.
  - `doc_retrieval.py`: Search and retrieval logic.
  - `healthcheck.py`: Database verification logic.
  - `chunk_by_*.py`: Specialized chunking logic for different file formats.
  - `embed_gemini.py`: Asynchronous Gemini API integration.
  - `embed_pytorch.py`: Local PyTorch integration.
- `tests/`: Comprehensive test suite.
- `database/`: Local LanceDB storage.
- `logs/`: Execution and error logs.

## Workflows

### 1. Generating the Database (Internal Only)
The generation process is restricted to internal use to ensure database integrity. It can be triggered via:
```powershell
# Default (Gemini 3072-dim)
uv run python -m src.docs_vectordb.generate_vectordb --embedder gemini

# Local (PyTorch 768-dim)
uv run python -m src.docs_vectordb.generate_vectordb --embedder pytorch
```

### 2. Searching
To search the database using the installed interface:
```powershell
# Search using the default embedder
doc-retrieval "how to use decorators"

# Search specifically using the local model
doc-retrieval "asyncio tasks" --embedder pytorch
```

### 3. Health Check
To verify the database status:
```powershell
database-healthcheck
```

### 4. Running Tests
The project includes a unified test runner for static analysis and unit tests:
```powershell
python run_tests.py
```

## CLI Configuration
All commands support standard help flags:
- `-h`
- `-?`
- `--help`

## Technical Details
- **Gemini Embeddings:** 3072 dimensions, optimized with asynchronous batching and rate limiting (1M TPM).
- **PyTorch Embeddings:** 768 dimensions using `all-mpnet-base-v2`.
- **Database:** LanceDB (Serverless vector database).
- **Python Version:** >= 3.14
