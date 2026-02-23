"""
LanceDB Inspection Module
=========================

This module provides utility functions for exploring and diagnosing LanceDB databases.
It is designed to be used as a library for scripts that need to inspect table schemas,
row counts, and perform various types of searches.

LanceDB Documentation:
----------------------
- Project Home: https://lancedb.github.io/lancedb/
- Python SDK: https://lancedb.github.io/lancedb/python/python/
- Searching: https://lancedb.github.io/lancedb/searching/
- Filtering: https://lancedb.github.io/lancedb/sql/

Concepts:
---------
- URI: The location of the database (local directory or s3:// path).
- Table: A collection of data with a schema, stored in Lance format.
- Vector Search: Finding the nearest neighbors of a query vector.
- Scalar Filter: A SQL-like 'WHERE' clause applied to non-vector columns.
"""

import lancedb
import polars as pl
import pyarrow as pa
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, cast

def connect_db(uri: str) -> lancedb.DBConnection:
    """
    Establishes a connection to the LanceDB instance.
    
    Returns:
        lancedb.DBConnection: A handle to the database directory.
    
    Ref: https://lancedb.github.io/lancedb/python/python/#connecting-to-a-database
    """
    try:
        return lancedb.connect(uri)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to LanceDB at {uri}: {e}")

def list_tables(db: lancedb.DBConnection) -> List[str]:
    """
    Returns a list of all table names in the database.
    
    Returns:
        List[str]: A list of table names found in the connection.
    
    Ref: https://lancedb.github.io/lancedb/python/python/#listing-tables
    """
    res = db.list_tables()
    
    # Handle the various response types from different LanceDB versions
    if hasattr(res, 'tables'): 
        return cast(List[str], res.tables)
    elif isinstance(res, dict) and 'tables' in res:
        return cast(List[str], res['tables'])
    elif isinstance(res, list):
        return cast(List[str], res)
    
    try:
        return list(res)
    except:
        return []

def get_table_details(db: lancedb.DBConnection, table_name: str) -> Dict[str, Any]:
    """
    Retrieves schema, row count, and metadata for a specific table.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'name' (str): The table name.
            - 'schema' (pyarrow.Schema): The full Arrow schema of the table.
            - 'count' (int): Total number of rows in the table.
            
    Ref: https://lancedb.github.io/lancedb/python/python/#opening-an-existing-table
    """
    if table_name not in list_tables(db):
        raise ValueError(f"Table '{table_name}' does not exist in the database.")
        
    table = db.open_table(table_name)
    return {
        "name": table_name,
        "schema": table.schema, # This is a pyarrow.Schema
        "count": table.count_rows(),
    }

def peek_rows(db: lancedb.DBConnection, table_name: str, limit: int = 5) -> pl.DataFrame:
    """
    Returns the first N rows of a table as a Polars DataFrame.
    
    Returns:
        polars.DataFrame: The retrieved rows.
        
    Ref: https://lancedb.github.io/lancedb/python/python/#reading-data
    """
    table = db.open_table(table_name)
    # head() returns a pyarrow.Table; we convert to Polars
    return cast(pl.DataFrame, pl.from_arrow(table.head(limit)))

def vector_search(db: lancedb.DBConnection, table_name: str, vector: List[float], limit: int = 5) -> pl.DataFrame:
    """
    Performs a basic vector similarity search (Approximate Nearest Neighbor).
    
    Returns:
        polars.DataFrame: Search results including distance and original columns.
        
    Ref: https://lancedb.github.io/lancedb/searching/#vector-search
    """
    table = db.open_table(table_name)
    try:
        # search() returns a LanceQueryBuilder
        # to_arrow() returns a pyarrow.Table
        return cast(pl.DataFrame, pl.from_arrow(table.search(vector).limit(limit).to_arrow()))
    except Exception as e:
        raise RuntimeError(f"Vector search failed on table '{table_name}': {e}")

def hybrid_search(db: lancedb.DBConnection, table_name: str, vector: List[float], filter_expr: str, limit: int = 5) -> pl.DataFrame:
    """
    Performs a vector search combined with a SQL-style scalar filter.
    
    Returns:
        polars.DataFrame: Search results matching the filter and similarity criteria.
        
    Ref: https://lancedb.github.io/lancedb/searching/#hybrid-search (Filter + Vector)
    """
    table = db.open_table(table_name)
    try:
        return cast(pl.DataFrame, pl.from_arrow(table.search(vector).where(filter_expr).limit(limit).to_arrow()))
    except Exception as e:
        raise RuntimeError(f"Hybrid search (vector + filter) failed on table '{table_name}': {e}")

def full_text_search(db: lancedb.DBConnection, table_name: str, query_text: str, limit: int = 5) -> pl.DataFrame:
    """
    Performs a full-text search (FTS) using a BM25 index.
    
    Returns:
        polars.DataFrame: Results of the keyword search.
        
    Requires an FTS index: table.create_fts_index("text_column")
    Ref: https://lancedb.github.io/lancedb/searching/#full-text-search
    """
    table = db.open_table(table_name)
    try:
        return cast(pl.DataFrame, pl.from_arrow(table.search(query_text).limit(limit).to_arrow()))
    except Exception as e:
        raise RuntimeError(f"Full-text search failed on table '{table_name}'. Did you create an FTS index? Error: {e}")

def get_schema_summary(db: lancedb.DBConnection, table_name: str) -> List[Dict[str, str]]:
    """
    Returns a simplified list of field names and their types from the Arrow schema.
    
    Returns:
        List[Dict[str, str]]: A list of dictionaries with 'name' and 'type' keys.
    """
    table = db.open_table(table_name)
    schema = table.schema # pyarrow.Schema
    return [{"name": f.name, "type": str(f.type)} for f in schema]

def check_table_existence(db: lancedb.DBConnection, table_name: str) -> bool:
    """
    Verifies if a table exists in the database.
    
    Returns:
        bool: True if the table exists, False otherwise.
    """
    return table_name in list_tables(db)
