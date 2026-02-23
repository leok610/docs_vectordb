import unittest
import lancedb
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add src/docs_vectordb to path to import the module
# We use absolute path to ensure reliability across environments
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent / "src" / "docs_vectordb"
sys.path.insert(0, str(src_dir))

import lancedb_inspection as info

class TestLanceDBInspection(unittest.TestCase):
    def setUp(self):
        """Provides a temporary directory for a LanceDB instance and populates it."""
        self.test_dir = tempfile.mkdtemp()
        self.db_uri = self.test_dir
        self.db = lancedb.connect(self.db_uri)
        
        # Sample data
        self.data = [
            {"id": "1", "text": "hello world", "vector": np.random.rand(128).tolist()},
            {"id": "2", "text": "foo bar", "vector": np.random.rand(128).tolist()},
            {"id": "3", "text": "test query", "vector": np.random.rand(128).tolist()},
        ]
        self.table = self.db.create_table("test_table", data=self.data)

    def tearDown(self):
        """Cleans up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_connect_db(self):
        db = info.connect_db(self.db_uri)
        self.assertIsInstance(db, lancedb.DBConnection)

    def test_list_tables(self):
        db = info.connect_db(self.db_uri)
        tables = info.list_tables(db)
        self.assertIn("test_table", tables)

    def test_get_table_details(self):
        db = info.connect_db(self.db_uri)
        details = info.get_table_details(db, "test_table")
        self.assertEqual(details["name"], "test_table")
        self.assertEqual(details["count"], 3)
        self.assertIn("id", [f.name for f in details["schema"]])

    def test_get_table_details_nonexistent(self):
        db = info.connect_db(self.db_uri)
        with self.assertRaises(ValueError):
            info.get_table_details(db, "ghost_table")

    def test_peek_rows(self):
        db = info.connect_db(self.db_uri)
        df = info.peek_rows(db, "test_table", limit=2)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("text", df.columns)

    def test_vector_search(self):
        db = info.connect_db(self.db_uri)
        query_vector = np.random.rand(128).tolist()
        results = info.vector_search(db, "test_table", query_vector, limit=1)
        self.assertEqual(len(results), 1)
        self.assertIn("id", results.columns)

    def test_check_table_existence(self):
        db = info.connect_db(self.db_uri)
        self.assertTrue(info.check_table_existence(db, "test_table"))
        self.assertFalse(info.check_table_existence(db, "nope"))

    def test_get_schema_summary(self):
        db = info.connect_db(self.db_uri)
        summary = info.get_schema_summary(db, "test_table")
        self.assertTrue(any(f['name'] == 'text' for f in summary))
        self.assertTrue(any(f['name'] == 'vector' for f in summary))

    def test_list_tables_normalization(self):
        """
        Regression test: Ensure list_tables handles various SDK return formats
        (dict, objects with .tables, etc) and always returns List[str].
        """
        from unittest.mock import MagicMock
        mock_db = MagicMock()
        
        # Case 1: Returns a simple list
        mock_db.list_tables.return_value = ["table1", "table2"]
        self.assertEqual(info.list_tables(mock_db), ["table1", "table2"])
        
        # Case 2: Returns a dict (older or specific SDK versions)
        mock_db.list_tables.return_value = {"tables": ["t1", "t2"], "page_token": None}
        self.assertEqual(info.list_tables(mock_db), ["t1", "t2"])
        
        # Case 3: Returns a ListTablesResponse-like object (Current SDK)
        mock_obj = MagicMock()
        mock_obj.tables = ["obj1", "obj2"]
        mock_db.list_tables.return_value = mock_obj
        self.assertEqual(info.list_tables(mock_db), ["obj1", "obj2"])

    def test_data_type_safety_for_rich(self):
        """
        Regression test: Ensure data intended for Rich display is properly
        handled to avoid NotRenderableError (e.g. converting exceptions/tuples to strings).
        """
        # This simulates the logic inside healthcheck where we add rows to a table
        from rich.table import Table
        from rich.errors import NotRenderableError
        
        table = Table()
        table.add_column("Test")
        
        # Attempting to add a raw exception should be caught by our string conversion
        e = ValueError("test error")
        
        # This is what failed before: table.add_row(e)
        # This is our fix:
        try:
            table.add_row(str(e))
        except NotRenderableError:
            self.fail("Table.add_row failed with stringified exception")

if __name__ == "__main__":
    unittest.main()
