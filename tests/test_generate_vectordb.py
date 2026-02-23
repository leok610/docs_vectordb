import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json
from src.docs_vectordb.generate_vectordb import run_script

class TestGenerateVectorDB(unittest.TestCase):
    @patch("subprocess.run")
    def test_run_script_success(self, mock_run):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "output"
        
        result = run_script("test.py", "arg1")
        self.assertEqual(result, "output")
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @patch("src.docs_vectordb.generate_vectordb.console.print")
    def test_run_script_failure(self, mock_print, mock_run):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "error"
        
        result = run_script("test.py", "arg1")
        self.assertIsNone(result)
        mock_print.assert_called_once()

    @patch("src.docs_vectordb.generate_vectordb.run_script")
    @patch("src.docs_vectordb.generate_vectordb.lancedb.connect")
    @patch("src.docs_vectordb.generate_vectordb.shutil.rmtree")
    @patch("src.docs_vectordb.generate_vectordb.Path.mkdir")
    @patch("src.docs_vectordb.generate_vectordb.Path.open", new_callable=mock_open)
    def test_main_flow_gemini(self, mock_path_open, mock_mkdir, mock_rmtree, mock_lancedb, mock_run_script):
        from src.docs_vectordb.generate_vectordb import main
        from click.testing import CliRunner
        
        # Mock responses
        mock_run_script.side_effect = [
            json.dumps(["file1.rst"]), # assemble_doclist
            "", # chunk_by_rst.py
            json.dumps({"vectors_stored": 10}) # embed_gemini.py
        ]
        
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        mock_db.list_tables.return_value = ["gemini_reference_docs"]
        
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        
        runner = CliRunner()
        result = runner.invoke(main, ["--embedder", "gemini"])
        
        self.assertEqual(result.exit_code, 0)
        mock_rmtree.assert_called()
        mock_lancedb.assert_called()
        # For Gemini, table is NOT dropped
        mock_db.drop_table.assert_not_called()
        
        # Verify indexing
        mock_db.open_table.assert_called_with("gemini_reference_docs")
        mock_table.create_index.assert_called_with(
            metric="cosine", 
            num_partitions=64, 
            num_sub_vectors=96
        )

    @patch("src.docs_vectordb.generate_vectordb.run_script")
    @patch("src.docs_vectordb.generate_vectordb.lancedb.connect")
    @patch("src.docs_vectordb.generate_vectordb.shutil.rmtree")
    @patch("src.docs_vectordb.generate_vectordb.Path.mkdir")
    @patch("src.docs_vectordb.generate_vectordb.Path.open", new_callable=mock_open)
    def test_main_flow_pytorch(self, mock_path_open, mock_mkdir, mock_rmtree, mock_lancedb, mock_run_script):
        from src.docs_vectordb.generate_vectordb import main
        from click.testing import CliRunner
        
        # Mock responses
        mock_run_script.side_effect = [
            json.dumps(["file1.rst"]), # assemble_doclist
            "", # chunk_by_rst.py
            json.dumps({"vectors_stored": 10}) # embed_and_store.py
        ]
        
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        mock_db.list_tables.return_value = ["reference_docs"]
        
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        
        runner = CliRunner()
        result = runner.invoke(main, ["--embedder", "pytorch"])
        
        self.assertEqual(result.exit_code, 0)
        mock_rmtree.assert_called()
        mock_lancedb.assert_called()
        # For PyTorch, table IS dropped
        mock_db.drop_table.assert_called_with("reference_docs")
        
        # Verify indexing
        mock_db.open_table.assert_called_with("reference_docs")
        mock_table.create_index.assert_called_with(
            metric="cosine", 
            num_partitions=64, 
            num_sub_vectors=32
        )

if __name__ == "__main__":
    unittest.main()
