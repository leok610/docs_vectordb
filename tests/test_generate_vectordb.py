import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json
from docs_vectordb.generate_vectordb import run_script

class TestGenerateVectorDB(unittest.TestCase):
    @patch("subprocess.run")
    def test_run_script_success(self, mock_run):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "output"
        
        result = run_script("test.py", "arg1")
        self.assertEqual(result, "output")
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_script_failure(self, mock_run):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "error"
        
        result = run_script("test.py", "arg1")
        self.assertIsNone(result)

    @patch("docs_vectordb.generate_vectordb.run_script")
    @patch("docs_vectordb.generate_vectordb.lancedb.connect")
    @patch("docs_vectordb.generate_vectordb.shutil.rmtree")
    @patch("docs_vectordb.generate_vectordb.Path.mkdir")
    @patch("docs_vectordb.generate_vectordb.Path.open", new_callable=mock_open)
    def test_main_flow_gemini(self, mock_path_open, mock_mkdir, mock_rmtree, mock_lancedb, mock_run_script):
        from docs_vectordb.generate_vectordb import main
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

    @patch("docs_vectordb.generate_vectordb.run_script")
    @patch("docs_vectordb.generate_vectordb.lancedb.connect")
    @patch("docs_vectordb.generate_vectordb.shutil.rmtree")
    @patch("docs_vectordb.generate_vectordb.Path.mkdir")
    @patch("docs_vectordb.generate_vectordb.Path.open", new_callable=mock_open)
    def test_main_flow_pytorch(self, mock_path_open, mock_mkdir, mock_rmtree, mock_lancedb, mock_run_script):
        from docs_vectordb.generate_vectordb import main
        from click.testing import CliRunner
        
        # Mock responses
        mock_run_script.side_effect = [
            json.dumps(["file1.rst"]), # assemble_doclist
            "", # chunk_by_rst.py
            json.dumps({"vectors_stored": 10}) # embed_pytorch.py
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
        # For PyTorch, table is now also NOT dropped by default (Resume mode)
        mock_db.drop_table.assert_not_called()
        
        # Verify indexing
        mock_db.open_table.assert_called_with("reference_docs")
        mock_table.create_index.assert_called_with(
            metric="cosine", 
            num_partitions=64, 
            num_sub_vectors=32
        )

    @patch("docs_vectordb.generate_vectordb.run_script")
    @patch("docs_vectordb.generate_vectordb.lancedb.connect")
    @patch("docs_vectordb.generate_vectordb.shutil.rmtree")
    @patch("docs_vectordb.generate_vectordb.Path.mkdir")
    @patch("docs_vectordb.generate_vectordb.Path.open", new_callable=mock_open)
    @patch("docs_vectordb.generate_vectordb.subprocess.Popen")
    @patch("docs_vectordb.generate_vectordb.requests.get")
    @patch("docs_vectordb.generate_vectordb.requests.post")
    def test_worker_single_startup_and_primer(
        self, mock_post, mock_get, mock_popen, mock_path_open, mock_mkdir, mock_rmtree, mock_lancedb, mock_run_script
    ):
        from docs_vectordb.generate_vectordb import main
        from click.testing import CliRunner
        
        # Mock responses
        mock_run_script.side_effect = [
            json.dumps(["file1.rst"]), # assemble_doclist
            "", # chunk_by_rst.py
            json.dumps({"vectors_stored": 10}) # embed_pytorch.py
        ]
        
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        mock_db.list_tables.return_value = ["reference_docs"]
        
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        
        mock_popen.return_value = MagicMock()
        mock_get.return_value = MagicMock(status_code=200)
        mock_post.return_value = MagicMock(status_code=200)
        
        runner = CliRunner()
        result = runner.invoke(main, ["--embedder", "pytorch"])
        
        self.assertEqual(result.exit_code, 0)
        
        # Should have called post once for the primer on port 5000
        self.assertEqual(mock_post.call_count, 1)
        
        # It should pass --port 5000 to embed_pytorch.py
        mock_run_script_calls = mock_run_script.call_args_list
        embed_call = mock_run_script_calls[-1]
        args = embed_call[0]
        self.assertIn("--port", args)
        self.assertIn("5000", args)

if __name__ == "__main__":
    unittest.main()
