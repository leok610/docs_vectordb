import unittest
from unittest.mock import MagicMock, patch
import json
from docs_vectordb.doc_retrieval import main, GEMINI_TABLE, PYTORCH_TABLE

class TestDocRetrievalIntegration(unittest.TestCase):
    
    @patch("docs_vectordb.doc_retrieval.lancedb.connect")
    @patch("docs_vectordb.doc_retrieval.get_gemini_embedding")
    def test_gemini_fallback_to_pytorch_table(self, mock_get_emb, mock_lancedb):
        """Test that Gemini embedder falls back to pytorch table if gemini table is missing."""
        mock_get_emb.return_value = [0.1] * 3072
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        # Only pytorch table exists
        mock_db.list_tables.return_value = [PYTORCH_TABLE]
        
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value.limit.return_value.to_list.return_value = []
        
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(main, ["test query", "--embedder", "gemini"])
        
        self.assertEqual(result.exit_code, 0)
        mock_db.open_table.assert_called_with(PYTORCH_TABLE)

    @patch("docs_vectordb.doc_retrieval.lancedb.connect")
    def test_pytorch_missing_table_failure(self, mock_lancedb):
        """Test that pytorch embedder fails if its table is missing."""
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        # Only gemini table exists
        mock_db.list_tables.return_value = [GEMINI_TABLE]
        
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(main, ["test query", "--embedder", "pytorch"])
        
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("error", data)
        self.assertIn("No suitable table found", data["error"])

    @patch("docs_vectordb.doc_retrieval.lancedb.connect")
    @patch("docs_vectordb.doc_retrieval.get_gemini_embedding")
    def test_dimension_mismatch_detection(self, mock_get_emb, mock_lancedb):
        """Test that dimension mismatch (Gemini 3072 on Pytorch 768 table) is caught."""
        mock_get_emb.return_value = [0.1] * 3072
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        mock_db.list_tables.return_value = [PYTORCH_TABLE]
        
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        # Simulate LanceDB dimension mismatch error
        mock_table.search.side_effect = Exception("lance error: Invalid user input: query dim(3072) doesn't match the column vector vector dim(768)")
        
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(main, ["test query", "--embedder", "gemini"])
        
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("error", data)
        self.assertIn("dim(3072) doesn't match", data["error"])

if __name__ == "__main__":
    unittest.main()
