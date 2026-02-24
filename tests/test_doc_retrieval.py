import unittest
from unittest.mock import MagicMock, patch
import json
from docs_vectordb.doc_retrieval import get_gemini_embedding, normalize_l2

class TestDocRetrieval(unittest.TestCase):
    def test_normalize_l2_single(self):
        vector = [3.0, 4.0]
        normalized = normalize_l2(vector)
        self.assertEqual(normalized, [0.6, 0.8])

    @patch("docs_vectordb.doc_retrieval.genai.Client")
    def test_get_gemini_embedding(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = [0.1, 0.2]
        mock_response.embeddings = [mock_emb]
        
        mock_client.models.embed_content.return_value = mock_response
        
        vector = get_gemini_embedding("test query")
        self.assertEqual(vector, [0.1, 0.2])
        mock_client.models.embed_content.assert_called_once()

    @patch("docs_vectordb.doc_retrieval.lancedb.connect")
    @patch("docs_vectordb.doc_retrieval.get_gemini_embedding")
    @patch("rich_click.command", lambda *args, **kwargs: lambda f: f) # Mock click
    def test_main_logic(self, mock_get_emb, mock_lancedb):
        from docs_vectordb.doc_retrieval import main
        
        mock_get_emb.return_value = [0.1, 0.2]
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        mock_db.list_tables.return_value = ["gemini_reference_docs"]
        
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = [
            {"_distance": 0.1, "source_doc": "doc1", "id": "id1", "text": "text1"}
        ]
        
        with patch("sys.stdout") as mock_stdout:
            # We can't easily call main() because it's decorated by click
            # But we can call the function it wraps if we had access to it.
            # Since main is decorated, we'll call it directly and hope the mock works.
            from click.testing import CliRunner
            runner = CliRunner()
            result = runner.invoke(main, ["query", "--embedder", "gemini"])
            
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            if "error" in data:
                raise RuntimeError(data["error"])
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["chunk_id"], "id1")

if __name__ == "__main__":
    unittest.main()
