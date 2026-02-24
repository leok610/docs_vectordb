import unittest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path
from docs_vectordb.embed_pytorch import embed_and_store_pytorch

class TestEmbedPytorch(unittest.TestCase):

    @patch("docs_vectordb.embed_pytorch.lancedb.connect")
    @patch("docs_vectordb.embed_pytorch.requests.get")
    @patch("docs_vectordb.embed_pytorch.requests.post")
    def test_multi_server_parallel_logic(self, mock_post, mock_get, mock_lancedb):
        # Mock health check
        mock_get.return_value.status_code = 200
        
        # Mock embedding response with 2 vectors (matching 2 chunks)
        mock_res = MagicMock()
        mock_res.json.return_value = {"embeddings": [[0.1] * 768, [0.2] * 768]}
        mock_res.status_code = 200
        mock_post.return_value = mock_res
        
        # Mock database
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        mock_db.list_tables.return_value = []
        
        # Create dummy chunk file
        test_chunks = Path("test_parallel_chunks.json")
        with test_chunks.open("w", encoding="utf-8") as f:
            json.dump(["chunk1", "chunk2"], f)
            
        try:
            # Call the core function directly
            stats = embed_and_store_pytorch(
                file_paths=[test_chunks],
                force=True,
                batch_size=1,
                ports="5000,5001"
            )
            
            self.assertEqual(stats["vectors_stored"], 2)
            self.assertGreaterEqual(mock_post.call_count, 1)
            
        finally:
            if test_chunks.exists(): test_chunks.unlink()

    @patch("docs_vectordb.embed_pytorch.lancedb.connect")
    @patch("docs_vectordb.embed_pytorch.requests.get")
    @patch("docs_vectordb.embed_pytorch.requests.post")
    def test_partial_failure_storage(self, mock_post, mock_get, mock_lancedb):
        """Verify that successful docs are stored even if the last doc fails."""
        mock_get.return_value.status_code = 200
        
        # Doc 1 succeeds, Doc 2 fails
        def side_effect(url, json=None, timeout=None):
            m = MagicMock()
            m.status_code = 200
            if "fail" in json["queries"][0]:
                return MagicMock(status_code=500) # Should trigger retry then None
            m.json.return_value = {"embeddings": [[0.1] * 768]}
            return m

        mock_post.side_effect = side_effect
        
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        mock_db.list_tables.return_value = []
        
        doc1 = Path("doc1_chunks.json")
        with doc1.open("w") as f: json.dump(["success"], f)
        doc2 = Path("doc2_chunks.json")
        with doc2.open("w") as f: json.dump(["fail"], f)
        
        try:
            stats = embed_and_store_pytorch(
                file_paths=[doc1, doc2],
                force=True,
                batch_size=10,
                ports="5000"
            )
            
            # Doc 1 has 1 vector, Doc 2 fails. 
            # Total stored should be 1.
            self.assertEqual(stats["vectors_stored"], 1)
            
        finally:
            if doc1.exists(): doc1.unlink()
            if doc2.exists(): doc2.unlink()

if __name__ == "__main__":
    unittest.main()
