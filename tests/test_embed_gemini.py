import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from pathlib import Path
import asyncio
import numpy as np
from docs_vectordb.embed_gemini import RateLimiter, normalize_l2, embed_and_store_gemini

class TestEmbedGemini(unittest.TestCase):
    def test_normalize_l2(self):
        embeddings = np.array([[1.0, 1.0], [0.0, 0.0]])
        normalized = normalize_l2(embeddings)
        self.assertAlmostEqual(np.linalg.norm(normalized[0]), 1.0)
        self.assertEqual(normalized[1], [0.0, 0.0]) # Handled zero norm

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_rate_limiter(self, mock_sleep):
        limiter = RateLimiter(tpm_limit=100)
        # First call should not wait
        asyncio.run(limiter.wait_if_needed(50))
        mock_sleep.assert_not_called()
        
        # Second call that exceeds limit should wait
        asyncio.run(limiter.wait_if_needed(60))
        mock_sleep.assert_called_once()

    @patch("docs_vectordb.embed_gemini.genai.Client")
    @patch("docs_vectordb.embed_gemini.lancedb.connect")
    @patch("docs_vectordb.embed_gemini.print_log")
    def test_embed_and_store_gemini(self, mock_print, mock_lancedb, mock_genai):
        # Setup mocks
        mock_client = MagicMock()
        mock_genai.return_value = mock_client
        
        # Mock response from embed_content
        mock_response = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = [0.1, 0.2, 0.3]
        mock_response.embeddings = [mock_emb]
        # In public Gemini API, metadata is often None or lacks billable_character_count
        mock_response.metadata = None 
        
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)
        
        mock_db = MagicMock()
        mock_lancedb.return_value = mock_db
        mock_db.list_tables.return_value = []
        
        # Prepare a fake chunk file
        test_chunk_file = Path("test_chunks.json")
        with test_chunk_file.open("w", encoding="utf-8") as f:
            # "chunk1" is 6 characters
            json.dump(["chunk1"], f)
            
        try:
            stats = asyncio.run(embed_and_store_gemini(
                file_paths=[str(test_chunk_file)],
                model="test-model",
                dimension=None,
                tpm_limit=1000
            ))
            
            self.assertEqual(stats["vectors_stored"], 1)
            mock_db.create_table.assert_called_once()
        finally:
            if test_chunk_file.exists():
                test_chunk_file.unlink()

if __name__ == "__main__":
    unittest.main()
