import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
from docs_vectordb.doc_search import main, get_context
from click.testing import CliRunner

class TestDocSearch(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("subprocess.run")
    def test_get_context_success(self, mock_run):
        mock_run.return_value.stdout = json.dumps([{"source": "test.rst", "chunk_id": "1", "text": "hello"}])
        mock_run.return_value.returncode = 0
        
        result = get_context("query")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "hello")

    @patch("docs_vectordb.doc_search.get_context")
    @patch("docs_vectordb.doc_search.genai.Client")
    def test_main_flow(self, mock_client, mock_get_context):
        # Mock context retrieval
        mock_get_context.return_value = [{"source": "test.rst", "chunk_id": "1", "text": "context content"}]
        
        # Mock Gemini response
        mock_instance = mock_client.return_value
        mock_response = MagicMock()
        mock_response.text = "AI Answer"
        mock_instance.models.generate_content.return_value = mock_response
        
        result = self.runner.invoke(main, ["What is test?", "--raw"])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("AI Answer", result.output)
        mock_get_context.assert_called_once()

if __name__ == "__main__":
    unittest.main()
