import unittest
from click.testing import CliRunner
from unittest.mock import patch
from docs_vectordb.embedding_server import main

class TestEmbeddingServer(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("docs_vectordb.embedding_server.serve")
    def test_server_port_arg(self, mock_serve):
        # Verify that passing a port argument works and calls serve with the correct port
        result = self.runner.invoke(main, ["--port", "5005"])
        self.assertEqual(result.exit_code, 0)
        # Check that the first argument to serve is the 'app' and port is 5005
        args, kwargs = mock_serve.call_args
        self.assertEqual(kwargs["port"], 5005)

if __name__ == "__main__":
    unittest.main()
