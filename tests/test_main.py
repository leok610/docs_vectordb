import unittest
from unittest.mock import patch
import io
import main

class TestMain(unittest.TestCase):
    def test_main_output(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            main.main()
            self.assertEqual(fake_out.getvalue().strip(), "Hello from docs-vectordb!")

if __name__ == "__main__":
    unittest.main()
