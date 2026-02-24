import unittest
from pathlib import Path
import json
import os
import shutil
from docs_vectordb.chunking_utils import split_long_unit, write_chunks_to_json, load_targets

class TestChunkingUtils(unittest.TestCase):
    def test_split_long_unit_no_split(self):
        unit = ["line1", "line2", "line3"]
        result = split_long_unit(unit, max_lines=5)
        self.assertEqual(result, [unit])

    def test_split_long_unit_with_split(self):
        unit = [f"line{i}" for i in range(10)]
        # max_lines=6, overlap=2
        # First chunk: 0-6
        # Next start: 6 - 2 = 4
        # Second chunk: 4-10
        result = split_long_unit(unit, max_lines=6, overlap=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [f"line{i}" for i in range(6)])
        self.assertEqual(result[1], [f"line{i}" for i in range(4, 10)])

    def test_write_chunks_to_json(self):
        test_dir = Path("test_output")
        test_dir.mkdir(exist_ok=True)
        output_path = test_dir / "test_chunks.json"
        chunks = ["chunk1", "chunk2"]
        
        count = write_chunks_to_json(chunks, output_path)
        self.assertEqual(count, 2)
        self.assertTrue(output_path.exists())
        
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["chunks"], chunks)
        
        shutil.rmtree(test_dir)

    def test_write_chunks_to_json_empty(self):
        test_dir = Path("test_output_empty")
        test_dir.mkdir(exist_ok=True)
        output_path = test_dir / "empty.json"
        
        count = write_chunks_to_json([], output_path)
        self.assertEqual(count, 0)
        self.assertFalse(output_path.exists())
        
        shutil.rmtree(test_dir)

    def test_load_targets_list(self):
        targets = ("file1.rst", "file2.md")
        result = load_targets(targets)
        self.assertEqual(result, [Path("file1.rst"), Path("file2.md")])

    def test_load_targets_json(self):
        json_file = Path("targets.json")
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(["file1.rst", "file2.md"], f)
        
        result = load_targets((str(json_file),))
        self.assertEqual(result, [Path("file1.rst"), Path("file2.md")])
        
        json_file.unlink()

    def test_load_targets_empty(self):
        self.assertEqual(load_targets(()), [])

if __name__ == "__main__":
    unittest.main()
