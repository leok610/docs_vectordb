import unittest
from click.testing import CliRunner
import importlib
import json

# List of all entry points and main scripts to check
COMMANDS = {
    "assemble_doclist": "src.docs_vectordb.assemble_doclist",
    "chunk_by_indents": "src.docs_vectordb.chunk_by_indents",
    "chunk_by_rst": "src.docs_vectordb.chunk_by_rst",
    "chunk_by_md": "src.docs_vectordb.chunk_by_md",
    "chunk_by_txt": "src.docs_vectordb.chunk_by_txt",
    "doc_retrieval": "src.docs_vectordb.doc_retrieval",
    "embed_and_store": "src.docs_vectordb.embed_and_store",
    "embed_gemini": "src.docs_vectordb.embed_gemini",
    "generate_vectordb": "src.docs_vectordb.generate_vectordb",
    "healthcheck": "src.docs_vectordb.healthcheck"
}

class TestCLIConsistency(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help_flags_consistency(self):
        """Verify that all scripts support -h, -?, and --help."""
        for name, module_path in COMMANDS.items():
            module = importlib.import_module(module_path)
            main_cmd = getattr(module, "main")
            
            # Check help output for each flag
            for flag in ["-h", "-?", "--help"]:
                result = self.runner.invoke(main_cmd, [flag])
                self.assertEqual(result.exit_code, 0, f"Command {name} failed to handle {flag}")
                self.assertIn("Usage:", result.output, f"Command {name} help output for {flag} is invalid")

    def test_embedder_option_consistency(self):
        """Verify that scripts using an embedder option use the name '--embedder' consistently."""
        scripts_with_embedder = ["doc_retrieval", "generate_vectordb"]
        
        for name in scripts_with_embedder:
            module = importlib.import_module(COMMANDS[name])
            main_cmd = getattr(module, "main")
            
            # Use click's introspection to find the option
            params = {p.name: p for p in main_cmd.params}
            self.assertIn("embedder", params, f"Command {name} is missing the 'embedder' option")
            self.assertIn("--embedder", params["embedder"].opts, f"Command {name} uses inconsistent flag for embedder")

    def test_variable_matching(self):
        """Verify that Click parameters match the function arguments to prevent runtime errors."""
        for name, module_path in COMMANDS.items():
            module = importlib.import_module(module_path)
            main_cmd = getattr(module, "main")
            
            # click.Command.callback is the actual function being called
            func = main_cmd.callback
            import inspect
            sig = inspect.signature(func)
            func_args = set(sig.parameters.keys())
            
            click_params = set(p.name for p in main_cmd.params)
            
            # Some parameters might be added by context or decorators, 
            # but usually they should match 1:1 for basic scripts.
            self.assertTrue(click_params.issubset(func_args) or func_args.issubset(click_params),
                            f"Mismatch in {name}: CLI params {click_params} vs Function args {func_args}")

if __name__ == "__main__":
    unittest.main()
