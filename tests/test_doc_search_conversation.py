import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json
from docs_vectordb.doc_search_conversation import save_history, load_history, GeminiModel
from google.genai import types

class TestDocSearchConversation(unittest.TestCase):
    def test_save_history(self):
        # Create mock history with Content objects
        mock_entry = MagicMock()
        mock_entry.role = "user"
        mock_part = MagicMock()
        mock_part.text = "hello"
        mock_entry.parts = [mock_part]
        history = [mock_entry]

        with patch("builtins.open", mock_open()) as m:
            save_history(history)
            m.assert_called()
            
            # Verify the structure being written
            handle = m()
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            data = json.loads(written_data)
            self.assertEqual(data[0]["role"], "user")
            self.assertEqual(data[0]["text"], "hello")

    def test_load_history(self):
        mock_data = [{"role": "user", "text": "hi"}]
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))):
            history = load_history("dummy.json")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].role, "user")
            self.assertEqual(history[0].parts[0].text, "hi")

    @patch("docs_vectordb.doc_search_conversation.genai.Client")
    @patch("docs_vectordb.doc_search_conversation.get_context")
    @patch("docs_vectordb.doc_search_conversation.console.input")
    def test_switch_model_preserves_history(self, mock_input, mock_context, mock_client):
        from docs_vectordb.doc_search_conversation import run_chat
        
        # We need to mock the sequence of events to simulate a /model switch
        # 1. User asks /model
        # 2. User selects model 2
        # 3. User exits
        mock_input.side_effect = ["/model", "2", "exit"]
        
        mock_chat = MagicMock()
        mock_chat.get_history.return_value = [MagicMock(spec=types.Content)]
        
        mock_instance = mock_client.return_value
        mock_instance.chats.create.return_value = mock_chat
        
        run_chat()
        
        # Verify chats.create was called at least twice (initial + switch)
        # And the second call passed the history
        self.assertGreaterEqual(mock_instance.chats.create.call_count, 2)
        history_passed = mock_instance.chats.create.call_args_list[1].kwargs.get("history")
        self.assertIsNotNone(history_passed)

if __name__ == "__main__":
    unittest.main()
