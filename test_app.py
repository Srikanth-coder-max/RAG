import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json


class TestAppUtilityFunctions:
    """Test utility functions used in the Streamlit app."""

    def test_format_chat_history_empty(self):
        """Test formatting empty chat history."""
        from app import _format_chat_history
        
        result = _format_chat_history([])
        assert result == "No prior conversation."

    def test_format_chat_history_single_message(self):
        """Test formatting chat history with single message."""
        from app import _format_chat_history
        
        messages = [{"role": "user", "content": "Hello"}]
        result = _format_chat_history(messages)
        
        assert "User: Hello" in result

    def test_format_chat_history_multiple_messages(self):
        """Test formatting chat history with multiple messages."""
        from app import _format_chat_history
        
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Tell me more."},
        ]
        result = _format_chat_history(messages)
        
        assert "User: What is AI?" in result
        assert "Assistant: AI is artificial intelligence." in result
        assert "User: Tell me more." in result

    def test_format_chat_history_limits_to_last_10(self):
        """Test that chat history is limited to last 10 messages."""
        from app import _format_chat_history
        
        # Create 15 messages
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(15)
        ]
        
        result = _format_chat_history(messages)
        
        # Last message should be present
        assert "Message 14" in result
        # First message should not be present
        assert "Message 0" not in result

    def test_export_chat_markdown_basic(self):
        """Test exporting chat as markdown."""
        from app import _export_chat_markdown
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        result = _export_chat_markdown(messages)
        
        assert "# Chat History" in result
        assert "## User" in result
        assert "Hello" in result
        assert "## Assistant" in result
        assert "Hi there!" in result
        assert "Generated:" in result

    def test_export_chat_markdown_empty(self):
        """Test exporting empty chat history."""
        from app import _export_chat_markdown
        
        result = _export_chat_markdown([])
        
        assert "# Chat History" in result
        assert "Generated:" in result

    def test_export_chat_markdown_formatting(self):
        """Test that markdown formatting is correct."""
        from app import _export_chat_markdown
        
        messages = [
            {"role": "user", "content": "Test message"},
        ]
        
        result = _export_chat_markdown(messages)
        lines = result.split("\n")
        
        # Should have structure
        assert lines[0] == "# Chat History"
        assert lines[1].startswith("Generated:")
        assert "## User" in result

    @patch("app.Path.mkdir")
    def test_ensure_kb_dirs(self, mock_mkdir):
        """Test that KB directories are created."""
        from app import _ensure_kb_dirs
        
        _ensure_kb_dirs()
        
        # mkdir should be called twice (for PDF and index directories)
        assert mock_mkdir.call_count >= 2

    @patch("app.Path.glob")
    def test_list_kb_sources_empty(self, mock_glob):
        """Test listing KB sources when none exist."""
        from app import _list_kb_sources
        
        mock_glob.return_value = []
        
        result = _list_kb_sources()
        
        assert result == []

    @patch("app.Path.glob")
    def test_list_kb_sources_multiple(self, mock_glob):
        """Test listing multiple KB sources."""
        from app import _list_kb_sources
        
        mock_glob.return_value = [
            Path("kb_store/pdfs/doc1.pdf"),
            Path("kb_store/pdfs/doc2.pdf"),
            Path("kb_store/pdfs/doc3.pdf"),
        ]
        
        result = _list_kb_sources()
        
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result
        assert "doc3.pdf" in result
        # Should be sorted
        assert result == sorted(result)

    @patch("app.Path.write_bytes")
    def test_save_uploaded_pdf(self, mock_write):
        """Test saving uploaded PDF."""
        from app import _save_uploaded_pdf
        
        # Create mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test_doc.pdf"
        mock_file.getvalue.return_value = b"PDF content"
        
        result = _save_uploaded_pdf(mock_file)
        
        # Verify file was saved
        mock_write.assert_called_once_with(b"PDF content")
        assert "test_doc.pdf" in result

    @patch("app.FAISS.load_local")
    @patch("app.get_embedding_model")
    def test_load_persistent_vectorstore_exists(self, mock_get_embeddings, mock_load_local):
        """Test loading existing persistent vectorstore."""
        from app import _load_persistent_vectorstore
        
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vectorstore = MagicMock()
        mock_load_local.return_value = mock_vectorstore
        
        with patch("app.KB_INDEX_DIR") as mock_index_dir:
            mock_index_file = MagicMock()
            mock_index_file.exists.return_value = True
            mock_index_dir.__truediv__.return_value = mock_index_file
            
            result = _load_persistent_vectorstore()
            
            mock_get_embeddings.assert_called_once()
            mock_load_local.assert_called_once()

    @patch("app.KB_INDEX_DIR")
    def test_load_persistent_vectorstore_not_exists(self, mock_index_dir):
        """Test loading non-existent vectorstore."""
        from app import _load_persistent_vectorstore
        
        mock_index_file = MagicMock()
        mock_index_file.exists.return_value = False
        mock_index_dir.__truediv__.return_value = mock_index_file
        
        result = _load_persistent_vectorstore()
        
        assert result is None


class TestAppHelperFunctions:
    """Test helper functions for chat and export."""

    def test_format_chat_history_assistant_role(self):
        """Test that assistant role is properly formatted."""
        from app import _format_chat_history
        
        messages = [
            {"role": "assistant", "content": "I am the assistant."}
        ]
        result = _format_chat_history(messages)
        
        assert "Assistant: I am the assistant." in result

    def test_export_chat_markdown_timestamp(self):
        """Test that exported markdown includes timestamp."""
        from app import _export_chat_markdown
        
        messages = [{"role": "user", "content": "test"}]
        result = _export_chat_markdown(messages)
        
        # Should contain generated timestamp
        assert "Generated:" in result
        # Timestamp should be ISO format
        assert "T" in result or any(char.isdigit() for char in result)

    def test_chat_history_preserves_message_order(self):
        """Test that message order is preserved."""
        from app import _format_chat_history
        
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        
        result = _format_chat_history(messages)
        
        # Find positions to verify order
        first_pos = result.find("First")
        second_pos = result.find("Second")
        third_pos = result.find("Third")
        
        assert first_pos < second_pos < third_pos

    def test_export_chat_preserves_all_messages(self):
        """Test that export preserves all messages."""
        from app import _export_chat_markdown
        
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]
        
        result = _export_chat_markdown(messages)
        
        # All messages should be present (unlike history formatter that limits to 10)
        for i in range(20):
            assert f"Message {i}" in result
