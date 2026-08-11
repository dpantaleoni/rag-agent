"""Tests for RAG pipeline"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.rag import SimpleRAG


class TestSimpleRAG:
    """Test SimpleRAG class"""
    
    @pytest.fixture
    def mock_rag(self, tmp_path):
        """Create RAG instance with temporary database"""
        db_path = tmp_path / "test_vectors.db"
        with patch('src.rag.ollama.Client'):
            rag = SimpleRAG(str(db_path))
            yield rag
            rag.close()
    
    def test_context_manager(self, tmp_path):
        """Test context manager protocol"""
        db_path = tmp_path / "test_vectors.db"
        with patch('src.rag.ollama.Client'):
            with SimpleRAG(str(db_path)) as rag:
                assert rag.conn is not None
            # Connection should be closed after context
    
    def test_embed_text_validation(self, mock_rag):
        """Test _embed_text input validation"""
        # Empty text
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            mock_rag._embed_text("")
        
        # Too long text
        with pytest.raises(ValueError, match="exceeds embedding model's maximum length"):
            mock_rag._embed_text("x" * 33000)
    
    def test_embed_text_retry_logic(self, mock_rag):
        """Test retry logic in _embed_text"""
        # Mock client to fail twice then succeed
        mock_rag.ollama_client.embeddings = Mock(
            side_effect=[
                Exception("Connection error"),
                Exception("Connection error"),
                {"embedding": [0.1, 0.2, 0.3]}
            ]
        )
        
        result = mock_rag._embed_text("test text", max_retries=3)
        assert result == [0.1, 0.2, 0.3]
        assert mock_rag.ollama_client.embeddings.call_count == 3
    
    def test_retrieve_validation(self, mock_rag):
        """Test retrieve input validation"""
        # Empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            mock_rag.retrieve("")
        
        # Query too long
        with pytest.raises(ValueError, match="Query exceeds maximum length"):
            mock_rag.retrieve("x" * 11000)
        
        # Invalid k
        with pytest.raises(ValueError, match="k must be at least 1"):
            mock_rag.retrieve("test", k=0)
        
        with pytest.raises(ValueError, match="k cannot exceed 100"):
            mock_rag.retrieve("test", k=101)
    
    def test_ingest_validation(self, mock_rag, tmp_path):
        """Test ingest input validation"""
        # Empty path
        with pytest.raises(ValueError, match="docs_path cannot be empty"):
            mock_rag.ingest("")
        
        # Non-existent path
        with pytest.raises(FileNotFoundError):
            mock_rag.ingest(str(tmp_path / "nonexistent"))
        
        # File instead of directory
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        with pytest.raises(ValueError, match="Path must be a directory"):
            mock_rag.ingest(str(test_file))

# Made with Bob
