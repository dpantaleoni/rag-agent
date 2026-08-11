"""Tests for configuration management"""

import pytest
from pathlib import Path
from src.config import Config, get_config


class TestConfig:
    """Test Config class"""
    
    def test_load_valid_config(self):
        """Test loading valid config file"""
        config = Config("config.yaml")
        assert config.get("llm.ollama_model") is not None
        assert config.get("rag.db_path") is not None
    
    def test_load_missing_config(self, tmp_path):
        """Test loading missing config file uses defaults"""
        missing_path = tmp_path / "missing.yaml"
        config = Config(str(missing_path))
        assert config.get("llm.ollama_model") == "gemma4:26b"
    
    def test_get_with_dot_notation(self):
        """Test dot notation access"""
        config = Config("config.yaml")
        assert config.get("llm.ollama_model") == config.llm["ollama_model"]
    
    def test_get_with_default(self):
        """Test get with default value"""
        config = Config("config.yaml")
        assert config.get("nonexistent.key", "default") == "default"
    
    def test_property_accessors(self):
        """Test property accessors"""
        config = Config("config.yaml")
        assert isinstance(config.llm, dict)
        assert isinstance(config.rag, dict)
        assert isinstance(config.podcast, dict)
        assert isinstance(config.quiz, dict)

# Made with Bob
