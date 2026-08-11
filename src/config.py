"""Configuration management for Agentic RAG System"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Config:
    """Configuration loader and manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return self._get_defaults()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else self._get_defaults()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults", exc_info=True)
            return self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            "llm": {
                "ollama_model": "gemma4:26b",
                "llamacpp_model": "",
                "llamacpp_path": "",
                "llamacpp_n_ctx": 4096,
                "llamacpp_n_gpu_layers": -1,
                "llamacpp_n_threads": 8,
                "llamacpp_use_mlock": True,
                "mlx_model": "/Users/dominikpantaleoniibm/Documents/dev/models/granite-4.1-30b-nvfp4",
                "mlx_max_tokens": 2000,
                "mlx_temp": 0.7,
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "rag": {
                "db_path": "data/vectors.db",
                "embedding_model": "nomic-embed-text",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k": 5
            },
            "podcast": {
                "default_duration": 5,
                "host_voice": "voices/en_US-lessac-medium.onnx",
                "expert_voice": "voices/en_US-libritts-high.onnx",
                "pause_duration": 300
            },
            "quiz": {
                "default_questions": 5,
                "difficulty": "medium"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def llm(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self._config.get("llm", {})
    
    @property
    def rag(self) -> Dict[str, Any]:
        """Get RAG configuration"""
        return self._config.get("rag", {})
    
    @property
    def podcast(self) -> Dict[str, Any]:
        """Get podcast configuration"""
        return self._config.get("podcast", {})
    
    @property
    def quiz(self) -> Dict[str, Any]:
        """Get quiz configuration"""
        return self._config.get("quiz", {})


# Global config instance
_config = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get global configuration instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

# Made with Bob
