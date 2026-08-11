"""Agentic RAG System - A local RAG system for educational content generation"""

__version__ = "0.1.0"

from .rag import SimpleRAG
from .agent import Agent
from .langchain_agent import LangChainAgent
from .config import Config, get_config

__all__ = ["SimpleRAG", "Agent", "LangChainAgent", "Config", "get_config"]

# Made with Bob
