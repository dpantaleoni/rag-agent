"""Tests for LangChain-based agent implementation"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.langchain_agent import LangChainAgent
from src.tools import Tool


class TestLangChainAgent:
    """Test LangChain Agent class with AgentExecutor"""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing"""
        def mock_func(query: str) -> str:
            return f"Mock result for: {query}"
        
        return Tool(
            name="mock_search",
            func=mock_func,
            description="A mock search tool for testing"
        )

    @pytest.fixture
    def mock_langchain_agent(self, mock_tool):
        """Create LangChain Agent instance with mocked components"""
        with patch('src.langchain_agent.ChatOllama') as mock_chat:
            mock_llm = Mock()
            mock_chat.return_value = mock_llm
            
            agent = LangChainAgent(
                model="test-model",
                tools=[mock_tool],
                verbose=False
            )
            yield agent

    def test_initialization(self, mock_tool):
        """Test LangChain agent initialization with tools"""
        with patch('src.langchain_agent.ChatOllama'):
            agent = LangChainAgent(model="test-model", tools=[mock_tool])
            assert agent.model == "test-model"
            assert agent.backend == "ollama"
            assert len(agent.langchain_tools) == 1
            assert agent.langchain_tools[0].name == "mock_search"

    def test_initialization_no_tools(self):
        """Test LangChain agent initialization without tools"""
        with patch('src.langchain_agent.ChatOllama'):
            agent = LangChainAgent(model="test-model")
            assert agent.model == "test-model"
            assert len(agent.langchain_tools) == 0
            assert agent.agent_executor is None

    def test_initialization_llamacpp(self, mock_tool):
        """Test llama.cpp backend initialization"""
        with patch('src.langchain_agent.get_config') as mock_get_config, \
             patch('langchain_community.llms.LlamaCpp') as mock_llama:
            
            mock_config = Mock()
            mock_config.get.side_effect = lambda key, default=None: {
                "llm.llamacpp_model": "/models/test.gguf",
                "llm.llamacpp_n_ctx": 4096,
                "llm.llamacpp_n_gpu_layers": -1,
                "llm.llamacpp_n_threads": 8,
                "llm.llamacpp_use_mlock": True,
                "llm.temperature": 0.7,
                "llm.max_tokens": 2000,
            }.get(key, default)
            mock_get_config.return_value = mock_config

            agent = LangChainAgent(backend="llamacpp", tools=[mock_tool])

            assert agent.backend == "llamacpp"
            assert agent.model == "/models/test.gguf"
            assert len(agent.langchain_tools) == 1

    def test_run_validation(self, mock_langchain_agent):
        """Test run input validation"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            mock_langchain_agent.run("")

        with pytest.raises(ValueError, match="Query exceeds maximum length"):
            mock_langchain_agent.run("x" * 11000)

    def test_run_no_tools_error(self):
        """Test run without tools configured"""
        with patch('src.langchain_agent.ChatOllama'):
            agent = LangChainAgent(model="test-model")
            
            with pytest.raises(ValueError, match="Agent not configured with tools"):
                agent.run("Test query")

    def test_run_success(self, mock_langchain_agent):
        """Test successful agent run"""
        # Mock the agent executor to return LangGraph format
        mock_message = Mock()
        mock_message.type = 'ai'
        mock_message.content = "Final answer from agent"
        
        mock_langchain_agent.agent_executor = Mock()
        mock_langchain_agent.agent_executor.invoke = Mock(
            return_value={"messages": [mock_message]}
        )
        
        result = mock_langchain_agent.run("Test query")
        assert result == "Final answer from agent"
        mock_langchain_agent.agent_executor.invoke.assert_called_once()

    def test_run_with_error(self, mock_langchain_agent):
        """Test agent run with error handling"""
        # Mock the agent executor to raise an error
        mock_langchain_agent.agent_executor = Mock()
        mock_langchain_agent.agent_executor.invoke = Mock(
            side_effect=Exception("Test error")
        )
        
        result = mock_langchain_agent.run("Test query")
        assert "Error" in result
        assert "Test error" in result

    def test_get_intermediate_steps(self, mock_langchain_agent):
        """Test getting intermediate steps"""
        # Mock the agent executor with LangGraph format
        mock_message = Mock()
        mock_message.type = 'ai'
        mock_message.content = "Final answer"
        
        mock_langchain_agent.agent_executor = Mock()
        mock_langchain_agent.agent_executor.invoke = Mock(
            return_value={
                "messages": [
                    ("user", "Test query"),
                    mock_message
                ]
            }
        )
        
        result = mock_langchain_agent.get_intermediate_steps("Test query")
        assert result["output"] == "Final answer"
        assert len(result["intermediate_steps"]) == 2


    def test_get_intermediate_steps_no_tools(self):
        """Test get_intermediate_steps without tools"""
        with patch('src.langchain_agent.ChatOllama'):
            agent = LangChainAgent(model="test-model")
            
            with pytest.raises(ValueError, match="Agent not configured with tools"):
                agent.get_intermediate_steps("Test query")

    def test_tool_conversion(self, mock_tool):
        """Test conversion of custom Tool to LangChain Tool"""
        with patch('src.langchain_agent.ChatOllama'):
            agent = LangChainAgent(model="test-model", tools=[mock_tool])
            
            # Check that tool was converted properly
            lc_tool = agent.langchain_tools[0]
            assert lc_tool.name == "mock_search"
            assert lc_tool.description == "A mock search tool for testing"
            
            # Test that the function works
            result = lc_tool.func(query="test")
            assert "Mock result for: test" in result

    def test_verbose_mode(self, mock_tool):
        """Test verbose mode initialization"""
        with patch('src.langchain_agent.ChatOllama'):
            agent = LangChainAgent(
                model="test-model",
                tools=[mock_tool],
                verbose=True
            )
            assert agent.verbose is True

    def test_max_iterations_config(self, mock_tool):
        """Test max_iterations configuration"""
        with patch('src.langchain_agent.ChatOllama'):
            agent = LangChainAgent(
                model="test-model",
                tools=[mock_tool],
                max_iterations=5
            )
            assert agent.max_iterations == 5
            # LangGraph agents don't expose max_iterations directly
            assert agent.agent_executor is not None

    def test_langgraph_agent_created(self, mock_tool):
        """Test that LangGraph agent is properly created"""
        with patch('src.langchain_agent.ChatOllama'):
            agent = LangChainAgent(
                model="test-model",
                tools=[mock_tool]
            )
            # Verify agent executor was created
            assert agent.agent_executor is not None
            # Verify tools were converted
            assert len(agent.langchain_tools) == 1


# Made with Bob