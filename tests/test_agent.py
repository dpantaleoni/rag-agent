"""Tests for Agent system with ReAct loop"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from src.agent import Agent
from src.tools import Tool


class TestAgent:
    """Test Agent class with iterative ReAct loop"""

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
    def mock_agent(self, mock_tool):
        """Create Agent instance with mocked Ollama client"""
        with patch('src.agent.ollama.Client') as mock_client:
            agent = Agent(model="test-model", tools=[mock_tool])
            agent.client = mock_client.return_value
            yield agent

    def test_initialization(self, mock_tool):
        """Test agent initialization with tools"""
        with patch('src.agent.ollama.Client'):
            agent = Agent(model="test-model", tools=[mock_tool])
            assert agent.model == "test-model"
            assert agent.backend == "ollama"
            assert "mock_search" in agent.tools
            assert len(agent.tools) == 1

    def test_initialization_no_tools(self):
        """Test agent initialization without tools"""
        with patch('src.agent.ollama.Client'):
            agent = Agent(model="test-model")
            assert agent.model == "test-model"
            assert len(agent.tools) == 0

    def test_initialization_llamacpp(self, mock_tool):
        """Test llama.cpp backend initialization"""
        mock_llama = Mock()
        with patch('src.agent.get_config') as mock_get_config, patch.dict(
            'sys.modules',
            {'llama_cpp': Mock(Llama=mock_llama)}
        ):
            mock_config = Mock()
            mock_config.get.side_effect = lambda key, default=None: {
                "llm.llamacpp_path": "/models/test.gguf",
                "llm.llamacpp_n_ctx": 4096,
                "llm.llamacpp_n_gpu_layers": -1,
                "llm.llamacpp_n_threads": 8,
                "llm.llamacpp_use_mlock": True,
            }.get(key, default)
            mock_get_config.return_value = mock_config

            agent = Agent(backend="llamacpp", tools=[mock_tool])

            assert agent.backend == "llamacpp"
            assert agent.model == "/models/test.gguf"
            assert "mock_search" in agent.tools

    def test_get_tool_descriptions(self, mock_agent):
        """Test tool description generation"""
        descriptions = mock_agent._get_tool_descriptions()
        assert "mock_search" in descriptions
        assert "A mock search tool" in descriptions

    def test_parse_action_valid(self, mock_agent):
        """Test parsing valid action JSON"""
        response = '''
        {
            "thought": "I need to search",
            "action": "mock_search",
            "action_input": {"query": "test"}
        }
        '''
        action = mock_agent._parse_action(response)
        assert action["thought"] == "I need to search"
        assert action["action"] == "mock_search"
        assert action["action_input"]["query"] == "test"

    def test_parse_action_with_extra_text(self, mock_agent):
        """Test parsing JSON with surrounding text"""
        response = '''
        Here is my response:
        {
            "thought": "Testing",
            "action": "final_answer",
            "action_input": {"answer": "Done"}
        }
        That's it!
        '''
        action = mock_agent._parse_action(response)
        assert action["action"] == "final_answer"

    def test_parse_action_invalid(self, mock_agent):
        """Test parsing invalid JSON"""
        with pytest.raises(ValueError, match="No JSON found in response"):
            mock_agent._parse_action("Not valid JSON")

    def test_parse_action_missing_action(self, mock_agent):
        """Test parsing JSON without action field"""
        with pytest.raises(ValueError, match="Missing 'action' field"):
            mock_agent._parse_action('{"thought": "test"}')

    def test_execute_tool_success(self, mock_agent):
        """Test successful tool execution"""
        result = mock_agent._execute_tool("mock_search", {"query": "test"})
        assert "Mock result for: test" in result

    def test_execute_tool_unknown(self, mock_agent):
        """Test executing unknown tool"""
        result = mock_agent._execute_tool("unknown_tool", {})
        assert "Unknown tool" in result
        assert "unknown_tool" in result

    def test_execute_tool_error(self, mock_agent):
        """Test tool execution error handling"""
        # Create a tool that raises an error
        def error_func(**kwargs):
            raise ValueError("Tool error")
        
        error_tool = Tool(
            name="error_tool",
            func=error_func,
            description="A tool that errors"
        )
        mock_agent.tools["error_tool"] = error_tool
        
        result = mock_agent._execute_tool("error_tool", {})
        assert "Error:" in result
        assert "Tool error" in result

    def test_run_validation(self, mock_agent):
        """Test run input validation"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            mock_agent.run("")

        with pytest.raises(ValueError, match="Query exceeds maximum length"):
            mock_agent.run("x" * 11000)

    def test_run_final_answer(self, mock_agent):
        """Test agent run with immediate final answer"""
        # Mock LLM to return final answer immediately
        mock_agent.client.chat = Mock(
            return_value={
                "message": {
                    "content": '''
                    {
                        "thought": "I can answer directly",
                        "action": "final_answer",
                        "action_input": {"answer": "The answer is 42"}
                    }
                    '''
                }
            }
        )
        
        result = mock_agent.run("What is the answer?")
        assert result == "The answer is 42"

    def test_run_with_tool_use(self, mock_agent):
        """Test agent run with tool usage"""
        # Mock LLM responses: first use tool, then final answer
        mock_agent.client.chat = Mock(
            side_effect=[
                {
                    "message": {
                        "content": '''
                        {
                            "thought": "I need to search",
                            "action": "mock_search",
                            "action_input": {"query": "test query"}
                        }
                        '''
                    }
                },
                {
                    "message": {
                        "content": '''
                        {
                            "thought": "Now I can answer",
                            "action": "final_answer",
                            "action_input": {"answer": "Based on search: test query"}
                        }
                        '''
                    }
                }
            ]
        )
        
        result = mock_agent.run("Search for something")
        assert "Based on search" in result

    def test_run_max_iterations(self, mock_agent):
        """Test agent hitting max iterations"""
        # Mock LLM to never return final answer
        mock_agent.client.chat = Mock(
            return_value={
                "message": {
                    "content": '''
                    {
                        "thought": "Keep searching",
                        "action": "mock_search",
                        "action_input": {"query": "more"}
                    }
                    '''
                }
            }
        )
        
        result = mock_agent.run("Test query", max_iterations=2)
        assert "couldn't complete the task" in result.lower()

    def test_run_parsing_error_recovery(self, mock_agent):
        """Test agent recovery from parsing errors"""
        # Mock LLM: first invalid JSON, then valid final answer
        mock_agent.client.chat = Mock(
            side_effect=[
                {"message": {"content": "Invalid JSON response"}},
                {
                    "message": {
                        "content": '''
                        {
                            "thought": "Fixed it",
                            "action": "final_answer",
                            "action_input": {"answer": "Recovered"}
                        }
                        '''
                    }
                }
            ]
        )
        
        result = mock_agent.run("Test query")
        assert result == "Recovered"

    def test_run_emits_trace_messages(self, mock_agent):
        """Test agent trace output includes iterations, LLM response, and tool calls."""
        trace_callback = Mock()
        mock_agent.trace_callback = trace_callback
        mock_agent.client.chat = Mock(
            side_effect=[
                {
                    "message": {
                        "content": '''
                        {
                            "thought": "I need to search",
                            "action": "mock_search",
                            "action_input": {"query": "test query"}
                        }
                        '''
                    }
                },
                {
                    "message": {
                        "content": '''
                        {
                            "thought": "Now I can answer",
                            "action": "final_answer",
                            "action_input": {"answer": "Done"}
                        }
                        '''
                    }
                }
            ]
        )

        result = mock_agent.run("Search for something", verbose=False)

        assert result == "Done"
        trace_callback.assert_any_call("\n============================================================")
        trace_callback.assert_any_call("Iteration 1/5")
        trace_callback.assert_any_call("\nLLM Response:\n\n                        {\n                            \"thought\": \"I need to search\",\n                            \"action\": \"mock_search\",\n                            \"action_input\": {\"query\": \"test query\"}\n                        }\n                        ")
        trace_callback.assert_any_call("Action: mock_search")
        trace_callback.assert_any_call("Calling tool: mock_search")
        trace_callback.assert_any_call("\nObservation: Mock result for: test query...")

    def test_sanitize_input(self, mock_agent):
        """Test input sanitization"""
        dirty = "test\x00\x01\x02text"
        clean = mock_agent._sanitize_input(dirty)
        assert "\x00" not in clean
        assert "\x01" not in clean
        assert "testtext" in clean


# Made with Bob
