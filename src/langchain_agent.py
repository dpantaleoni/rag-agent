"""LangChain-based agent implementation with LangGraph ReAct agent"""

import logging
from typing import Optional, List, Dict, Any, Callable

from langchain_ollama import ChatOllama
from langchain_core.tools import Tool as LangChainTool
from langgraph.prebuilt import create_react_agent

from .config import get_config
from .tools import Tool

logger = logging.getLogger(__name__)


class LangChainAgent:
    """LangChain-based agent with LangGraph's ReAct implementation
    
    This agent uses LangGraph's create_react_agent which provides:
    - Iterative reasoning loop (ReAct pattern)
    - Tool selection and execution
    - Error handling and retries
    - State management via graph-based execution
    
    This provides a comparison to the manual implementation in Agent class.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        backend: str = "ollama",
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        trace_callback: Optional[Callable[[str], None]] = None
    ):
        """Initialize the LangChain agent with specified backend, model, and tools.
        
        Args:
            model: Backend-specific model identifier
            backend: LLM backend to use ("ollama" or "llamacpp")
            tools: List of Tool objects available to the agent
            max_iterations: Maximum iterations for the agent loop
            verbose: If True, print intermediate steps
            trace_callback: Optional callback for streaming reasoning/action traces
            
        Note:
            MLX backend is not supported with LangChain due to MLXPipeline lacking
            bind_tools support required by LangGraph. Use the direct Agent class
            with --backend mlx instead.
        """
        self.backend = backend
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.trace_callback = trace_callback
        config = get_config()
        
        # Initialize LLM
        if backend == "ollama":
            self.model = model or config.get("llm.ollama_model", "gemma4:26b")
            self.llm = ChatOllama(
                model=self.model,
                temperature=config.get("llm.temperature", 0.7),
                num_predict=config.get("llm.max_tokens", 2000)
            )
        elif backend == "llamacpp":
            try:
                from langchain_community.llms import LlamaCpp
            except ImportError as exc:
                raise ImportError(
                    "langchain-community is required for the 'llamacpp' backend. "
                    "Install it with `uv add langchain-community`."
                ) from exc
            
            self.model = model or config.get("llm.llamacpp_model")
            if not self.model:
                raise ValueError(
                    "No llama.cpp model configured. Set llm.llamacpp_model in config.yaml "
                    "or pass a model path with --model."
                )
            
            self.llm = LlamaCpp(
                model_path=self.model,
                n_ctx=config.get("llm.llamacpp_n_ctx", 4096),
                n_gpu_layers=config.get("llm.llamacpp_n_gpu_layers", -1),
                n_threads=config.get("llm.llamacpp_n_threads", 8),
                use_mlock=config.get("llm.llamacpp_use_mlock", True),
                temperature=config.get("llm.temperature", 0.7),
                max_tokens=config.get("llm.max_tokens", 2000),
                verbose=False
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from: ollama, llamacpp")
        
        # Convert custom Tool objects to LangChain Tool objects
        self.langchain_tools = []
        if tools:
            for tool in tools:
                # Create LangChain-compatible tool
                lc_tool = LangChainTool(
                    name=tool.name,
                    func=tool.func,
                    description=tool.description
                )
                self.langchain_tools.append(lc_tool)
        
        # Create LangGraph ReAct agent
        if self.langchain_tools:
            self.agent_executor = create_react_agent(
                model=self.llm,
                tools=self.langchain_tools
            )
        else:
            self.agent_executor = None
            logger.warning("No tools provided - agent will not be able to use tools")
        
        logger.info(
            "LangChain Agent initialized with backend=%s model=%s tools=%s",
            self.backend,
            self.model,
            [t.name for t in self.langchain_tools]
        )
    def _emit_trace(self, message: str, verbose: Optional[bool] = None) -> None:
        """Emit a trace message to the configured callback and optional stdout."""
        should_print = self.verbose if verbose is None else verbose
        if self.trace_callback:
            self.trace_callback(message)
        if should_print:
            print(message)

    def _format_intermediate_step(self, message: Any, index: int) -> Optional[str]:
        """Format a LangGraph intermediate message for trace output."""
        message_type = getattr(message, "type", None)

        if message_type == "ai":
            content = getattr(message, "content", "")
            return f"\nLLM Response {index}:\n{content}"

        if message_type == "tool":
            tool_name = getattr(message, "name", "unknown_tool")
            content = getattr(message, "content", "")
            return f"Tool called: {tool_name}\nObservation: {content}"

        if isinstance(message, tuple) and len(message) >= 2:
            role, content = message[0], message[1]
            if role == "ai":
                return f"\nLLM Response {index}:\n{content}"
            if role == "tool":
                return f"Tool called: unknown_tool\nObservation: {content}"

        return None

    def generate(self, prompt: str, role: str = "assistant", context: Optional[str] = None, temperature: float = 0.7) -> str:
        """Direct text generation without tool calling
        
        Args:
            prompt: The generation prompt
            role: Role context (not currently used)
            context: Optional context to include in the prompt
            temperature: Sampling temperature
            
        Returns:
            Generated text from the LLM
        """
        full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
        
        response = self.llm.invoke(full_prompt)
        
        # Extract string content from response
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Join list items if it's a list
                return ' '.join(str(item) for item in content)
            else:
                return str(content)
        else:
            return str(response)
    
    
    def run(
        self,
        query: str,
        temperature: Optional[float] = None,
        verbose: Optional[bool] = None
    ) -> str:
        """Run the agent with LangGraph's ReAct implementation
        
        This method uses LangGraph's built-in iterative loop which:
        1. Reasons about what to do next
        2. Executes tools as needed
        3. Observes results
        4. Repeats until final answer or max iterations
        
        Args:
            query: User's question or task
            temperature: Optional temperature override (not used in LangGraph)
            verbose: Optional verbose override
            
        Returns:
            Final answer string
            
        Raises:
            ValueError: If query is invalid or no tools configured
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > 10000:
            raise ValueError("Query exceeds maximum length of 10000 characters")
        
        if not self.agent_executor:
            raise ValueError("Agent not configured with tools. Provide tools during initialization.")
        
        try:
            effective_verbose = self.verbose if verbose is None else verbose

            # Run the agent executor
            # LangGraph agents use a different interface - they return a dict with messages
            result = self.agent_executor.invoke(
                {"input": query}
            )
            
            # Extract the final message from the agent
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]

                if effective_verbose:
                    self._emit_trace(f"\n{'='*60}", effective_verbose)
                    self._emit_trace("LangChain trace", effective_verbose)
                    self._emit_trace(f"{'='*60}", effective_verbose)
                    for index, msg in enumerate(messages, 1):
                        formatted = self._format_intermediate_step(msg, index)
                        if formatted:
                            self._emit_trace(formatted, effective_verbose)

                # Get the last AI message
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        return msg.content
                    elif isinstance(msg, tuple) and msg[0] == 'ai':
                        return msg[1]
                # Fallback: return last message content
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, 'content'):
                        return last_msg.content
                    elif isinstance(last_msg, tuple):
                        return last_msg[1]
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error running LangChain agent: {e}", exc_info=True)
            return f"Error: Agent encountered an error: {str(e)}"
    
    def get_intermediate_steps(self, query: str) -> Dict[str, Any]:
        """Run agent and return intermediate steps for debugging
        
        Args:
            query: User's question or task
            
        Returns:
            Dictionary with output and intermediate_steps
        """
        if not self.agent_executor:
            raise ValueError("Agent not configured with tools")
        
        try:
            result = self.agent_executor.invoke(
                {"messages": [("user", query)]}
            )
            
            # Extract messages for intermediate steps
            messages = result.get("messages", [])
            output = ""
            
            # Get final output
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'ai':
                    output = msg.content
                    break
                elif isinstance(msg, tuple) and msg[0] == 'ai':
                    output = msg[1]
                    break
            
            return {
                "output": output,
                "intermediate_steps": messages
            }
        except Exception as e:
            logger.error(f"Error getting intermediate steps: {e}", exc_info=True)
            return {
                "output": f"Error: {str(e)}",
                "intermediate_steps": []
            }


# Made with Bob