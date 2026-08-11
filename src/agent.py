"""Agent system with iterative ReAct loop and tool calling"""

import json
import logging
import re
import time
from typing import Optional, Dict, Any, List, Callable

import ollama
from llama_cpp import Llama

from .config import get_config
from .tools import Tool

logger = logging.getLogger(__name__)

# Input validation constants
MAX_PROMPT_LENGTH = 10000
MAX_CONTEXT_LENGTH = 100000


class Agent:
    """Iterative agent with manual ReAct loop and tool calling
    
    This agent implements the ReAct (Reasoning + Acting) pattern:
    1. Reason: Analyze the current state and decide what to do
    2. Act: Execute a tool based on the reasoning
    3. Observe: Process the tool result
    4. Repeat until task is complete
    
    The agent uses a while loop to iterate through these steps until
    it decides to provide a final answer or reaches max iterations.
    """
    
    # System prompt for ReAct agent
    REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that uses tools to answer questions.

You have access to the following tools:
{tool_descriptions}

To use a tool, respond with a JSON object in this exact format:
{{
    "thought": "your reasoning about what to do next",
    "action": "tool_name",
    "action_input": {{"param": "value"}}
}}

To provide a final answer, use:
{{
    "thought": "I now have enough information to answer",
    "action": "final_answer",
    "action_input": {{"answer": "your complete answer here"}}
}}

IMPORTANT:
- Always think step-by-step
- Use tools to gather information before answering
- Only use final_answer when you have complete information
- Your response must be valid JSON
- Include your reasoning in the "thought" field"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        backend: str = "ollama",
        tools: Optional[List[Tool]] = None,
        trace_callback: Optional[Callable[[str], None]] = None
    ):
        """Initialize the agent with specified backend, model, and tools.
        
        Args:
            model: Backend-specific model identifier
            backend: LLM backend to use ("ollama", "llamacpp", or "mlx")
            tools: List of Tool objects available to the agent
            trace_callback: Optional callback for streaming reasoning/action traces
        """
        self.backend = backend
        self.client: Any = None
        self.tools: Dict[str, Tool] = {}
        self.mlx_tokenizer: Any = None
        self.trace_callback = trace_callback
        config = get_config()
        
        # Register tools
        if tools:
            for tool in tools:
                self.tools[tool.name] = tool
        
        if backend == "ollama":
            self.client = ollama.Client()
            self.model = model or config.get("llm.ollama_model", "granite4.1:3b")
        elif backend == "llamacpp":
            try:
                from llama_cpp import Llama  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise ImportError(
                    "llama-cpp-python is required for the 'llamacpp' backend. "
                    "Install it with `uv add llama-cpp-python`."
                ) from exc
            
            self.model = model or config.get("llm.llamacpp_path")
            if not self.model:
                raise ValueError(
                    "No llama.cpp path configured. Set llm.llamacpp_path in config.yaml "
                    "or pass a model path with --model."
                )
            
            self.client = Llama(
                model_path=self.model,
                n_ctx=config.get("llm.llamacpp_n_ctx", 4096),
                n_gpu_layers=config.get("llm.llamacpp_n_gpu_layers", -1),
                n_threads=config.get("llm.llamacpp_n_threads", 8),
                use_mlock=config.get("llm.llamacpp_use_mlock", True),
                verbose=False
            )
        elif backend == "mlx":
            try:
                from mlx_lm import load, generate  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise ImportError(
                    "mlx-lm is required for the 'mlx' backend. "
                    "Install it with `uv add mlx-lm`. "
                    "Note: MLX only works on Apple Silicon (M1/M2/M3/M4)."
                ) from exc
            
            self.model = model or config.get("llm.mlx_model")
            if not self.model:
                raise ValueError(
                    "No MLX model configured. Set llm.mlx_model in config.yaml "
                    "or pass a model with --model."
                )
            
            # Load MLX model and tokenizer
            logger.info(f"Loading MLX model: {self.model}")
            model_data = load(self.model)
            # MLX load() can return 2 or 3 values (model, tokenizer, [config])
            self.client = model_data[0]
            self.mlx_tokenizer = model_data[1]
            logger.info(f"MLX model loaded successfully")
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from: ollama, llamacpp, mlx")
            
        logger.info(
            "Agent initialized with backend=%s model=%s tools=%s",
            self.backend,
            self.model,
            list(self.tools.keys())
        )
    
    def _get_tool_descriptions(self) -> str:
        """Generate formatted tool descriptions for the system prompt
        
        Returns:
            Formatted string describing all available tools
        """
        descriptions = []
        for tool in self.tools.values():
            desc = f"- {tool.name}: {tool.description}"
            if tool.parameters:
                params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
                desc += f"\n  Parameters: {params}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """Call the LLM with retry logic
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            ConnectionError: If all retry attempts fail
        """
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if self.backend == "ollama":
                    response = self.client.chat(
                        model=self.model,
                        messages=messages,
                        format="json",
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    )
                    return response['message']['content']
                elif self.backend == "llamacpp":
                    response = self.client.create_chat_completion(  # pyright: ignore[reportArgumentType]
                        messages=messages,
                        temperature=temperature,
                        max_tokens=2048,
                        stream = True
                    )
                    return str(response["choices"][0]["message"]["content"])
                elif self.backend == "mlx":
                    from mlx_lm import generate  # pyright: ignore[reportMissingImports]
                    
                    # Format messages into a single prompt
                    prompt_parts = []
                    for msg in messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "system":
                            prompt_parts.append(f"System: {content}")
                        elif role == "user":
                            prompt_parts.append(f"User: {content}")
                        elif role == "assistant":
                            prompt_parts.append(f"Assistant: {content}")
                    
                    prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
                    
                    # Generate with MLX - uses temperature parameter
                    response = generate(
                        model=self.client,
                        tokenizer=self.mlx_tokenizer,
                        prompt=prompt,
                        max_tokens=2048,
                        max_kv_size = 4096,
                        verbose=True
                    )
                    return response
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
        
        raise ConnectionError(
            f"Failed to call LLM after {max_retries} attempts. "
            f"Last error: {last_exception}"
        )
    
    def _parse_action(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract action
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dictionary with thought, action, and action_input
            
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Try to extract JSON from response
            # Handle cases where LLM adds extra text around JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate required fields
                if "action" not in parsed:
                    raise ValueError("Missing 'action' field in response")
                
                return {
                    "thought": parsed.get("thought", ""),
                    "action": parsed["action"],
                    "action_input": parsed.get("action_input", {})
                }
            else:
                raise ValueError("No JSON found in response")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {response}")
            raise ValueError(f"Invalid JSON in response: {e}")
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool with given input
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of parameters for the tool
            
        Returns:
            String result from tool execution
        """
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"
        
        tool = self.tools[tool_name]
        try:
            result = tool.execute(**tool_input)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error executing {tool_name}: {str(e)}"

    def _emit_trace(self, message: str, verbose: bool = False) -> None:
        """Emit a trace message to the configured callback and optional stdout."""
        if self.trace_callback:
            self.trace_callback(message)
        if verbose:
            print(message)

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
        
        if self.backend == "ollama":
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                options={"temperature": temperature}
            )
            return response['message']['content']
        elif self.backend == "llamacpp":
            response = self.client.create_chat_completion(
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature
            )
            return response["choices"][0]["message"]["content"]
        elif self.backend == "mlx":
            from mlx_lm import generate  # pyright: ignore[reportMissingImports]
            
            config = get_config()
            response = generate(
                model=self.client,
                tokenizer=self.mlx_tokenizer,
                prompt=full_prompt,
                max_tokens=config.get("llm.mlx_max_tokens", 2000),
                verbose=True
            )
            return response
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    
    def run(
        self,
        query: str,
        max_iterations: int = 5,
        temperature: float = 0.4,
        verbose: bool = False
    ) -> str:
        """Run the agent with iterative ReAct loop
        
        This is the main agent loop that implements:
        1. Reason: LLM decides what to do next
        2. Act: Execute the chosen tool
        3. Observe: Process the result
        4. Repeat until final_answer or max_iterations
        
        Args:
            query: User's question or task
            max_iterations: Maximum number of reasoning iterations
            temperature: LLM sampling temperature
            verbose: If True, print intermediate steps
            
        Returns:
            Final answer string
            
        Raises:
            ValueError: If query is invalid
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > MAX_PROMPT_LENGTH:
            raise ValueError(f"Query exceeds maximum length of {MAX_PROMPT_LENGTH} characters")
        
        # Initialize conversation history
        system_prompt = self.REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self._get_tool_descriptions()
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Agent loop
        for iteration in range(max_iterations):
            self._emit_trace(f"\n{'='*60}", verbose)
            self._emit_trace(f"Iteration {iteration + 1}/{max_iterations}", verbose)
            self._emit_trace(f"{'='*60}", verbose)
            
            # STEP 1: REASON - Get LLM's next action
            try:
                response = self._call_llm(messages, temperature=temperature)
                
                # STEP 2: PARSE - Extract action from response
                action_data = self._parse_action(response)
                
                thought = action_data["thought"]
                action = action_data["action"]
                action_input = action_data["action_input"]
                
                self._emit_trace(f"\nThought: {thought}", verbose)
                self._emit_trace(f"Action: {action}", verbose)
                self._emit_trace(f"Input: {action_input}", verbose)
                
                # STEP 3: CHECK COMPLETION - Is this the final answer?
                if action == "final_answer":
                    final_answer = action_input.get("answer") or action_input.get("answer:") or ""
                    self._emit_trace(f"\n{'='*60}", verbose)
                    self._emit_trace(f"FINAL ANSWER: {final_answer}", verbose)
                    self._emit_trace(f"{'='*60}", verbose)
                    return final_answer
                
                # STEP 4: ACT - Execute the tool
                self._emit_trace(f"Calling tool: {action}", verbose)
                observation = self._execute_tool(action, action_input)
                
                self._emit_trace(f"\nObservation: {observation[:500]}...", verbose)
                
                # STEP 5: OBSERVE - Add result to conversation
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\nWhat should you do next?"
                })
                
            except ValueError as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}")
                self._emit_trace(f"Parsing error in iteration {iteration + 1}: {e}", verbose)
                # Try to recover by asking LLM to fix its response
                messages.append({
                    "role": "user",
                    "content": f"Error: {str(e)}. Please provide a valid JSON response."
                })
                continue
            
            except Exception as e:
                logger.error(f"Unexpected error in iteration {iteration + 1}: {e}", exc_info=True)
                self._emit_trace(f"Unexpected error in iteration {iteration + 1}: {e}", verbose)
                return f"Error: Agent encountered an unexpected error: {str(e)}"
        
        # Max iterations reached
        logger.warning(f"Agent reached max iterations ({max_iterations}) without final answer")
        return f"I apologize, but I couldn't complete the task within {max_iterations} steps. Please try rephrasing your question or breaking it into smaller parts."
    
    def _sanitize_input(self, text: str) -> str:
        """Remove potentially problematic characters from input
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text with control characters removed
        """
        # Remove null bytes and other control characters except newlines/tabs
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        return sanitized.strip()


# Made with Bob
