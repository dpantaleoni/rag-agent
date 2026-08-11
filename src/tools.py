"""Tool definitions for agent systems"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Tool definition for agent use
    
    Attributes:
        name: Unique identifier for the tool
        func: Callable function to execute
        description: Clear description of what the tool does and when to use it
        parameters: Optional dict describing expected parameters
    """
    name: str
    func: Callable
    description: str
    parameters: Optional[Dict[str, Any]] = None
    
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters
        
        Args:
            **kwargs: Parameters to pass to the tool function
            
        Returns:
            String result from tool execution
        """
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}", exc_info=True)
            return f"Error: {str(e)}"


def create_rag_tools(rag_instance) -> List[Tool]:
    """Create RAG-related tools for document retrieval
    
    Args:
        rag_instance: Instance of SimpleRAG class
        
    Returns:
        List of Tool objects for RAG operations
    """
    def search_documents(query: str, k: int = 5) -> str:
        """Search for relevant documents using semantic search
        
        Args:
            query: Search query
            k: Number of results to return (default: 5)
            
        Returns:
            Formatted string with search results
        """
        try:
            results = rag_instance.retrieve(query, k=k)
            
            if not results:
                return "No relevant documents found."
            
            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"[Result {i}] (Score: {result['score']:.3f})\n"
                    f"Source: {result['metadata'].get('source', 'Unknown')}\n"
                    f"Content: {result['content'][:500]}...\n"
                )
            
            return "\n".join(formatted)
        except Exception as e:
            return f"Search error: {str(e)}"
    
    return [
        Tool(
            name="search_documents",
            func=search_documents,
            description=(
                "Search the document database for relevant information using semantic search. "
                "Use this when you need to find specific information from the knowledge base. "
                "Input should be a clear search query describing what you're looking for."
            ),
            parameters={
                "query": "string - the search query",
                "k": "integer - number of results (default: 5)"
            }
        )
    ]


def create_calculator_tool() -> Tool:
    """Create a simple calculator tool for mathematical operations
    
    Returns:
        Tool object for calculations
    """
    def calculate(expression: str) -> str:
        """Safely evaluate mathematical expressions
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation
        """
        try:
            # Only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/()., ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    return Tool(
        name="calculator",
        func=calculate,
        description=(
            "Perform mathematical calculations. "
            "Use this when you need to compute numerical values. "
            "Input should be a valid mathematical expression (e.g., '2 + 2', '10 * 5')."
        ),
        parameters={
            "expression": "string - mathematical expression to evaluate"
        }
    )


def create_final_answer_tool() -> Tool:
    """Create a tool for providing final answers
    
    Returns:
        Tool object for final answers
    """
    def final_answer(answer: str) -> str:
        """Provide the final answer to the user's query
        
        Args:
            answer: The final answer to return
            
        Returns:
            The answer string
        """
        return answer
    
    return Tool(
        name="final_answer",
        func=final_answer,
        description=(
            "Use this to provide your final answer to the user's question. "
            "Only use this when you have gathered all necessary information "
            "and are ready to give a complete response."
        ),
        parameters={
            "answer": "string - the final answer to provide"
        }
    )


def create_quiz_tool(rag_instance, agent) -> Tool:
    """Create a tool for generating quizzes from documentation
    
    Args:
        rag_instance: Instance of SimpleRAG class
        agent: Agent instance for content generation
        
    Returns:
        Tool object for quiz generation
    """
    def generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "medium") -> str:
        """Generate a quiz on a given topic using RAG context
        
        Args:
            topic: Topic to generate quiz about
            num_questions: Number of questions (1-20, default: 5)
            difficulty: Difficulty level - 'easy', 'medium', or 'hard' (default: 'medium')
            
        Returns:
            JSON string containing the quiz data
        """
        try:
            from .quiz_generator import QuizGenerator
            
            # Validate inputs
            if not topic or not topic.strip():
                return "Error: Topic cannot be empty"
            
            if num_questions < 1 or num_questions > 20:
                return "Error: Number of questions must be between 1 and 20"
            
            if difficulty not in ['easy', 'medium', 'hard']:
                return "Error: Difficulty must be 'easy', 'medium', or 'hard'"
            
            # Use QuizGenerator to create the quiz
            generator = QuizGenerator(rag_instance, agent)
            quiz = generator.generate(topic, num_questions, difficulty)
            
            return f"Quiz generated successfully! {len(quiz.get('quiz', []))} questions created about '{topic}'."
            
        except Exception as e:
            logger.error(f"Error generating quiz: {e}", exc_info=True)
            return f"Error generating quiz: {str(e)}"
    
    return Tool(
        name="generate_quiz",
        func=generate_quiz,
        description=(
            "Generate an educational quiz on a specific topic from the documentation. "
            "Creates multiple-choice questions with explanations. "
            "Use this when the user asks to create a quiz or test their knowledge."
        ),
        parameters={
            "topic": "string - the topic to generate quiz about",
            "num_questions": "integer - number of questions (1-20, default: 5)",
            "difficulty": "string - 'easy', 'medium', or 'hard' (default: 'medium')"
        }
    )


def create_podcast_tool(rag_instance, agent) -> Tool:
    """Create a tool for generating podcast scripts from documentation
    
    Args:
        rag_instance: Instance of SimpleRAG class
        agent: Agent instance for content generation
        
    Returns:
        Tool object for podcast generation
    """
    def generate_podcast(topic: str, duration_min: int = 3, generate_audio: bool = True) -> str:
        """Generate a conversational podcast script on a given topic
        
        Args:
            topic: Topic to generate podcast about
            duration_min: Target duration in minutes (1-30, default: 3)
            generate_audio: Whether to also generate audio file (default: True)
            
        Returns:
            Formatted string with the podcast script or audio path
        """
        try:
            from .podcast_generator import PodcastGenerator
            from .tts_engine import TTSEngine
            
            # Validate inputs
            if not topic or not topic.strip():
                return "Error: Topic cannot be empty"
            
            if duration_min < 1 or duration_min > 30:
                return "Error: Duration must be between 1 and 30 minutes"
            
            # Initialize with TTS engine if audio generation requested
            tts_engine = TTSEngine() if generate_audio else None
            generator = PodcastGenerator(rag_instance, agent, tts_engine=tts_engine)
            
            if generate_audio:
                # Generate both script and audio
                audio_path = generator.generate_audio(topic, duration_min)
                return f"Podcast generated successfully! Audio saved to: {audio_path}"
            else:
                # Generate script only
                script_lines = generator.generate_script(topic, duration_min)
                total_words = sum(len(text.split()) for _, text in script_lines)
                estimated_duration = total_words / 150
                return f"Podcast script generated successfully! {len(script_lines)} dialogue segments created (~{estimated_duration:.1f} min) about '{topic}'."
            
        except Exception as e:
            logger.error(f"Error generating podcast: {e}", exc_info=True)
            return f"Error generating podcast: {str(e)}"
    
    return Tool(
        name="generate_podcast",
        func=generate_podcast,
        description=(
            "Generate a conversational podcast script about a topic from the documentation. "
            "Creates dialogue between a HOST and an EXPERT. "
            "Can optionally generate audio file using TTS voices. "
            "Use this when the user asks to create a podcast or audio content."
        ),
        parameters={
            "topic": "string - the topic to generate podcast about",
            "duration_min": "integer - target duration in minutes (1-30, default: 3)",
            "generate_audio": "boolean - whether to generate audio file (default: True)"
        }
    )


# Made with Bob