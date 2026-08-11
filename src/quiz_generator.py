"""Quiz Generator for creating educational quizzes from documentation"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class QuizGenerator:
    """Generate multiple-choice quizzes from documentation
    
    Uses RAG to retrieve relevant context and Agent to generate
    structured quiz questions with explanations.
    """
    
    def __init__(self, rag, agent):
        """Initialize quiz generator
        
        Args:
            rag: SimpleRAG instance for document retrieval
            agent: Agent instance for content generation
        """
        self.rag = rag
        self.agent = agent
    
    def generate(
        self, 
        topic: str, 
        num_questions: int = 5, 
        difficulty: str = "medium"
    ) -> Dict[str, Any]:
        """Generate quiz on given topic
        
        Args:
            topic: Topic to generate quiz about
            num_questions: Number of questions to generate (default: 5)
            difficulty: Difficulty level - 'easy', 'medium', or 'hard' (default: 'medium')
            
        Returns:
            Dictionary containing quiz data with questions, options, answers, and explanations
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If quiz generation fails
        """
        # Validate inputs
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if num_questions < 1 or num_questions > 20:
            raise ValueError("Number of questions must be between 1 and 20")
        
        if difficulty not in ['easy', 'medium', 'hard']:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")
        
        logger.info(f"Generating {num_questions} {difficulty} questions about: {topic}")
        
        try:
            # Retrieve relevant context from documentation
            context_chunks = self.rag.retrieve(topic, k=5)
            
            if not context_chunks:
                raise ValueError(f"No relevant documentation found for topic: {topic}")
            
            context = "\n\n".join([c["content"] for c in context_chunks])
            
            # Build prompt for quiz generation
            prompt = self._build_quiz_prompt(topic, num_questions, difficulty)
            
            # Generate quiz with JSON schema (with retry logic for JSON parsing)
            max_attempts = 3
            quiz = None
            
            for attempt in range(max_attempts):
                try:
                    quiz_json = self.agent.generate(
                        prompt=prompt,
                        role="synthesis",
                        context=context,
                        temperature=0.4  # Balanced temp for factual accuracy with question variety
                    )

                    # Log the raw response for debugging
                    logger.info(f"Raw agent response: {quiz_json[:500]}...")

                    
                    # Try to extract JSON if wrapped in markdown or text
                    quiz_json = self._extract_json(quiz_json)
                    
                    # Parse JSON response
                    quiz = json.loads(quiz_json)
                    break  # Success!
                    
                except json.JSONDecodeError as e:
                    if attempt < max_attempts - 1:
                        logger.warning(f"JSON parse attempt {attempt + 1} failed, retrying...")
                        continue
                    else:
                        logger.error(f"Failed to parse quiz JSON after {max_attempts} attempts: {e}")
                        raise Exception(f"Quiz generation produced invalid JSON: {e}")
            
            if quiz is None:
                raise Exception("Failed to generate valid quiz JSON")
            
            # Validate quiz structure
            self._validate_quiz(quiz, num_questions)
            
            # Save to file
            output_path = self._save_quiz(quiz, topic)
            logger.info(f"Quiz saved to: {output_path}")
            
            return quiz
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse quiz JSON: {e}")
            raise Exception(f"Quiz generation produced invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}", exc_info=True)
            raise
    
    def _build_quiz_prompt(self, topic: str, num_questions: int, difficulty: str) -> str:
        """Build the prompt for quiz generation
        
        Args:
            topic: Quiz topic
            num_questions: Number of questions
            difficulty: Difficulty level
            
        Returns:
            Formatted prompt string
        """
        difficulty_guidelines = {
            'easy': 'Focus on basic concepts and definitions. Questions should test recall and understanding.',
            'medium': 'Include application and analysis questions. Test understanding and practical knowledge.',
            'hard': 'Include complex scenarios and edge cases. Test deep understanding and problem-solving.'
        }
        
        prompt = f"""CRITICAL INSTRUCTION: You MUST respond with ONLY valid JSON wrapped in <JSON> tags. No explanations, no markdown, no extra text before or after.

Create {num_questions} multiple-choice questions about {topic}.

Difficulty: {difficulty}
Guidelines: {difficulty_guidelines[difficulty]}

Requirements:
- Each question must have exactly 4 options (A, B, C, D)
- Only one option should be correct
- Provide a clear explanation for why the correct answer is right
- Base questions on the provided context
- Make distractors (wrong answers) plausible but clearly incorrect
- Use clear, unambiguous language

Your ENTIRE response must be wrapped in <JSON> tags and contain ONLY the JSON structure shown below.

<JSON>
{{{{
  "quiz": [
    {{{{
      "question": "Question text?",
      "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
      "correct_answer": "A",
      "explanation": "Detailed explanation of why this is correct and why others are wrong..."
    }}}}
  ]
}}}}
</JSON>

EXAMPLE OF CORRECT OUTPUT:
<JSON>
{{{{
  "quiz": [
    {{{{
      "question": "What is Docling's primary purpose?",
      "options": [
        "A) Parse and convert documents into structured formats",
        "B) Create new documents from scratch",
        "C) Compress PDF files for storage",
        "D) Encrypt sensitive documents"
      ],
      "correct_answer": "A",
      "explanation": "Docling is designed to parse various document formats (PDF, DOCX, etc.) and convert them into structured formats like JSON or Markdown. Options B, C, and D describe different document processing tasks that are not Docling's primary purpose."
    }}}}
  ]
}}}}
</JSON>

WRONG - DO NOT DO THIS:
"Based on the documentation, here are the questions: {{{{...}}}}"

WRONG - DO NOT DO THIS:
"Let me create a quiz about {topic}..."

RIGHT - DO THIS:
<JSON>{{{{...}}}}</JSON>

Now generate exactly {num_questions} questions. Output ONLY <JSON>...</JSON> with the quiz data, nothing else."""
        
        return prompt
    
    def _validate_quiz(self, quiz: Dict[str, Any], expected_questions: int):
        """Validate quiz structure and content
        
        Args:
            quiz: Quiz dictionary to validate
            expected_questions: Expected number of questions
            
        Raises:
            ValueError: If quiz structure is invalid
        """
        if 'quiz' not in quiz:
            raise ValueError("Quiz JSON missing 'quiz' key")
        
        questions = quiz['quiz']
        
        if not isinstance(questions, list):
            raise ValueError("Quiz 'quiz' field must be a list")
        
        if len(questions) != expected_questions:
            logger.warning(
                f"Expected {expected_questions} questions but got {len(questions)}"
            )
        
        # Validate each question
        for i, q in enumerate(questions, 1):
            required_fields = ['question', 'options', 'correct_answer', 'explanation']
            
            for field in required_fields:
                if field not in q:
                    raise ValueError(f"Question {i} missing required field: {field}")
            
            # Validate options
            if not isinstance(q['options'], list) or len(q['options']) != 4:
                raise ValueError(f"Question {i} must have exactly 4 options")
            
            # Validate correct_answer
            if q['correct_answer'] not in ['A', 'B', 'C', 'D']:
                raise ValueError(
                    f"Question {i} correct_answer must be A, B, C, or D"
                )
    
    def _save_quiz(self, quiz: Dict[str, Any], topic: str) -> Path:
        """Save quiz to JSON file
        
        Args:
            quiz: Quiz dictionary to save
            topic: Topic name for filename
            
        Returns:
            Path to saved file
        """
        # Create safe filename from topic
        safe_topic = topic.replace(' ', '_').replace('/', '_')
        safe_topic = ''.join(c for c in safe_topic if c.isalnum() or c == '_')
        
        output_path = Path(f"outputs/quizzes/{safe_topic}_quiz.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(quiz, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def display_quiz(self, quiz: Dict[str, Any], show_answers: bool = False):
        """Display quiz in a readable format
        
        Args:
            quiz: Quiz dictionary to display
            show_answers: Whether to show correct answers and explanations
        """
        questions = quiz.get('quiz', [])
        
        print(f"\n{'='*80}")
        print(f"QUIZ: {len(questions)} Questions")
        print(f"{'='*80}\n")
        
        for i, q in enumerate(questions, 1):
            print(f"Question {i}: {q['question']}\n")
            
            for option in q['options']:
                print(f"  {option}")
            
            if show_answers:
                print(f"\n  ✓ Correct Answer: {q['correct_answer']}")
                print(f"  Explanation: {q['explanation']}")
            
            print(f"\n{'-'*80}\n")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown or other formatting
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string
        """
        # Strategy 1: Extract from <JSON> tags (highest priority)
        if '<JSON>' in text and '</JSON>' in text:
            start = text.find('<JSON>') + 6
            end = text.find('</JSON>')
            if end > start:
                text = text[start:end].strip()
        
        # Strategy 2: Remove markdown code blocks if present
        elif '```json' in text:
            # Extract content between ```json and ```
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        elif '```' in text:
            # Extract content between ``` and ```
            start = text.find('```') + 3
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        
        # Strategy 3: Try to find JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]
        
        return text.strip()


# Made with Bob