"""Tests for quiz generator"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from src.quiz_generator import QuizGenerator


class TestQuizGenerator:
    """Test suite for QuizGenerator"""
    
    def test_init(self):
        """Test QuizGenerator initialization"""
        mock_rag = Mock()
        mock_agent = Mock()
        
        quiz_gen = QuizGenerator(mock_rag, mock_agent)
        
        assert quiz_gen.rag == mock_rag
        assert quiz_gen.agent == mock_agent
    
    def test_generate_valid_quiz(self, tmp_path):
        """Test successful quiz generation"""
        # Mock RAG to return context
        mock_rag = Mock()
        mock_rag.retrieve.return_value = [
            {"content": "Tables are used to organize data in rows and columns."},
            {"content": "Document tables can be extracted using Docling."}
        ]
        
        # Mock Agent to return valid JSON
        mock_agent = Mock()
        valid_quiz_json = json.dumps({
            "quiz": [
                {
                    "question": "What is a table?",
                    "options": ["A) A data structure", "B) A chair", "C) A file", "D) A folder"],
                    "correct_answer": "A",
                    "explanation": "Tables organize data in rows and columns."
                }
            ]
        })
        mock_agent.generate.return_value = valid_quiz_json
        
        # Create quiz generator
        quiz_gen = QuizGenerator(mock_rag, mock_agent)
        
        # Override output path to use tmp_path
        original_save = quiz_gen._save_quiz
        def mock_save(quiz, topic):
            output_path = tmp_path / f"{topic}_quiz.json"
            with open(output_path, 'w') as f:
                json.dump(quiz, f, indent=2)
            return output_path
        quiz_gen._save_quiz = mock_save
        
        # Generate quiz
        quiz = quiz_gen.generate("tables", num_questions=1, difficulty="medium")
        
        # Verify structure
        assert "quiz" in quiz
        assert len(quiz["quiz"]) == 1
        assert quiz["quiz"][0]["question"] == "What is a table?"
        assert len(quiz["quiz"][0]["options"]) == 4
        assert quiz["quiz"][0]["correct_answer"] == "A"
    
    def test_generate_invalid_topic(self):
        """Test quiz generation with invalid topic"""
        mock_rag = Mock()
        mock_agent = Mock()
        quiz_gen = QuizGenerator(mock_rag, mock_agent)
        
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            quiz_gen.generate("", num_questions=5)
    
    def test_generate_invalid_num_questions(self):
        """Test quiz generation with invalid number of questions"""
        mock_rag = Mock()
        mock_agent = Mock()
        quiz_gen = QuizGenerator(mock_rag, mock_agent)
        
        with pytest.raises(ValueError, match="Number of questions must be between"):
            quiz_gen.generate("tables", num_questions=0)
        
        with pytest.raises(ValueError, match="Number of questions must be between"):
            quiz_gen.generate("tables", num_questions=25)
    
    def test_generate_invalid_difficulty(self):
        """Test quiz generation with invalid difficulty"""
        mock_rag = Mock()
        mock_agent = Mock()
        quiz_gen = QuizGenerator(mock_rag, mock_agent)
        
        with pytest.raises(ValueError, match="Difficulty must be"):
            quiz_gen.generate("tables", difficulty="impossible")
    
    def test_extract_json_from_markdown(self):
        """Test JSON extraction from markdown code blocks"""
        mock_rag = Mock()
        mock_agent = Mock()
        quiz_gen = QuizGenerator(mock_rag, mock_agent)
        
        # Test with ```json``` blocks
        text_with_json = '```json\n{"quiz": []}\n```'
        extracted = quiz_gen._extract_json(text_with_json)
        assert extracted == '{"quiz": []}'
        
        # Test with ``` blocks
        text_with_code = '```\n{"quiz": []}\n```'
        extracted = quiz_gen._extract_json(text_with_code)
        assert extracted == '{"quiz": []}'
        
        # Test with plain JSON
        plain_json = '{"quiz": []}'
        extracted = quiz_gen._extract_json(plain_json)
        assert extracted == '{"quiz": []}'
    
    def test_validate_quiz_structure(self):
        """Test quiz validation"""
        mock_rag = Mock()
        mock_agent = Mock()
        quiz_gen = QuizGenerator(mock_rag, mock_agent)
        
        # Valid quiz
        valid_quiz = {
            "quiz": [
                {
                    "question": "Test?",
                    "options": ["A) 1", "B) 2", "C) 3", "D) 4"],
                    "correct_answer": "A",
                    "explanation": "Because."
                }
            ]
        }
        quiz_gen._validate_quiz(valid_quiz, 1)  # Should not raise
        
        # Missing 'quiz' key
        with pytest.raises(ValueError, match="missing 'quiz' key"):
            quiz_gen._validate_quiz({}, 1)
        
        # Missing required field
        invalid_quiz = {
            "quiz": [
                {
                    "question": "Test?",
                    "options": ["A) 1", "B) 2", "C) 3", "D) 4"]
                    # Missing correct_answer and explanation
                }
            ]
        }
        with pytest.raises(ValueError, match="missing required field"):
            quiz_gen._validate_quiz(invalid_quiz, 1)
        
        # Wrong number of options
        invalid_options = {
            "quiz": [
                {
                    "question": "Test?",
                    "options": ["A) 1", "B) 2"],  # Only 2 options
                    "correct_answer": "A",
                    "explanation": "Because."
                }
            ]
        }
        with pytest.raises(ValueError, match="must have exactly 4 options"):
            quiz_gen._validate_quiz(invalid_options, 1)
        
        # Invalid correct_answer
        invalid_answer = {
            "quiz": [
                {
                    "question": "Test?",
                    "options": ["A) 1", "B) 2", "C) 3", "D) 4"],
                    "correct_answer": "E",  # Invalid
                    "explanation": "Because."
                }
            ]
        }
        with pytest.raises(ValueError, match="must be A, B, C, or D"):
            quiz_gen._validate_quiz(invalid_answer, 1)
    
    def test_build_quiz_prompt(self):
        """Test quiz prompt building"""
        mock_rag = Mock()
        mock_agent = Mock()
        quiz_gen = QuizGenerator(mock_rag, mock_agent)
        
        prompt = quiz_gen._build_quiz_prompt("tables", 5, "medium")
        
        assert "5" in prompt
        assert "tables" in prompt
        assert "medium" in prompt
        assert "JSON" in prompt
        assert "exactly 4 options" in prompt


# Made with Bob