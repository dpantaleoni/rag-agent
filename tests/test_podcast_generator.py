"""Tests for PodcastGenerator"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock
from src.podcast_generator import PodcastGenerator


class TestPodcastGenerator:
    """Test suite for PodcastGenerator class"""
    
    @pytest.fixture
    def mock_rag(self):
        """Create mock RAG instance"""
        rag = Mock()
        rag.retrieve.return_value = [
            {
                "content": "Docling is a document processing library.",
                "score": 0.95,
                "metadata": {"source": "README.md"}
            },
            {
                "content": "It supports PDF, DOCX, and other formats.",
                "score": 0.90,
                "metadata": {"source": "features.md"}
            }
        ]
        return rag
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock Agent instance"""
        agent = Mock()
        agent.generate.return_value = """HOST: Welcome to today's episode! Can you tell us about Docling?
EXPERT: Absolutely! Docling is a powerful document processing library that makes it easy to work with various document formats.
HOST: That sounds interesting! What formats does it support?
EXPERT: It supports PDF, DOCX, and many other common document formats. The library uses advanced AI models to extract content accurately.
HOST: How would someone get started with Docling?
EXPERT: Getting started is quite simple. You can install it via pip and start processing documents with just a few lines of code.
HOST: That's great! Any final tips for our listeners?
EXPERT: I'd recommend checking out the documentation and trying it with your own documents. The community is very helpful too!"""
        return agent
    
    @pytest.fixture
    def generator(self, mock_rag, mock_agent):
        """Create PodcastGenerator instance with mocks"""
        return PodcastGenerator(mock_rag, mock_agent)
    
    @pytest.fixture
    def mock_tts_engine(self, tmp_path):
        """Create mock TTS engine instance"""
        tts = Mock()
        tts.create_podcast.return_value = tmp_path / "outputs" / "podcasts" / "test_topic_podcast.mp3"
        return tts
    
    def test_initialization(self, mock_rag, mock_agent):
        """Test generator initialization"""
        gen = PodcastGenerator(mock_rag, mock_agent)
        assert gen.rag == mock_rag
        assert gen.agent == mock_agent
    
    def test_generate_script_success(self, generator, mock_rag, mock_agent):
        """Test successful script generation"""
        script = generator.generate_script("Docling basics", duration_min=5)
        
        # Verify RAG was called
        mock_rag.retrieve.assert_called_once_with("Docling basics", k=5)
        
        # Verify agent was called
        assert mock_agent.generate.called
        call_args = mock_agent.generate.call_args
        assert call_args[1]['role'] == 'synthesis'
        assert 'Docling basics' in call_args[1]['prompt']
        
        # Verify script structure
        assert isinstance(script, list)
        assert len(script) > 0
        
        # Verify each line is a tuple
        for speaker, text in script:
            assert speaker in ['HOST', 'EXPERT']
            assert isinstance(text, str)
            assert len(text) > 0
    
    def test_generate_script_empty_topic(self, generator):
        """Test that empty topic raises ValueError"""
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            generator.generate_script("")
    
    def test_generate_script_invalid_duration(self, generator):
        """Test that invalid duration raises ValueError"""
        with pytest.raises(ValueError, match="Duration must be between"):
            generator.generate_script("test", duration_min=0)
        
        with pytest.raises(ValueError, match="Duration must be between"):
            generator.generate_script("test", duration_min=31)
    
    def test_generate_script_no_context(self, generator, mock_rag):
        """Test that no context raises ValueError"""
        mock_rag.retrieve.return_value = []
        
        with pytest.raises(ValueError, match="No relevant documentation found"):
            generator.generate_script("nonexistent topic")
    
    def test_parse_script_basic(self, generator):
        """Test basic script parsing"""
        script_text = """HOST: Hello everyone!
EXPERT: Thanks for having me.
HOST: Let's talk about the topic.
EXPERT: Sure, I'd be happy to explain."""
        
        lines = generator._parse_script(script_text)
        
        assert len(lines) == 4
        assert lines[0] == ('HOST', 'Hello everyone!')
        assert lines[1] == ('EXPERT', 'Thanks for having me.')
        assert lines[2] == ('HOST', "Let's talk about the topic.")
        assert lines[3] == ('EXPERT', "Sure, I'd be happy to explain.")
    
    def test_parse_script_multiline(self, generator):
        """Test parsing script with multiline dialogue"""
        script_text = """HOST: Welcome to the show!
This is going to be great.
EXPERT: Thank you for the introduction.
I'm excited to be here today."""
        
        lines = generator._parse_script(script_text)
        
        assert len(lines) == 2
        assert lines[0] == ('HOST', 'Welcome to the show! This is going to be great.')
        assert lines[1] == ('EXPERT', "Thank you for the introduction. I'm excited to be here today.")
    
    def test_parse_script_empty_lines(self, generator):
        """Test parsing script with empty lines"""
        script_text = """HOST: First line.

EXPERT: Second line.

HOST: Third line."""
        
        lines = generator._parse_script(script_text)
        
        assert len(lines) == 3
        assert lines[0] == ('HOST', 'First line.')
        assert lines[1] == ('EXPERT', 'Second line.')
        assert lines[2] == ('HOST', 'Third line.')
    
    def test_build_podcast_prompt(self, generator):
        """Test prompt building"""
        prompt = generator._build_podcast_prompt("test topic", 5)
        
        assert "test topic" in prompt
        assert "5-minute" in prompt
        assert "HOST:" in prompt
        assert "EXPERT:" in prompt
        assert "750" in prompt  # 5 * 150 words
    
    def test_save_script(self, generator, tmp_path, monkeypatch):
        """Test script saving"""
        # Change output directory to tmp_path
        monkeypatch.chdir(tmp_path)
        
        script_lines = [
            ('HOST', 'Hello!'),
            ('EXPERT', 'Hi there!')
        ]
        
        output_path = generator._save_script(script_lines, "test topic", 5)
        
        # Verify file was created
        assert output_path.exists()
        assert output_path.name == "test_topic_script.json"
        
        # Verify content
        with open(output_path) as f:
            data = json.load(f)
        
        assert data['topic'] == "test topic"
        assert data['duration_min'] == 5
        assert len(data['script']) == 2
        assert data['script'][0]['speaker'] == 'HOST'
        assert data['script'][0]['text'] == 'Hello!'
    
    def test_save_script_special_characters(self, generator, tmp_path, monkeypatch):
        """Test script saving with special characters in topic"""
        monkeypatch.chdir(tmp_path)
        
        script_lines = [('HOST', 'Test')]
        
        # Topic with special characters
        output_path = generator._save_script(
            script_lines, 
            "test/topic with spaces!", 
            5
        )
        
        # Verify filename is sanitized
        assert output_path.name == "test_topic_with_spaces_script.json"
    
    def test_estimate_duration(self, generator):
        """Test duration estimation"""
        script_lines = [
            ('HOST', 'This is a test with ten words in it here.'),
            ('EXPERT', 'Another line with exactly ten words in this sentence.')
        ]
        
        # 20 words total / 150 words per minute = 0.133 minutes
        duration = generator.estimate_duration(script_lines)
        assert 0.1 < duration < 0.2
    
    def test_display_script(self, generator, capsys):
        """Test script display"""
        script_lines = [
            ('HOST', 'Hello!'),
            ('EXPERT', 'Hi there!')
        ]
        
        generator.display_script(script_lines)
        
        captured = capsys.readouterr()
        assert 'HOST' in captured.out
        assert 'EXPERT' in captured.out
        assert 'Hello!' in captured.out
        assert 'Hi there!' in captured.out
    
    def test_generate_script_retry_on_parse_failure(self, generator, mock_agent):
        """Test retry logic when parsing fails initially"""
        # First call returns invalid format, second call returns valid
        mock_agent.generate.side_effect = [
            "Invalid format without proper prefixes",
            """HOST: Valid format now.
EXPERT: This should work."""
        ]
        
        script = generator.generate_script("test topic", duration_min=5)
        
        # Should have retried and succeeded
        assert mock_agent.generate.call_count == 2
        assert len(script) == 2
    
    def test_generate_audio_success(self, mock_rag, mock_agent, mock_tts_engine, tmp_path):
        """Test generating podcast audio from topic"""
        generator = PodcastGenerator(mock_rag, mock_agent, tts_engine=mock_tts_engine)
        
        output_path = generator.generate_audio("test topic", duration_min=5)
        
        assert output_path == tmp_path / "outputs" / "podcasts" / "test_topic_podcast.mp3"
        mock_tts_engine.create_podcast.assert_called_once()
        call_args = mock_tts_engine.create_podcast.call_args[0]
        assert len(call_args[0]) > 0
        assert call_args[1].endswith("test_topic_podcast.mp3")
    
    def test_create_audio_from_script_success(self, mock_rag, mock_agent, mock_tts_engine, tmp_path):
        """Test creating audio from existing script lines"""
        generator = PodcastGenerator(mock_rag, mock_agent, tts_engine=mock_tts_engine)
        script_lines = [
            ("HOST", "Hello there."),
            ("EXPERT", "Welcome to the show.")
        ]
        
        output_path = generator.create_audio_from_script(script_lines, "test topic")
        
        assert output_path == tmp_path / "outputs" / "podcasts" / "test_topic_podcast.mp3"
        mock_tts_engine.create_podcast.assert_called_once_with(
            script_lines,
            "outputs/podcasts/test_topic_podcast.mp3"
        )
    
    def test_create_audio_from_script_custom_output_path(self, mock_rag, mock_agent, mock_tts_engine, tmp_path):
        """Test creating audio with explicit output path"""
        custom_output = tmp_path / "custom.mp3"
        mock_tts_engine.create_podcast.return_value = custom_output
        generator = PodcastGenerator(mock_rag, mock_agent, tts_engine=mock_tts_engine)
        
        script_lines = [("HOST", "Hello there.")]
        output_path = generator.create_audio_from_script(
            script_lines,
            "test topic",
            output_path=str(custom_output)
        )
        
        assert output_path == custom_output
        mock_tts_engine.create_podcast.assert_called_once_with(
            script_lines,
            str(custom_output)
        )
    
    def test_create_audio_from_script_empty_lines(self, mock_rag, mock_agent):
        """Test creating audio with empty script lines"""
        generator = PodcastGenerator(mock_rag, mock_agent, tts_engine=Mock())
        
        with pytest.raises(ValueError, match="Script lines cannot be empty"):
            generator.create_audio_from_script([], "test topic")


# Made with Bob