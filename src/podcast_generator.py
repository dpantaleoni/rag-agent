"""Podcast Script Generator for creating conversational podcasts from documentation"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from src.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class PodcastGenerator:
    """Generate conversational podcast scripts and audio from documentation
    
    Uses RAG to retrieve relevant context and Agent to generate
    natural dialogue between a HOST and an EXPERT.
    """
    
    def __init__(self, rag, agent, tts_engine: Optional[TTSEngine] = None):
        """Initialize podcast generator
        
        Args:
            rag: SimpleRAG instance for document retrieval
            agent: Agent instance for content generation
            tts_engine: Optional TTSEngine instance for audio generation
        """
        self.rag = rag
        self.agent = agent
        self.tts_engine = tts_engine
    
    def generate_script(
        self, 
        topic: str, 
        duration_min: int = 5
    ) -> List[Tuple[str, str]]:
        """Generate podcast dialogue script
        
        Args:
            topic: Topic to generate podcast about
            duration_min: Target duration in minutes (default: 5)
            
        Returns:
            List of (speaker, text) tuples representing the dialogue
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If script generation fails
        """
        # Validate inputs
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if duration_min < 1 or duration_min > 30:
            raise ValueError("Duration must be between 1 and 30 minutes")
        
        logger.info(f"Generating {duration_min}-minute podcast script about: {topic}")
        
        try:
            # Retrieve relevant context from documentation
            context_chunks = self.rag.retrieve(topic, k=5)
            
            if not context_chunks:
                raise ValueError(f"No relevant documentation found for topic: {topic}")
            
            context = "\n\n".join([c["content"] for c in context_chunks])
            
            # Build prompt for podcast generation
            prompt = self._build_podcast_prompt(topic, duration_min)
            
            # Generate script with retry logic (now expecting JSON)
            max_attempts = 3
            script_data = None
            script_lines: List[Tuple[str, str]] = []
            
            for attempt in range(max_attempts):
                try:
                    script_json = self.agent.generate(
                        prompt=prompt,
                        role="synthesis",
                        context=context,
                        temperature=0.7  # Higher temp for creative podcast generation
                    )
                    
                    # Log the raw response for debugging
                    logger.info(f"Raw agent response: {script_json[:500]}...")
                    
                    # Extract JSON from response
                    script_json = self._extract_json(script_json)
                    
                    # Parse JSON response
                    script_data = json.loads(script_json)
                    
                    # Convert JSON to script_lines format
                    if 'dialogue' not in script_data:
                        raise ValueError("JSON missing 'dialogue' key")
                    
                    script_lines = [
                        (item['speaker'], item['text'])
                        for item in script_data['dialogue']
                    ]
                    
                    # Validate we got some dialogue
                    if len(script_lines) < 2:
                        raise ValueError("Script too short - need at least 2 dialogue lines")
                    
                    break  # Success!
                    
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Script parse attempt {attempt + 1} failed, retrying...")
                        continue
                    else:
                        logger.error(f"Failed to generate valid script after {max_attempts} attempts: {e}")
                        raise Exception(f"Script generation failed: {e}")
            
            if script_lines is None or len(script_lines) == 0:
                raise Exception("Failed to generate valid podcast script")
            
            # Save to file
            output_path = self._save_script(script_lines, topic, duration_min)
            logger.info(f"Podcast script saved to: {output_path}")
            
            return script_lines
            
        except Exception as e:
            logger.error(f"Podcast generation failed: {e}", exc_info=True)
            raise
    
    def _build_podcast_prompt(self, topic: str, duration_min: int) -> str:
        """Build the prompt for podcast generation
        
        Args:
            topic: Podcast topic
            duration_min: Target duration in minutes
            
        Returns:
            Formatted prompt string
        """
        # Estimate word count (average speaking rate: 150 words/minute)
        target_words = duration_min * 150
        
        prompt = f"""CRITICAL INSTRUCTION: You MUST respond with ONLY valid JSON wrapped in <JSON> tags. No explanations, no markdown, no extra text before or after.

Create a {duration_min}-minute podcast dialogue about {topic}.

Your ENTIRE response must be wrapped in <JSON> tags and contain ONLY the JSON structure shown below.

<JSON>
{{
  "dialogue": [
    {{"speaker": "HOST", "text": "Dialogue text here..."}},
    {{"speaker": "EXPERT", "text": "Response text here..."}}
  ]
}}
</JSON>

EXAMPLE OF CORRECT OUTPUT:
<JSON>
{{
  "dialogue": [
    {{"speaker": "HOST", "text": "Welcome everyone to today's episode! We're diving into Docling, a powerful tool for document processing. Can you give us a quick overview of what Docling does?"}},
    {{"speaker": "EXPERT", "text": "Absolutely! Docling is an open-source library designed to parse and convert various document formats like PDFs, Word documents, and more into structured formats such as JSON or Markdown. It's particularly useful when you need to extract information from documents programmatically."}},
    {{"speaker": "HOST", "text": "That sounds really useful! What makes Docling different from other document parsing tools?"}},
    {{"speaker": "EXPERT", "text": "Great question! Docling stands out because it uses advanced AI models for layout analysis and table detection. This means it can understand the structure of complex documents, not just extract raw text."}}
  ]
}}
</JSON>

Dialogue Structure:
- HOST: Friendly introduction and opening question
- EXPERT: Clear, accessible explanation of the concept
- HOST: Follow-up question or request for clarification
- EXPERT: Detailed technical explanation with examples
- HOST: Question about practical applications or use cases
- EXPERT: Real-world examples and best practices
- HOST: Wrap-up question or summary
- EXPERT: Key takeaways and final thoughts

Guidelines:
- HOST should ask clarifying questions that a learner would ask
- EXPERT should provide technical but accessible explanations
- Use natural, conversational language
- Include specific examples from the documentation
- Maintain an educational yet friendly tone
- Aim for approximately {target_words} words total
- Build on previous points in the conversation
- Make it feel like a real dialogue, not a lecture

WRONG - DO NOT DO THIS:
"Here's a podcast script about {topic}..."

WRONG - DO NOT DO THIS:
"Based on the documentation, I'll create a dialogue..."

RIGHT - DO THIS:
<JSON>{{"dialogue": [...]}}</JSON>

Now generate the complete podcast dialogue. Output ONLY <JSON>...</JSON> with the dialogue data, nothing else."""
        
        return prompt
    
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
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        elif '```' in text:
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
    
    def _save_script(
        self, 
        script_lines: List[Tuple[str, str]], 
        topic: str,
        duration_min: int
    ) -> Path:
        """Save script to JSON file
        
        Args:
            script_lines: List of (speaker, text) tuples
            topic: Topic name for filename
            duration_min: Duration in minutes
            
        Returns:
            Path to saved file
        """
        # Create safe filename from topic
        safe_topic = topic.replace(' ', '_').replace('/', '_')
        safe_topic = ''.join(c for c in safe_topic if c.isalnum() or c == '_')
        
        # Prepare script data
        script_data = {
            "topic": topic,
            "duration_min": duration_min,
            "script": [
                {"speaker": speaker, "text": text}
                for speaker, text in script_lines
            ]
        }
        
        output_path = Path(f"outputs/podcasts/{safe_topic}_script.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def display_script(self, script_lines: List[Tuple[str, str]]):
        """Display script in a readable format
        
        Args:
            script_lines: List of (speaker, text) tuples to display
        """
        print(f"\n{'='*80}")
        print(f"PODCAST SCRIPT: {len(script_lines)} dialogue segments")
        print(f"{'='*80}\n")
        
        for speaker, text in script_lines:
            # Format speaker name
            speaker_label = f"🎙️  {speaker}" if speaker == "HOST" else f"👨‍🏫 {speaker}"
            
            print(f"{speaker_label}:")
            print(f"  {text}\n")
        
        print(f"{'='*80}\n")
    
    def estimate_duration(self, script_lines: List[Tuple[str, str]]) -> float:
        """Estimate podcast duration based on word count
        
        Args:
            script_lines: List of (speaker, text) tuples
            
        Returns:
            Estimated duration in minutes
        """
        total_words = sum(len(text.split()) for _, text in script_lines)
        # Average speaking rate: 150 words per minute
        duration_min = total_words / 150
        return duration_min
    
    def generate_audio(
        self,
        topic: str,
        duration_min: int = 5,
        output_path: Optional[str] = None
    ) -> Path:
        """Generate complete podcast audio from a topic.
        
        Args:
            topic: Topic to generate podcast about
            duration_min: Target duration in minutes
            output_path: Optional explicit output path for MP3
            
        Returns:
            Path to generated MP3 file
        """
        script_lines = self.generate_script(topic, duration_min)
        return self.create_audio_from_script(script_lines, topic, output_path)
    
    def create_audio_from_script(
        self,
        script_lines: List[Tuple[str, str]],
        topic: str,
        output_path: Optional[str] = None
    ) -> Path:
        """Create podcast audio from an existing script.
        
        Args:
            script_lines: List of (speaker, text) tuples
            topic: Topic name used for default filename generation
            output_path: Optional explicit output path for MP3
            
        Returns:
            Path to generated MP3 file
        """
        if not script_lines:
            raise ValueError("Script lines cannot be empty")
        
        tts_engine = self.tts_engine or TTSEngine()
        
        if output_path is None:
            safe_topic = topic.replace(' ', '_').replace('/', '_')
            safe_topic = ''.join(c for c in safe_topic if c.isalnum() or c == '_')
            output_path = f"outputs/podcasts/{safe_topic}_podcast.mp3"
        
        final_path = tts_engine.create_podcast(script_lines, output_path)
        logger.info(f"Podcast audio saved to: {final_path}")
        return final_path


# Made with Bob