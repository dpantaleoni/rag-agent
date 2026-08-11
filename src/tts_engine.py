"""Text-to-speech engine using Piper voices for podcast generation."""

import io
import logging
import subprocess
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from piper import PiperVoice

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-speech with Piper TTS."""

    def __init__(self, voices_dir: str = "voices"):
        """Initialize Piper TTS with configured voice models.

        Args:
            voices_dir: Directory containing Piper .onnx voice files
        """
        self.voices_dir = Path(voices_dir)

        # Configure voices for HOST and EXPERT
        self.host_voice_path = self.voices_dir / "en_US-amy-medium.onnx"
        self.expert_voice_path = self.voices_dir / "en_GB-cori-high.onnx"

        self._validate_voice_files()

        # Load voice models eagerly so failures happen early
        self.host_voice = PiperVoice.load(str(self.host_voice_path))
        self.expert_voice = PiperVoice.load(str(self.expert_voice_path))


    def _validate_voice_files(self):
        """Validate required voice model files exist."""
        missing_files = []

        if not self.host_voice_path.exists():
            missing_files.append(str(self.host_voice_path))

        if not self.expert_voice_path.exists():
            missing_files.append(str(self.expert_voice_path))

        if missing_files:
            raise FileNotFoundError(
                "Missing Piper voice model files: " + ", ".join(missing_files)
            )

    def synthesize(self, text: str, voice: str = "host") -> Tuple[np.ndarray, int]:
        """Convert text to audio.

        Args:
            text: Text to synthesize
            voice: Voice selection, either "host" or "expert"

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if voice not in {"host", "expert"}:
            raise ValueError("Voice must be 'host' or 'expert'")

        voice_model = self.host_voice if voice == "host" else self.expert_voice

        
        chunks = list(voice_model.synthesize(text))
      

        first_chunk = chunks[0]
        sample_rate = first_chunk.sample_rate
        
        # Collect all audio data
        audio_data = []
        for chunk in chunks:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            audio_data.append(audio_array)
        
        # Concatenate all chunks
        full_audio = np.concatenate(audio_data)
        
        # Convert to float32 in range [-1.0, 1.0] for soundfile
        audio_float = full_audio.astype(np.float32) / 32768.0
        
        return audio_float, sample_rate

    def create_podcast(
        self,
        script_lines: List[Tuple[str, str]],
        output_path: str,
        pause_ms: int = 300
    ) -> Path:
        """Create complete podcast audio from script lines.

        Args:
            script_lines: List of (speaker, text) tuples
            output_path: Path to save MP3 file
            pause_ms: Silence duration between segments in milliseconds

        Returns:
            Path to exported MP3 file
        """
        if not script_lines:
            raise ValueError("Script lines cannot be empty")

        if pause_ms < 0:
            raise ValueError("pause_ms cannot be negative")

        segments = []
        sample_rate = None

        for speaker, text in script_lines:
            normalized_speaker = speaker.upper()
            if normalized_speaker not in {"HOST", "EXPERT"}:
                raise ValueError(f"Unsupported speaker: {speaker}")

            voice = "host" if normalized_speaker == "HOST" else "expert"
            audio_data, sr = self.synthesize(text, voice=voice)
            
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise ValueError("Sample rate mismatch between voice models")
            
            segments.append(audio_data)
            
            # Add silence between segments
            silence_samples = int((pause_ms / 1000.0) * sample_rate)
            silence = np.zeros(silence_samples, dtype=np.float32)
            segments.append(silence)

        # Concatenate all segments
        final_audio = np.concatenate(segments)

        # Ensure output directory exists
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to MP3 using soundfile
        # Note: soundfile writes to various formats, but for MP3 we need to use WAV first
        # then convert, or use a different approach
        
        # For now, let's write as WAV (lossless, widely supported)
        # If MP3 is required, we can add ffmpeg conversion
        wav_path = output.with_suffix('.wav')
        sf.write(str(wav_path), final_audio, sample_rate)
        
        # If MP3 was requested, try to convert using ffmpeg
        if output.suffix.lower() == '.mp3':
            try:
                subprocess.run(
                    ['ffmpeg', '-i', str(wav_path), '-codec:a', 'libmp3lame',
                     '-b:a', '128k', '-y', str(output)],
                    check=True,
                    capture_output=True
                )
                # Remove temporary WAV file
                wav_path.unlink()
                logger.info("Podcast audio exported to: %s", output)
                return output
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If ffmpeg fails or isn't available, keep the WAV file
                logger.warning("ffmpeg not available, saved as WAV instead of MP3")
                logger.info("Podcast audio exported to: %s", wav_path)
                return wav_path
        else:
            logger.info("Podcast audio exported to: %s", wav_path)
            return wav_path


# Made with Bob