"""
Whisper Service for Voice-to-Text Transcription

This service uses faster-whisper (CTranslate2) for transcription.
Runs locally with no API calls - completely free.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from ..config import get_settings
from ..models import TranscriptSegment

logger = logging.getLogger(__name__)


class WhisperService:
    """
    Service for transcribing audio files using faster-whisper.
    
    This service loads the Whisper model once and reuses it for all
    transcription requests. The model runs locally without any API costs.
    
    Attributes:
        model: The loaded Whisper model instance
        model_name: Name of the Whisper model being used
    """
    
    _instance: Optional["WhisperService"] = None
    _model: Optional[WhisperModel] = None
    
    def __new__(cls) -> "WhisperService":
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the Whisper service with the configured model."""
        if self._model is None:
            settings = get_settings()
            self.model_name = settings.whisper_model
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Use CPU for Docker compatibility, auto for GPU if available
            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
            )
            
            logger.info(f"Whisper model '{self.model_name}' loaded successfully")
    
    @property
    def model(self) -> WhisperModel:
        """Get the loaded Whisper model."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")
        return self._model
    
    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code (e.g., 'en', 'es'). 
                      If None, Whisper will auto-detect.
        
        Returns:
            Dictionary containing:
                - text: Full transcript text
                - segments: List of segments with timestamps
                - language: Detected or specified language
                - duration: Audio duration in seconds
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio file: {audio_path}")
        
        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
        )
        
        # Convert segments to list (it's a generator)
        segment_list = list(segments)
        
        # Build full text
        full_text = " ".join(seg.text.strip() for seg in segment_list)
        
        logger.info(
            f"Transcription complete. "
            f"Language: {info.language}, "
            f"Segments: {len(segment_list)}"
        )
        
        return {
            "text": full_text,
            "segments": segment_list,
            "language": info.language,
            "duration": info.duration,
        }
    
    def transcribe_to_segments(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
    ) -> tuple[list[TranscriptSegment], str, float]:
        """
        Transcribe audio and return structured segments.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code
        
        Returns:
            Tuple of (segments, detected_language, duration_seconds)
        """
        result = self.transcribe(audio_path, language)
        
        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    speaker="Speaker",  # Whisper doesn't do speaker diarization
                    timestamp=self._format_timestamp(seg.start),
                    text=seg.text.strip(),
                    start_seconds=seg.start,
                    end_seconds=seg.end,
                )
            )
        
        return segments, result.get("language", "en"), result.get("duration", 0.0)
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        file_extension: str = ".wav",
        language: Optional[str] = None,
    ) -> tuple[list[TranscriptSegment], str, float]:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Raw audio data
            file_extension: File extension for the temp file
            language: Optional language code
        
        Returns:
            Tuple of (segments, detected_language, duration_seconds)
        """
        # Write bytes to temporary file
        with tempfile.NamedTemporaryFile(
            suffix=file_extension, delete=False
        ) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            return self.transcribe_to_segments(tmp_path, language)
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def get_whisper_service() -> WhisperService:
    """
    Get the Whisper service instance.
    
    Returns:
        WhisperService: Singleton instance of the Whisper service
    """
    return WhisperService()
