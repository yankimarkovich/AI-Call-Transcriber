"""
Transcription Agent - orchestrates the transcription pipeline.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .tools import (
    convert_to_wav,
    load_diarization_pipeline,
    perform_diarization,
    transcribe_audio,
    map_speakers_to_transcript,
    create_download_file,
    cleanup_file,
)


@dataclass
class TranscriptionResult:
    """Result of the transcription process."""
    success: bool
    message: str
    transcript: str
    download_path: Optional[str]
    num_speakers: int


class TranscriptionAgent:
    """
    Agent that orchestrates the audio transcription pipeline.
    Handles speaker diarization and transcription with fallback support.
    """

    def __init__(self, groq_api_key: str, hf_token: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.hf_token = hf_token
        self.diarization_pipeline = None

    def _load_diarization(self) -> bool:
        """Load the diarization pipeline if not already loaded."""
        if self.diarization_pipeline is None:
            logger.info("Loading pyannote diarization pipeline...")
            self.diarization_pipeline = load_diarization_pipeline(self.hf_token)
            if self.diarization_pipeline:
                logger.info("Diarization pipeline loaded successfully")
            else:
                logger.warning("Failed to load diarization pipeline")
        return self.diarization_pipeline is not None

    def process(
        self,
        audio_path: str,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> TranscriptionResult:
        """
        Process an audio file through the full transcription pipeline.

        Args:
            audio_path: Path to the audio file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Steps:
        1. Convert audio to WAV
        2. Perform speaker diarization
        3. Transcribe with Whisper
        4. Combine results
        5. Create downloadable file
        """
        wav_path = None

        try:
            # Step 1: Convert audio
            wav_path = convert_to_wav(audio_path)

            # Step 2: Diarization
            diarization_segments = []
            detected_speakers = 0

            logger.info("Starting speaker diarization...")
            if num_speakers:
                logger.info(f"Using specified num_speakers={num_speakers}")
            if self._load_diarization():
                diarization_segments = perform_diarization(
                    self.diarization_pipeline,
                    wav_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                unique_speakers = set(seg["speaker"] for seg in diarization_segments)
                detected_speakers = len(unique_speakers)
                logger.info(f"Diarization complete: found {detected_speakers} speakers, {len(diarization_segments)} segments")
            else:
                logger.warning("Diarization skipped - pipeline not available")

            # Step 3: Transcription
            logger.info("Starting transcription with Groq Whisper large-v3...")
            transcript = transcribe_audio(self.groq_client, wav_path)
            num_words = len(getattr(transcript, 'words', []))
            logger.info(f"Transcription complete: {num_words} words with timestamps")

            # Step 4: Combine
            final_transcript = map_speakers_to_transcript(
                diarization_segments,
                transcript
            )

            # Step 5: Create download file
            download_path = create_download_file(final_transcript)

            # Cleanup
            cleanup_file(wav_path)

            return TranscriptionResult(
                success=True,
                message=self._create_success_message(audio_path, detected_speakers),
                transcript=final_transcript,
                download_path=download_path,
                num_speakers=detected_speakers
            )

        except Exception as e:
            # Log the actual error
            logger.error(f"Error in transcription pipeline: {str(e)}", exc_info=True)

            # Cleanup on error
            cleanup_file(wav_path)

            # Attempt fallback
            return self._fallback_transcription(audio_path, str(e))

    def _fallback_transcription(self, audio_path: str, original_error: str) -> TranscriptionResult:
        """
        Fallback: transcription without speaker diarization.
        """
        wav_path = None

        try:
            wav_path = convert_to_wav(audio_path)
            transcript = transcribe_audio(self.groq_client, wav_path)

            fallback_text = (
                f"[Speaker identification failed - showing plain transcript]\n\n"
                f"{transcript.text}"
            )
            download_path = create_download_file(fallback_text)

            cleanup_file(wav_path)

            return TranscriptionResult(
                success=True,
                message="Partial success: Transcription completed but speaker identification failed.",
                transcript=fallback_text,
                download_path=download_path,
                num_speakers=0
            )

        except Exception as fallback_error:
            cleanup_file(wav_path)

            return TranscriptionResult(
                success=False,
                message=f"Error: {original_error}\nFallback also failed: {fallback_error}",
                transcript="",
                download_path=None,
                num_speakers=0
            )

    def _create_success_message(self, audio_path: str, num_speakers: int) -> str:
        """Create a success status message."""
        filename = os.path.basename(audio_path)
        speakers_text = str(num_speakers) if num_speakers > 0 else "Unknown"

        return (
            f"Transcription completed successfully!\n"
            f"- Participants identified: {speakers_text}\n"
            f"- Audio processed: {filename}"
        )


def create_agent(groq_api_key: str, hf_token: str) -> TranscriptionAgent:
    """Factory function to create a TranscriptionAgent."""
    return TranscriptionAgent(groq_api_key, hf_token)
