"""
Audio processing tools and utilities.
"""

import os
import tempfile
from typing import Optional

from pydub import AudioSegment
from pyannote.audio import Pipeline
from openai import OpenAI
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_to_wav(audio_path: str) -> str:
    """Convert audio file to WAV format for processing."""
    try:
        audio = AudioSegment.from_file(audio_path)
        wav_path = tempfile.mktemp(suffix=".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        raise Exception(f"Failed to convert audio file: {e}")


def load_diarization_pipeline(hf_token: str) -> Optional[Pipeline]:
    """Load the pyannote speaker diarization pipeline."""
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        pipeline.to(DEVICE)
        return pipeline
    except Exception as e:
        print(f"Error loading diarization pipeline: {e}")
        return None


def perform_diarization(
    pipeline: Pipeline,
    audio_path: str,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None
) -> list:
    """
    Perform speaker diarization on the audio file.
    Returns a list of segment dictionaries with start, end, and speaker.

    Args:
        pipeline: The pyannote diarization pipeline
        audio_path: Path to the audio file
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
    """
    try:
        # Build kwargs for pipeline
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        diarization = pipeline(audio_path, **kwargs)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments
    except Exception as e:
        print(f"Diarization error: {e}")
        return []


def transcribe_audio(client: OpenAI, audio_path: str, language: str = "he") -> dict:
    """Transcribe audio using OpenAI Whisper API with word-level timestamps."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )
        return transcript
    except Exception as e:
        raise Exception(f"Transcription failed: {e}")


def _find_speaker_for_time(timestamp: float, diarization_segments: list) -> str:
    """Find the speaker active at a given timestamp."""
    # First, check if timestamp falls within any segment
    for seg in diarization_segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return seg["speaker"]

    # If not in any segment, find the closest segment
    closest_seg = min(
        diarization_segments,
        key=lambda x: min(abs(x["start"] - timestamp), abs(x["end"] - timestamp))
    )
    return closest_seg["speaker"]


def map_speakers_to_transcript(diarization_segments: list, transcript) -> str:
    """
    Map speaker labels from diarization to transcribed text.
    Uses diarization segments to attribute text to speakers.
    Creates organized output with participant labels.
    """
    # Get transcript text
    if hasattr(transcript, 'text'):
        full_text = transcript.text
    elif isinstance(transcript, str):
        full_text = transcript
    else:
        full_text = str(transcript)

    if not diarization_segments:
        return f"[Speaker identification unavailable]\n\n{full_text}"

    if not full_text or not full_text.strip():
        return "[No transcript available]"

    # Sort diarization segments by start time
    diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])

    # Map speaker IDs to participant numbers based on order of first appearance
    speaker_order = []
    for seg in diarization_segments:
        if seg["speaker"] not in speaker_order:
            speaker_order.append(seg["speaker"])
    speaker_map = {speaker: f"Participant {i+1}" for i, speaker in enumerate(speaker_order)}

    # Try to use word-level timestamps first (if available from whisper-1)
    words = getattr(transcript, 'words', [])

    if words:
        # Word-level approach - most accurate
        labeled_words = []

        for word_info in words:
            # Handle both dict and object access
            if hasattr(word_info, 'start'):
                word_start = word_info.start
                word_text = word_info.word if hasattr(word_info, 'word') else str(word_info)
            else:
                word_start = word_info.get('start', 0)
                word_text = word_info.get('word', '')

            if not word_text or not word_text.strip():
                continue

            # Find speaker for this word's timestamp
            speaker_id = _find_speaker_for_time(word_start, diarization_segments)
            assigned_speaker = speaker_map[speaker_id]
            labeled_words.append((assigned_speaker, word_text.strip()))

        if labeled_words:
            # Group consecutive words by same speaker
            output_lines = []
            current_speaker = labeled_words[0][0]
            current_words = [labeled_words[0][1]]

            for speaker, word in labeled_words[1:]:
                if speaker == current_speaker:
                    current_words.append(word)
                else:
                    output_lines.append(f"{current_speaker}: {' '.join(current_words)}")
                    current_speaker = speaker
                    current_words = [word]

            output_lines.append(f"{current_speaker}: {' '.join(current_words)}")
            return "\n\n".join(output_lines)

    # Try segment-level timestamps (if available)
    segments = getattr(transcript, 'segments', [])

    if segments:
        labeled_segments = []

        for segment in segments:
            if hasattr(segment, 'start'):
                seg_start = segment.start
                seg_end = segment.end
                seg_text = segment.text.strip() if segment.text else ""
            else:
                seg_start = segment.get('start', 0)
                seg_end = segment.get('end', 0)
                seg_text = segment.get('text', '').strip()

            if not seg_text:
                continue

            seg_mid = (seg_start + seg_end) / 2
            speaker_id = _find_speaker_for_time(seg_mid, diarization_segments)
            assigned_speaker = speaker_map[speaker_id]
            labeled_segments.append((assigned_speaker, seg_text))

        if labeled_segments:
            output_lines = []
            current_speaker = labeled_segments[0][0]
            current_texts = [labeled_segments[0][1]]

            for speaker, text in labeled_segments[1:]:
                if speaker == current_speaker:
                    current_texts.append(text)
                else:
                    output_lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                    current_speaker = speaker
                    current_texts = [text]

            output_lines.append(f"{current_speaker}: {' '.join(current_texts)}")
            return "\n\n".join(output_lines)

    # Fallback: Split text by diarization turns (for gpt-4o-transcribe which has no timestamps)
    # Merge consecutive segments from same speaker first
    merged_segments = []
    for seg in diarization_segments:
        if merged_segments and merged_segments[-1]["speaker"] == seg["speaker"]:
            merged_segments[-1]["end"] = seg["end"]
        else:
            merged_segments.append(seg.copy())

    # Calculate proportion of time for each speaker turn
    total_duration = merged_segments[-1]["end"] - merged_segments[0]["start"]

    # Split text proportionally based on speaker turn durations
    words_list = full_text.split()
    total_words = len(words_list)

    output_lines = []
    word_index = 0

    for seg in merged_segments:
        duration = seg["end"] - seg["start"]
        proportion = duration / total_duration if total_duration > 0 else 1 / len(merged_segments)
        words_for_segment = max(1, int(total_words * proportion))

        # Get words for this segment
        segment_words = words_list[word_index:word_index + words_for_segment]
        word_index += words_for_segment

        if segment_words:
            speaker_label = speaker_map[seg["speaker"]]
            output_lines.append(f"{speaker_label}: {' '.join(segment_words)}")

    # Add any remaining words to last speaker
    if word_index < total_words:
        remaining = ' '.join(words_list[word_index:])
        if output_lines:
            output_lines[-1] += ' ' + remaining
        else:
            output_lines.append(f"Participant 1: {remaining}")

    return "\n\n".join(output_lines) if output_lines else full_text


def create_download_file(transcript_text: str) -> str:
    """Create a downloadable text file with the transcript (UTF-8 with BOM for Hebrew support)."""
    temp_file = tempfile.mktemp(suffix=".txt")
    with open(temp_file, "w", encoding="utf-8-sig") as f:
        f.write("=" * 60 + "\n")
        f.write("AI CALL TRANSCRIPTION\n")
        f.write("=" * 60 + "\n\n")
        f.write(transcript_text)
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("Generated by AI Call Transcriber\n")
        f.write("=" * 60 + "\n")
    return temp_file


def cleanup_file(file_path: str) -> None:
    """Safely remove a temporary file."""
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
