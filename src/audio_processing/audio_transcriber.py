import logging
import os
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
from pathlib import Path
import json
from decouple import config
import assemblyai as aai

from src.document_processing.doc_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)  # Fixed: should be INFO not info
logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    speaker: str
    start_time: float
    end_time: float
    text: str
    confidence: float

    def get_timestamp(self) -> str:
        def format_time(seconds):
            minutes = seconds // 60
            seconds = seconds % 60

            return f"{minutes:02d}:{seconds:02d}"

        return f"[{format_time(self.start_time)} - {format_time(self.end_time)}]"


class AudioTranscriber:
    def __init__(self, api_key: str):
        self.api_key = api_key
        aai.settings.api_key = api_key

        self.supported_formats = {
            ".mp3",
            ".wav",
            ".m4a",
            ".aac",
            ".ogg",
            ".flac",
            ".wma",
            ".opus",
            ".mp4",
            ".mov",
            ".avi",
        }

        logger.info("AudioTranscriber initialized with AssemblyAI")  # Fixed typo

    def transcribe_audio(
        self,
        audio_path: str,
        enable_speaker_diarization: bool = True,
        enable_auto_punctuation: bool = True,
        audio_language: str = "en",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> List[DocumentChunk]:

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")  # Fixed typo

        if audio_path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}"
            )  # Fixed typos

        logger.info(f"Transcription started for: {audio_path.name}")

        try:
            config = aai.TranscriptionConfig(
                speaker_labels=enable_speaker_diarization,
                punctuate=enable_auto_punctuation,
                language_code=audio_language,
                speech_models=["universal-3-pro", "universal-2"],
            )

            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(str(audio_path))

            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(
                    f"Transcription failed: {transcript.error}"
                )  # Fixed typo

            logger.info(f"Transcription completed for: {audio_path.name}")  # Fixed typo

            return self._process_transcript_to_chunks(  # Fixed typo
                transcript, audio_path.name, chunk_size, chunk_overlap
            )
        except Exception as e:
            logger.error(
                f"Transcription failed for audio: {audio_path.name}: {str(e)}"  # Fixed spacing
            )
            raise

    def _process_transcript_to_chunks(  # Fixed typo
        self,
        transcript: aai.Transcript,
        source_file: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[DocumentChunk]:

        chunks = []

        transcript_metadata = {
            "duration_seconds": transcript.audio_duration,
            "confidence": transcript.confidence,
            "audio_url": transcript.audio_url,
            "transcription_id": transcript.id,
        }

        if hasattr(transcript, "utterances") and transcript.utterances:
            chunks = self._create_chunks_with_speakers(  # Fixed typo
                transcript.utterances,  # Fixed typo
                source_file,
                chunk_size,
                chunk_overlap,
                transcript_metadata,
            )
        else:
            chunks = self._create_chunks_without_speakers(
                transcript.text,
                source_file,
                chunk_size,
                chunk_overlap,
                transcript_metadata,
            )

        logger.info(f"Created {len(chunks)} chunks from transcript")  # Fixed typo

        return chunks

    def _create_chunks_with_speakers(
        self,
        utterances: List[aai.Utterance],  # Fixed typo
        source_file: str,
        chunk_size: int,
        chunk_overlap: int,
        base_metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        chunks = []
        current_text = ""
        current_speakers = []
        current_timestamps = []
        chunk_index = 0
        start_char = 0

        for utterance in utterances:  # Fixed typo
            speaker_label = f"Speaker {utterance.speaker}"
            timestamp_str = f"[{self.format_milliseconds(utterance.start)}]"

            speaker_text = f"{timestamp_str} {speaker_label}: {utterance.text}\n"

            if len(current_text + speaker_text) > chunk_size and current_text:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update(
                    {
                        "speakers": list(
                            set(current_speakers)
                        ),  # Fixed: was current_text
                        "start_timestamp": (
                            current_timestamps[0] if current_timestamps else None
                        ),
                        "end_timestamp": (
                            current_timestamps[-1] if current_timestamps else None
                        ),
                        "speaker_count": len(set(current_speakers)),
                    }
                )

                chunk = DocumentChunk(
                    content=current_text.strip(),
                    source_file=source_file,
                    source_type="audio",
                    page_number=None,
                    chunk_id=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(current_text) - 1,
                    metadata=chunk_metadata,
                )

                chunks.append(chunk)
                chunk_index += 1  # Added: increment chunk index

                overlap_text = (
                    current_text[-chunk_overlap:] if chunk_overlap > 0 else ""
                )
                current_text = overlap_text + speaker_text
                start_char += len(current_text) - len(overlap_text) - len(speaker_text)

                current_speakers = [speaker_label]
                current_timestamps = [utterance.start, utterance.end]

            else:
                current_text += speaker_text  # Fixed: was adding list
                current_speakers.append(speaker_label)
                current_timestamps.extend([utterance.start, utterance.end])

        if current_text.strip():
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update(
                {
                    "speakers": list(set(current_speakers)),
                    "start_timestamp": (
                        current_timestamps[0] if current_timestamps else None
                    ),
                    "end_timestamp": (
                        current_timestamps[-1] if current_timestamps else None
                    ),  # Fixed typo
                    "speaker_count": len(set(current_speakers)),
                }
            )

            chunk = DocumentChunk(
                content=current_text.strip(),
                source_file=source_file,
                source_type="audio",
                page_number=None,
                chunk_id=chunk_index,
                start_char=start_char,
                end_char=start_char + len(current_text) - 1,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

        return chunks

    def _create_chunks_without_speakers(  # Added missing method
        self,
        text: str,
        source_file: str,
        chunk_size: int,
        chunk_overlap: int,
        base_metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        chunks = []
        start_char = 0
        chunk_index = 0

        while start_char < len(text):
            end_char = min(start_char + chunk_size, len(text))
            chunk_text = text[start_char:end_char]

            chunk = DocumentChunk(
                content=chunk_text.strip(),
                source_file=source_file,
                source_type="audio",
                page_number=None,
                chunk_id=chunk_index,
                start_char=start_char,
                end_char=end_char - 1,
                metadata=base_metadata.copy(),
            )
            chunks.append(chunk)

            chunk_index += 1
            start_char = end_char - chunk_overlap if chunk_overlap > 0 else end_char

        return chunks

    def format_milliseconds(self, ms: int) -> str:
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60

        return f"{minutes:02d}:{seconds:02d}"  # Fixed spacing

    def get_transcript_summary(self, audio_path: str) -> Dict[str, Any]:
        try:
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                summarization=True,
                speech_models=["universal-3-pro", "universal-2"],
            )

            transcriber = aai.Transcriber(config=config)  # Fixed typo
            transcript = transcriber.transcribe(str(audio_path))  # Fixed typo

            if transcript.status == aai.TranscriptStatus.error:
                return {"error": transcript.error}

            return {
                "id": transcript.id,
                "file_name": Path(audio_path).name,
                "duration_seconds": transcript.audio_duration,
                "confidence": transcript.confidence,
                "word_count": len(transcript.text.split()) if transcript.text else 0,
                "character_count": len(transcript.text) if transcript.text else 0,
                "summary": getattr(transcript, "summary", "Not available"),
                "speaker_count": (
                    len(set(u.speaker for u in transcript.utterances))
                    if hasattr(transcript, "utterances") and transcript.utterances
                    else 1
                ),
            }

        except Exception as e:
            logger.error(
                f"Failed to get transcript for file: {Path(audio_path).name}: {str(e)}"
            )  # Fixed
            return {"error": str(e)}  # Fixed: return dict properly

    def batch_transcribe(
        self, audio_paths: List[str]
    ) -> List[List[DocumentChunk]]:  # Fixed typo
        all_chunks = []

        for audio_path in audio_paths:  # Fixed typo
            try:
                chunks = self.transcribe_audio(audio_path)
                all_chunks.append(chunks)
                logger.info(
                    f"Successfully transcribed {audio_path}: {len(chunks)} chunks"
                )
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {str(e)}")
                all_chunks.append([])

        return all_chunks


if __name__ == "__main__":
    api_key = config("ASSEMBLYAI_API_KEY")

    if not api_key:
        print("Please set ASSEMBLYAI_API_KEY environment variable")
        exit(1)

    transcriber = AudioTranscriber(api_key)

    try:
        file_path = "data/harvard.wav"
        audio_file = str(Path(file_path).resolve())

        summary = transcriber.get_transcript_summary(audio_file)
        print(f"Transcript Summary: {summary}")

        chunks = transcriber.transcribe_audio(audio_file)

        print(f"\nTranscription Results:")  # Fixed spacing
        print(f"Generated {len(chunks)} chunks")

        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Content: {chunk.content[:200]}...")
            print(f"Speakers: {chunk.metadata.get('speakers', [])}")
            print(f"Citation: Source: {chunk.source_file}, Type: Audio Transcript")

    except Exception as e:
        print(f"Error in transcription example: {e}")
