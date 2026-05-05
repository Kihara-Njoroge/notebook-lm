import logging
import os
import tempfile
import json
from typing import Dict, List, Any, Optional
import yt_dlp
import assemblyai as aai
from pathlib import Path
from decouple import config

from src.document_processing.doc_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YoutubeTranscriber:
    def __init__(self, assemblyai_api_key: str):
        self.assembly_ai_api_key = assemblyai_api_key
        self.temp_dir = Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)

        aai.settings.api_key = assemblyai_api_key

        logger.info("YouTubeTranscriber initialized")

    def extract_video_id(self, url: str) -> Optional[str]:
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = None
        return video_id

    def download_audio(self, url: str) -> str:
        video_id = self.extract_video_id(url)

        if not video_id:
            raise ValueError("Could  not extract video ID from URL")

        expected_path = self.temp_dir / f"{video_id}.m4a"

        if expected_path.exists():
            logger.info("Audio already exists: {expected_path}")
            return str(expected_path)

        logger.info(f"Downloading audio from: {url}")

        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "outtmpl": str(self.temp_dir / "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(url)

            if error_code != 0:
                raise Exception(f"yt-dlp download failed with error code: {error_code}")

            if not expected_path.exists():
                raise FileNotFoundError(
                    f"Expected audio file not found: {expected_path}"
                )

            logger.info(f"Audio downloaded successfully: {expected_path}")

            return str(expected_path)

    def transcribe_youtube_video(
        self, url: str, cleanup_audio: bool = True
    ) -> List[DocumentChunk]:
        try:
            audi_path = self.download_audio(url)

            config = aai.TranscriptionConfig(
                speaker_labels=True,
                punctuate=True,
                speech_models=["universal-3-pro", "universal-2"],
            )

            logger.info("Starting transcription with speaker diarization...")
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(audi_path)

            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")

            chunks = []
            video_id = self.extract_video_id(url)

            for i, utterance in enumerate(transcript.utterances):
                chunk = DocumentChunk(
                    content=f"Speaker {utterance.speaker}: {utterance.text}",
                    source_file=f"Youtube Videp {video_id}",
                    source_type="youtube",
                    chunk_index=i,
                    start_char=utterance.start,
                    end_char=utterance.end,
                    metadata={
                        "speaker": utterance.speaker,
                        "start_time": utterance.start,
                        "end_time": utterance.end,
                        "confidence": getattr(utterance, "confidence", None),
                        "video_url": url,
                        "video_id": video_id,
                    },
                )
                chunks.append(chunk)

                logger.info(f"Transcription completed: {len(chunks)} utterances")

                if cleanup_audio and os.path.exists(audi_path):
                    os.unlink(audi_path)
                    logger.info("Audio file cleaned up")

                return chunks
        except Exception as e:
            logger.error(f"Error transcribing Youtube video: {str(e)}")
            raise e

    def clean_up_temp_files(self):
        try:
            if self.temp_dir.exists():
                for file in self.temp_dir.glob("*.m4a"):
                    file.unlink
                logger.info("Temp file cleanup")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")


if __name__ == "__main__":
    api_key = config("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("Please set ASSEMBLYAI_API_KEY environment variable")
        exit(1)

    transcriber = YoutubeTranscriber(api_key)

    try:
        test_url = "https://www.youtube.com/watch?v=D26sUZ6DHNQ"
        chunks = transcriber.transcribe_youtube_video(test_url)
        print(f"Transcribed {len(chunks)} utterances:")

        for chunk in chunks[:5]:
            print(f" {chunk.content}")

    except Exception as e:
        print(f"Error: {e}")
