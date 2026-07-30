"""Podcast transcription helpers.

This module is intentionally independent from the rest of the pipeline so the
podcast can be converted to text before any retrieval work begins.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ensure_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "transcripts"
TRANSCRIPTION_MODEL = "whisper-1"


class TranscriptionError(RuntimeError):
    """Raised when transcription fails for a recoverable reason."""


@dataclass(slots=True)
class TranscriptionResult:
    """Return type for a completed transcription."""

    text: str
    metadata: dict[str, Any]
    transcript_path: Path
    metadata_path: Path


def _probe_duration_seconds(audio_path: Path) -> float | None:
    """Try to read the audio duration locally with `ffprobe`.

    This is optional and only used when the transcription response does not
    provide a duration.
    """

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    output = completed.stdout.strip()
    if not output:
        return None

    try:
        return float(output)
    except ValueError:
        return None


def _output_paths(audio_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Build transcript and metadata file paths for a given audio file."""

    target_dir = ensure_directory(output_dir or DEFAULT_OUTPUT_DIR)
    stem = audio_path.stem
    return target_dir / f"{stem}.md", target_dir / f"{stem}.json"


def transcribe_audio_file(
    audio_file_path: str | Path,
    *,
    client: Any | None = None,
    model: str = TRANSCRIPTION_MODEL,
    output_dir: Path | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file and persist the transcript plus metadata.

    Args:
        audio_file_path: Path to the podcast audio file.
        client: Optional OpenAI client for dependency injection in tests.
        model: Speech-to-text model name.
        output_dir: Optional custom output directory for transcripts.

    Returns:
        A `TranscriptionResult` with transcript text, metadata, and file paths.
    """

    audio_path = Path(audio_file_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio path is not a file: {audio_path}")

    try:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency issue in local env only
                raise TranscriptionError(
                    "The openai package is required to transcribe audio."
                ) from exc

            client = OpenAI()

        openai_client = client
        with audio_path.open("rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
            )
    except Exception as exc:  # noqa: BLE001 - wrap API/auth/network failures cleanly
        raise TranscriptionError(f"Failed to transcribe {audio_path.name}: {exc}") from exc

    text = getattr(transcript, "text", "") or ""
    language = getattr(transcript, "language", None)
    duration = getattr(transcript, "duration", None)
    if duration is None:
        duration = _probe_duration_seconds(audio_path)

    metadata: dict[str, Any] = {
        "source": "podcast",
        "filename": audio_path.name,
        "transcription_model": model,
        "language": language,
        "duration": duration,
        "speaker": getattr(transcript, "speaker", None),
    }

    transcript_path, metadata_path = _output_paths(audio_path, output_dir=output_dir)

    try:
        transcript_path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise TranscriptionError(f"Failed to save transcript for {audio_path.name}: {exc}") from exc

    return TranscriptionResult(
        text=text,
        metadata=metadata,
        transcript_path=transcript_path,
        metadata_path=metadata_path,
    )
