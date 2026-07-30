"""Audio preprocessing helpers for Whisper compatibility.

The podcast source is slightly larger than the Whisper upload limit, so this
module provides a minimal ffmpeg-based transcode step before transcription.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .utils import ensure_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "audio_preprocessed"


class AudioPreprocessingError(RuntimeError):
    """Raised when preprocessing fails for a recoverable reason."""


def ensure_ffmpeg_available() -> str:
    """Return the ffmpeg executable path or raise a clear error."""

    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    raise AudioPreprocessingError(
        "ffmpeg is required for audio preprocessing but was not found. "
        "Install it first (for example with `brew install ffmpeg` on macOS) "
        "and then rerun the notebook."
    )


def _validate_audio_file(audio_file_path: str | Path) -> Path:
    """Validate the input path and return it as a resolved Path."""

    audio_path = Path(audio_file_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio path is not a file: {audio_path}")
    if not os.access(audio_path, os.R_OK):
        raise PermissionError(f"Audio file is not readable: {audio_path}")
    return audio_path


def preprocess_audio_for_transcription(
    audio_file_path: str | Path,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Create a smaller transcription-friendly copy of an audio file."""

    audio_path = _validate_audio_file(audio_file_path)
    source_size_bytes = audio_path.stat().st_size
    if source_size_bytes <= 0:
        raise AudioPreprocessingError(f"Audio file is empty: {audio_path}")

    ffmpeg = ensure_ffmpeg_available()
    target_dir = ensure_directory(output_dir or DEFAULT_OUTPUT_DIR)
    output_path = target_dir / f"{audio_path.stem}.mp3"

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "48k",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise AudioPreprocessingError(
            f"Failed to preprocess {audio_path.name} with ffmpeg: {stderr or exc}"
        ) from exc

    if not output_path.exists():
        raise AudioPreprocessingError(
            f"ffmpeg finished successfully but did not create {output_path}"
        )

    if output_path.stat().st_size >= source_size_bytes:
        raise AudioPreprocessingError(
            "Preprocessed audio is not smaller than the original file. "
            "Try a different bitrate or a different preprocessing strategy."
        )

    return output_path
