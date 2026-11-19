"""
Video assembler.

Combines background visuals, TTS narration audio, citation overlay,
and AI-disclosure text into a 1080×1920 (9:16) vertical video using ffmpeg.

Burns in:
- On-screen citation (book + hadith number, or surah:ayah)
- "AI-Generated Content" disclosure text
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from src.sourcing.base_source import SourceItem

logger = logging.getLogger(__name__)


class VideoAssembler:
    """
    Assembles final TikTok-ready video from visual + audio components.

    Output: 1080×1920, 9:16, H.264/AAC, 60-90 seconds.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30,
        ai_disclosure_text: str = "AI-Generated Content",
    ):
        self._output_dir = output_dir or Path(
            os.getenv("OUTPUT_DIR", "output")
        ) / "videos"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._width = width
        self._height = height
        self._fps = fps
        self._ai_disclosure_text = ai_disclosure_text

    async def assemble(
        self,
        background_path: Path,
        audio_path: Path,
        source_item: SourceItem,
        narration_text: str,
        citation_text: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Assemble final video from components.

        Args:
            background_path: Path to background image (PNG).
            audio_path: Path to narration audio (MP3/WAV).
            source_item: The source item for metadata.
            narration_text: The narration text (for potential subtitle overlay).
            citation_text: Citation text to burn into the video.
            filename: Optional output filename.

        Returns:
            Path to the assembled video file (MP4).
        """
        if not filename:
            ref_safe = source_item.reference.replace(":", "_").replace("/", "_")
            filename = f"video_{source_item.source_type}_{ref_safe}.mp4"

        output_path = self._output_dir / filename

        # Build the ffmpeg command
        cmd = self._build_ffmpeg_command(
            background_path=background_path,
            audio_path=audio_path,
            citation_text=citation_text,
            output_path=output_path,
        )

        logger.info(f"Assembling video: {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute timeout
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                raise VideoAssemblyError(
                    f"FFmpeg exited with code {result.returncode}: "
                    f"{result.stderr[:500]}"
                )

            logger.info(
                f"Video assembled successfully: {output_path} "
                f"({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
            )
            return output_path

        except subprocess.TimeoutExpired:
            raise VideoAssemblyError("FFmpeg timed out after 5 minutes.")
        except FileNotFoundError:
            raise VideoAssemblyError(
                "FFmpeg not found. Please install ffmpeg and add it to PATH."
            )

    def _build_ffmpeg_command(
        self,
        background_path: Path,
        audio_path: Path,
        citation_text: str,
        output_path: Path,
    ) -> list[str]:
        """
        Build the ffmpeg command for video assembly.

        Creates a video from:
        - A static background image with slow Ken Burns (pan/zoom) effect
        - Narration audio
        - Citation text overlay (bottom-center, semi-transparent background)
        - AI-disclosure text (top-right corner)
        """
        # Escape special characters for ffmpeg drawtext
        citation_escaped = self._escape_ffmpeg_text(citation_text)
        disclosure_escaped = self._escape_ffmpeg_text(self._ai_disclosure_text)

        # Font configuration — prefer a system font that supports Arabic
        fontfile = self._find_font()
        font_opt = f":fontfile='{fontfile}'" if fontfile else ""

        # Ken Burns effect: slow zoom from 100% to 110% over the video duration
        # zoompan filter: zoom from 1 to 1.1, panning slightly
        ken_burns = (
            f"zoompan=z='min(zoom+0.0005,1.1)'"
            f":x='iw/2-(iw/zoom/2)'"
            f":y='ih/2-(ih/zoom/2)'"
            f":d=1:s={self._width}x{self._height}:fps={self._fps}"
        )

        # Citation overlay: bottom-center with semi-transparent box
        citation_filter = (
            f"drawtext=text='{citation_escaped}'"
            f"{font_opt}"
            f":fontsize=32"
            f":fontcolor=white"
            f":x=(w-text_w)/2"
            f":y=h-120"
            f":box=1"
            f":boxcolor=black@0.6"
            f":boxborderw=15"
        )

        # AI disclosure: top-right corner
        disclosure_filter = (
            f"drawtext=text='{disclosure_escaped}'"
            f"{font_opt}"
            f":fontsize=22"
            f":fontcolor=white@0.8"
            f":x=w-text_w-30"
            f":y=40"
            f":box=1"
            f":boxcolor=black@0.4"
            f":boxborderw=8"
        )

        # Combine filters
        filter_complex = f"[0:v]{ken_burns},{citation_filter},{disclosure_filter}[outv]"

        cmd = [
            "ffmpeg",
            "-y",                       # Overwrite output
            "-loop", "1",               # Loop the image
            "-i", str(background_path), # Input image
            "-i", str(audio_path),      # Input audio
            "-filter_complex", filter_complex,
            "-map", "[outv]",           # Use filtered video
            "-map", "1:a",              # Use audio from second input
            "-c:v", "libx264",          # H.264 video codec
            "-preset", "medium",
            "-crf", "23",               # Quality factor
            "-c:a", "aac",              # AAC audio codec
            "-b:a", "192k",
            "-shortest",                # End when shortest stream ends
            "-pix_fmt", "yuv420p",      # Compatibility
            str(output_path),
        ]

        return cmd

    @staticmethod
    def _find_font() -> Optional[str]:
        """Find a suitable font that supports Arabic text."""
        font_candidates = [
            # Windows
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialuni.ttf",
            # Linux
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            # macOS
            "/System/Library/Fonts/GeezaPro.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
        for font_path in font_candidates:
            if Path(font_path).exists():
                return font_path
        return None

    @staticmethod
    def _escape_ffmpeg_text(text: str) -> str:
        """Escape special characters for ffmpeg drawtext filter."""
        # ffmpeg drawtext requires escaping: ' : \
        text = text.replace("\\", "\\\\")
        text = text.replace("'", "'\\''")
        text = text.replace(":", "\\:")
        text = text.replace("%", "%%")
        return text

    async def get_audio_duration(self, audio_path: Path) -> float:
        """Get the duration of an audio file in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return float(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError) as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 75.0  # Default to 75s (middle of 60-90s range)


class VideoAssemblyError(Exception):
    """Raised when video assembly fails."""
    pass
