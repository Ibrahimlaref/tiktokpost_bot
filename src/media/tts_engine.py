"""
TTS engine.

Handles two distinct audio pipelines:
1. Narration text → TTS synthesis (ElevenLabs primary, with stubs for Azure/Google)
2. Quran recitation → licensed qari audio download (NEVER synthesized via TTS)

The split is non-negotiable: reciting the actual mushaf is a precise practice
that should never be left to a TTS engine.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-speech engine with separate paths for narration and Quran recitation.

    Narration: Synthesized via ElevenLabs (or other TTS provider).
    Quran recitation: Downloaded from licensed qari audio source (everyayah.com).
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._output_dir = output_dir or Path(
            os.getenv("OUTPUT_DIR", "output")
        ) / "audio"
        self._cache_dir = cache_dir or Path(
            os.getenv("DATA_DIR", "data")
        ) / "audio_cache"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

        # Qari audio configuration (everyayah.com)
        self._qari_base_url = "https://everyayah.com/data"
        self._qari_reciter = "Alafasy_128kbps"

        self._client = httpx.AsyncClient(timeout=60.0)

    async def synthesize_narration(self, text: str, filename: Optional[str] = None) -> Path:
        """
        Synthesize narration text to speech using ElevenLabs.

        Args:
            text: The narration text to synthesize.
            filename: Optional output filename. Auto-generated if not provided.

        Returns:
            Path to the generated audio file (MP3).
        """
        if not filename:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            filename = f"narration_{text_hash}.mp3"

        output_path = self._output_dir / filename

        # Check cache
        cache_path = self._cache_dir / filename
        if cache_path.exists():
            logger.info(f"Using cached narration audio: {cache_path}")
            return cache_path

        if not self._elevenlabs_api_key:
            logger.warning(
                "No ElevenLabs API key configured. "
                "Generating placeholder audio file."
            )
            return await self._generate_placeholder_audio(output_path, text)

        try:
            response = await self._client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self._elevenlabs_voice_id}",
                headers={
                    "xi-api-key": self._elevenlabs_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.6,
                        "similarity_boost": 0.8,
                        "style": 0.3,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()

            # Write audio to file
            output_path.write_bytes(response.content)

            # Also cache it
            cache_path.write_bytes(response.content)

            logger.info(f"Narration audio generated: {output_path}")
            return output_path

        except httpx.HTTPError as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            raise TTSError(f"Failed to synthesize narration: {e}")

    async def fetch_quran_recitation(
        self, surah: int, ayah: int, filename: Optional[str] = None
    ) -> Path:
        """
        Download licensed qari recitation audio for a specific ayah.

        This NEVER uses TTS — it downloads real recitation from everyayah.com.

        Args:
            surah: Surah number (1-114).
            ayah: Ayah number within the surah.
            filename: Optional output filename.

        Returns:
            Path to the downloaded audio file (MP3).
        """
        if not filename:
            filename = f"recitation_{surah:03d}_{ayah:03d}.mp3"

        cache_path = self._cache_dir / filename
        if cache_path.exists():
            logger.info(f"Using cached qari recitation: {cache_path}")
            return cache_path

        # everyayah.com URL format: {base}/{reciter}/{surah:03d}{ayah:03d}.mp3
        audio_url = (
            f"{self._qari_base_url}/{self._qari_reciter}/"
            f"{surah:03d}{ayah:03d}.mp3"
        )

        try:
            response = await self._client.get(audio_url)
            response.raise_for_status()

            cache_path.write_bytes(response.content)
            logger.info(f"Qari recitation downloaded: {cache_path}")
            return cache_path

        except httpx.HTTPError as e:
            logger.error(f"Failed to download qari recitation for {surah}:{ayah}: {e}")
            raise TTSError(
                f"Failed to download Quran recitation for {surah}:{ayah}: {e}"
            )

    async def fetch_quran_recitation_range(
        self, surah: int, start_ayah: int, end_ayah: int
    ) -> list[Path]:
        """
        Download recitation audio for a range of ayahs.

        Args:
            surah: Surah number.
            start_ayah: First ayah number.
            end_ayah: Last ayah number (inclusive).

        Returns:
            List of paths to downloaded audio files, in order.
        """
        paths = []
        for ayah in range(start_ayah, end_ayah + 1):
            path = await self.fetch_quran_recitation(surah, ayah)
            paths.append(path)
        return paths

    async def _generate_placeholder_audio(self, output_path: Path, text: str) -> Path:
        """
        Generate a silent placeholder audio file when no TTS API is configured.

        This allows the pipeline to run end-to-end during development
        without an ElevenLabs key.
        """
        # Generate a minimal valid MP3 file (silence)
        # In production, this would use ffmpeg to create proper silence
        import struct

        # Create a minimal WAV file with silence
        wav_path = output_path.with_suffix(".wav")
        sample_rate = 22050
        duration = max(len(text.split()) / 2.5, 10)  # ~2.5 words/sec
        num_samples = int(sample_rate * duration)

        with open(wav_path, "wb") as f:
            # WAV header
            data_size = num_samples * 2  # 16-bit mono
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # chunk size
            f.write(struct.pack("<H", 1))   # PCM format
            f.write(struct.pack("<H", 1))   # mono
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", sample_rate * 2))  # byte rate
            f.write(struct.pack("<H", 2))   # block align
            f.write(struct.pack("<H", 16))  # bits per sample
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            # Write silence
            f.write(b"\x00" * data_size)

        logger.info(
            f"Placeholder audio generated ({duration:.1f}s silence): {wav_path}"
        )
        return wav_path

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class TTSError(Exception):
    """Raised when TTS synthesis or audio download fails."""
    pass
