"""
Quran source provider.

Fetches verses and tafsir from the Quran.com API v4.
Maps story topics to known surah:ayah ranges defined in config/sources.yaml.
"""

import logging
from pathlib import Path
from typing import Optional

import httpx
import yaml

from src.sourcing.base_source import BaseSourceProvider, SourceItem

logger = logging.getLogger(__name__)

# Load sources config
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"


class QuranSourceProvider(BaseSourceProvider):
    """Fetches Quran verses and tafsir via the Quran.com API v4."""

    def __init__(self, config_path: Optional[Path] = None):
        config_path = config_path or _CONFIG_PATH
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)["quran"]

        self._api_base = self._config["api_base"]
        self._translation_id = self._config["translation_id"]
        self._tafsir_id = self._config["tafsir_id"]
        self._story_mappings = self._config.get("story_mappings", {})
        self._client = httpx.AsyncClient(timeout=30.0)

    async def fetch_candidates(
        self, topic: Optional[str] = None, limit: int = 5
    ) -> list[SourceItem]:
        """
        Fetch Quran verses matching a topic.

        If the topic matches a known story mapping, fetches the corresponding
        ayah range. Otherwise returns an empty list — never guesses.
        """
        if topic is None:
            logger.warning("No topic provided for Quran source fetch.")
            return []

        # Normalize topic for lookup
        topic_key = topic.lower().strip()

        # Check story mappings
        mapping = self._story_mappings.get(topic_key)
        if mapping is None:
            # Try partial match
            for key, val in self._story_mappings.items():
                if key in topic_key or topic_key in key:
                    mapping = val
                    topic_key = key
                    break

        if mapping is None:
            logger.info(f"No story mapping found for topic: {topic}")
            return []

        surah = mapping["surah"]
        start_ayah = mapping["start_ayah"]
        end_ayah = mapping["end_ayah"]

        return await self._fetch_ayah_range(surah, start_ayah, end_ayah, limit)

    async def _fetch_ayah_range(
        self, surah: int, start_ayah: int, end_ayah: int, limit: int = 5
    ) -> list[SourceItem]:
        """Fetch a range of ayahs using the by_key endpoint for each verse."""
        items = []

        # Get chapter info for metadata
        chapter_info = await self._get_chapter_info(surah)

        # Fetch each verse individually via /verses/by_key
        # This is more reliable than by_chapter pagination
        for ayah_num in range(start_ayah, end_ayah + 1):
            verse_key = f"{surah}:{ayah_num}"
            item = await self._fetch_single_verse(verse_key, chapter_info)
            if item:
                items.append(item)
                if len(items) >= limit:
                    break

        logger.info(
            f"Fetched {len(items)} Quran verses for surah {surah}, "
            f"ayah {start_ayah}-{end_ayah}"
        )
        return items

    async def _fetch_single_verse(
        self, verse_key: str, chapter_info: Optional[dict] = None
    ) -> Optional[SourceItem]:
        """Fetch a single verse using the /verses/by_key endpoint."""
        try:
            response = await self._client.get(
                f"{self._api_base}/verses/by_key/{verse_key}",
                params={
                    "language": "en",
                    "translations": str(self._translation_id),
                    "fields": "text_uthmani,verse_key",
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch verse {verse_key}: {e}")
            return None

        verse = data.get("verse", {})
        if not verse:
            return None

        actual_key = verse.get("verse_key", verse_key)
        parts = actual_key.split(":")
        surah = int(parts[0]) if len(parts) == 2 else 0
        ayah_num = int(parts[1]) if len(parts) == 2 else 0

        # Extract Arabic text
        arabic_text = verse.get("text_uthmani", "")

        # Extract translation
        translations = verse.get("translations", [])
        translation_text = ""
        if translations:
            translation_text = translations[0].get("text", "")
            translation_text = self._strip_html(translation_text)

        # Fetch chapter info if not provided
        if chapter_info is None:
            chapter_info = await self._get_chapter_info(surah)

        return SourceItem(
            source_type="quran",
            collection="Quran",
            reference=actual_key,
            grade=None,
            arabic_text=arabic_text,
            translation_text=translation_text,
            translator="Sahih International",
            source_url=f"https://quran.com/{actual_key}",
            metadata={
                "surah": surah,
                "ayah": ayah_num,
                "surah_name": chapter_info.get("name_simple", ""),
                "surah_name_arabic": chapter_info.get("name_arabic", ""),
                "revelation_place": chapter_info.get("revelation_place", ""),
            },
        )

    async def fetch_by_reference(self, reference: str) -> Optional[SourceItem]:
        """
        Fetch a specific ayah by verse key.

        Args:
            reference: Verse key in format "surah:ayah", e.g. "18:9".
        """
        parts = reference.split(":")
        if len(parts) != 2:
            logger.error(f"Invalid Quran reference format: {reference}")
            return None

        try:
            int(parts[0])
            int(parts[1])
        except ValueError:
            logger.error(f"Non-numeric Quran reference: {reference}")
            return None

        return await self._fetch_single_verse(reference)

    async def fetch_tafsir(self, verse_key: str) -> Optional[str]:
        """
        Fetch tafsir (exegesis) for a specific verse.

        Args:
            verse_key: e.g. "18:9"

        Returns:
            Tafsir text as string, or None if unavailable.
        """
        try:
            response = await self._client.get(
                f"{self._api_base}/tafsirs/{self._tafsir_id}/by_ayah/{verse_key}",
            )
            response.raise_for_status()
            data = response.json()

            tafsir = data.get("tafsir", {})
            text = tafsir.get("text", "")
            return self._strip_html(text) if text else None

        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch tafsir for {verse_key}: {e}")
            return None

    async def verify(self, item: SourceItem) -> bool:
        """
        Verify a Quran SourceItem by re-fetching from the API.

        For Quran, grade is always None (no grading concept).
        Verification re-fetches the verse to confirm text matches.
        """
        if not item.is_quran:
            return False

        refetched = await self.fetch_by_reference(item.reference)
        if refetched is None:
            logger.warning(f"Verification failed: could not re-fetch {item.reference}")
            return False

        # Verify the Arabic text matches
        if refetched.arabic_text != item.arabic_text:
            logger.warning(
                f"Verification failed: Arabic text mismatch for {item.reference}"
            )
            return False

        return True

    async def _get_chapter_info(self, surah: int) -> dict:
        """Fetch chapter metadata."""
        try:
            response = await self._client.get(
                f"{self._api_base}/chapters/{surah}",
                params={"language": "en"},
            )
            response.raise_for_status()
            return response.json().get("chapter", {})
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch chapter info for surah {surah}: {e}")
            return {}

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from API response text."""
        import re
        return re.sub(r"<[^>]+>", "", text).strip()

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
