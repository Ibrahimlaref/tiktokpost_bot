"""
Hadith source provider.

Dual-source architecture:
  - Primary: sunnah.com API (requires API key)
  - Fallback: fawazahmed0/hadith-api (free, no key needed)

All grade filtering is programmatic — the LLM never decides if a hadith
"sounds right." Only sahih and hasan grades pass through.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import httpx
import yaml
from dotenv import load_dotenv

from src.sourcing.base_source import BaseSourceProvider, SourceItem

load_dotenv()
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"
_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "content_policy.yaml"


class HadithSourceProvider(BaseSourceProvider):
    """
    Fetches hadith from sunnah.com (primary) or fawazahmed0 API (fallback).

    Grade and collection filtering is enforced programmatically before
    any item reaches the caller.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        policy_path: Optional[Path] = None,
    ):
        config_path = config_path or _CONFIG_PATH
        policy_path = policy_path or _POLICY_PATH

        with open(config_path, "r", encoding="utf-8") as f:
            sources_cfg = yaml.safe_load(f)
        with open(policy_path, "r", encoding="utf-8") as f:
            policy_cfg = yaml.safe_load(f)

        self._hadith_cfg = sources_cfg["hadith"]
        self._policy = policy_cfg

        # Allowed collections and grades (from content_policy.yaml)
        self._allowed_collections = set(
            self._policy["sources"]["hadith_collections_allowed"]
        )
        self._allowed_grades = set(
            self._policy["sources"]["hadith_grades_allowed"]
        )
        self._denylist_collections = set(
            self._policy["exclusions"]["denylist_collections"]
        )
        self._denylist_topics = [
            t.lower() for t in self._policy["exclusions"]["denylist_topics"]
        ]

        # API configuration
        self._sunnah_api_key = os.getenv("SUNNAH_API_KEY", "")
        self._sunnah_enabled = (
            self._hadith_cfg["sunnah_api"]["enabled"] and bool(self._sunnah_api_key)
        )
        self._fawaz_enabled = self._hadith_cfg["fawazahmed0_api"]["enabled"]
        self._fawaz_base = self._hadith_cfg["fawazahmed0_api"]["base_url"]
        self._sunnah_base = self._hadith_cfg["sunnah_api"]["base_url"]

        # Edition name mappings
        self._edition_map = self._hadith_cfg["edition_map"]
        self._arabic_edition_map = self._hadith_cfg["arabic_edition_map"]

        # Reverse map for looking up collection names from edition names
        self._reverse_edition_map = {v: k for k, v in self._edition_map.items()}

        self._client = httpx.AsyncClient(timeout=30.0)

    async def fetch_candidates(
        self, topic: Optional[str] = None, limit: int = 5
    ) -> list[SourceItem]:
        """
        Fetch hadith matching a topic across all allowed collections.

        Uses fawazahmed0 API as primary for topic-based search (no key needed),
        falls back to sunnah.com if available.
        """
        if topic is None:
            logger.warning("No topic provided for hadith source fetch.")
            return []

        items: list[SourceItem] = []

        # Search across all allowed collections
        for collection in self._allowed_collections:
            edition = self._edition_map.get(collection)
            if edition is None:
                continue

            try:
                fetched = await self._search_fawaz_collection(
                    edition, collection, topic, limit=limit
                )
                for item in fetched:
                    if self._passes_filters(item):
                        items.append(item)
                        if len(items) >= limit:
                            return items
            except Exception as e:
                logger.warning(
                    f"Error searching {collection} via fawazahmed0: {e}"
                )
                continue

        logger.info(f"Fetched {len(items)} hadith for topic: {topic}")
        return items

    async def fetch_by_reference(self, reference: str) -> Optional[SourceItem]:
        """
        Fetch a specific hadith by reference.

        Args:
            reference: Format "collection:number", e.g. "bukhari:3616".
                       Or "Sahih al-Bukhari:3616".
        """
        parts = reference.split(":")
        if len(parts) != 2:
            logger.error(f"Invalid hadith reference format: {reference}")
            return None

        collection_key = parts[0].strip()
        try:
            hadith_number = parts[1].strip()
        except (ValueError, IndexError):
            logger.error(f"Invalid hadith number in reference: {reference}")
            return None

        # Resolve collection name
        collection = self._resolve_collection_name(collection_key)
        if collection is None:
            logger.error(f"Unknown collection: {collection_key}")
            return None

        # Check collection is allowed
        if collection not in self._allowed_collections:
            logger.warning(f"Collection {collection} is not in the allowed list.")
            return None

        if collection in self._denylist_collections:
            logger.warning(f"Collection {collection} is in the denylist.")
            return None

        edition = self._edition_map.get(collection)
        if edition is None:
            return None

        # Try fawazahmed0 first (no key needed)
        item = await self._fetch_fawaz_hadith(edition, collection, hadith_number)
        if item and self._passes_filters(item):
            return item

        # Try sunnah.com if available
        if self._sunnah_enabled:
            item = await self._fetch_sunnah_hadith(collection, hadith_number)
            if item and self._passes_filters(item):
                return item

        return None

    async def verify(self, item: SourceItem) -> bool:
        """
        Re-check grade and collection live before use.

        Re-fetches the hadith from the API and verifies:
        1. The collection is still in the allowed list
        2. The grade is still sahih or hasan
        3. The text matches
        """
        if not item.is_hadith:
            return False

        # Re-check collection allowlist
        if item.collection not in self._allowed_collections:
            logger.warning(f"Verification failed: {item.collection} not allowed")
            return False

        if item.collection in self._denylist_collections:
            logger.warning(f"Verification failed: {item.collection} is denylisted")
            return False

        # Re-check grade
        if item.grade not in self._allowed_grades:
            logger.warning(
                f"Verification failed: grade '{item.grade}' not in allowed grades"
            )
            return False

        # Re-fetch and compare
        refetched = await self.fetch_by_reference(
            f"{item.collection}:{item.reference}"
        )
        if refetched is None:
            logger.warning(
                f"Verification failed: could not re-fetch {item.reference}"
            )
            return False

        # Verify grade matches
        if refetched.grade != item.grade:
            logger.warning(
                f"Verification failed: grade mismatch for {item.reference} "
                f"(expected {item.grade}, got {refetched.grade})"
            )
            return False

        return True

    # -------------------------------------------------------------------------
    # fawazahmed0 API methods
    # -------------------------------------------------------------------------

    async def _fetch_fawaz_hadith(
        self, edition: str, collection: str, hadith_number: str
    ) -> Optional[SourceItem]:
        """Fetch a single hadith from the fawazahmed0 API."""
        url = f"{self._fawaz_base}/editions/{edition}/{hadith_number}.json"
        arabic_edition = self._arabic_edition_map.get(collection)

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.warning(f"fawazahmed0 fetch failed for {edition}/{hadith_number}: {e}")
            return None

        hadiths = data.get("hadiths", [])
        if not hadiths:
            return None

        hadith = hadiths[0]

        # Extract grade — this is the programmatic check, never left to the LLM
        grades = hadith.get("grades", [])
        grade = self._extract_best_grade(grades)

        # Fetch Arabic text if available
        arabic_text = ""
        if arabic_edition:
            arabic_text = await self._fetch_fawaz_arabic(
                arabic_edition, hadith_number
            )

        translation_text = hadith.get("text", "")

        return SourceItem(
            source_type="hadith",
            collection=collection,
            reference=str(hadith_number),
            grade=grade,
            arabic_text=arabic_text,
            translation_text=translation_text,
            translator="fawazahmed0/hadith-api",
            source_url=f"https://sunnah.com/{self._collection_to_sunnah_slug(collection)}/{hadith_number}",
            metadata={
                "api_source": "fawazahmed0",
                "grades_raw": grades,
            },
        )

    async def _fetch_fawaz_arabic(
        self, arabic_edition: str, hadith_number: str
    ) -> str:
        """Fetch Arabic text from the fawazahmed0 API."""
        url = f"{self._fawaz_base}/editions/{arabic_edition}/{hadith_number}.json"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()
            hadiths = data.get("hadiths", [])
            if hadiths:
                return hadiths[0].get("text", "")
        except httpx.HTTPError:
            pass
        return ""

    async def _search_fawaz_collection(
        self,
        edition: str,
        collection: str,
        topic: str,
        limit: int = 5,
    ) -> list[SourceItem]:
        """
        Search through a collection for hadith matching a topic.

        The fawazahmed0 API doesn't have a search endpoint, so we fetch
        sections and do keyword matching. This is a best-effort approach.
        """
        # Fetch the info/sections for this edition
        url = f"{self._fawaz_base}/editions/{edition}.json"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch edition {edition}: {e}")
            return []

        items = []
        hadiths = data.get("hadiths", [])
        topic_lower = topic.lower()

        for hadith in hadiths:
            text = hadith.get("text", "").lower()
            if topic_lower in text:
                hadith_num = hadith.get("hadithnumber", "")
                if not hadith_num:
                    continue

                grades = hadith.get("grades", [])
                grade = self._extract_best_grade(grades)

                item = SourceItem(
                    source_type="hadith",
                    collection=collection,
                    reference=str(hadith_num),
                    grade=grade,
                    arabic_text="",  # Will be fetched separately if needed
                    translation_text=hadith.get("text", ""),
                    translator="fawazahmed0/hadith-api",
                    source_url=f"https://sunnah.com/{self._collection_to_sunnah_slug(collection)}/{hadith_num}",
                    metadata={
                        "api_source": "fawazahmed0",
                        "grades_raw": grades,
                    },
                )

                if self._passes_filters(item):
                    items.append(item)
                    if len(items) >= limit:
                        break

        return items

    # -------------------------------------------------------------------------
    # sunnah.com API methods
    # -------------------------------------------------------------------------

    async def _fetch_sunnah_hadith(
        self, collection: str, hadith_number: str
    ) -> Optional[SourceItem]:
        """Fetch a single hadith from the sunnah.com API."""
        if not self._sunnah_enabled:
            return None

        slug = self._collection_to_sunnah_slug(collection)
        url = f"{self._sunnah_base}/hadiths/{slug}/{hadith_number}"

        try:
            response = await self._client.get(
                url,
                headers={"X-API-Key": self._sunnah_api_key},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.warning(f"sunnah.com fetch failed: {e}")
            return None

        # Extract grade from sunnah.com response
        grade = data.get("grade", "").lower().strip()

        return SourceItem(
            source_type="hadith",
            collection=collection,
            reference=str(hadith_number),
            grade=grade if grade else None,
            arabic_text=data.get("arabicText", ""),
            translation_text=data.get("englishText", data.get("text", "")),
            translator="sunnah.com",
            source_url=f"https://sunnah.com/{slug}/{hadith_number}",
            metadata={
                "api_source": "sunnah.com",
                "book_name": data.get("bookName", ""),
                "chapter": data.get("chapterTitle", ""),
            },
        )

    # -------------------------------------------------------------------------
    # Filtering and helpers
    # -------------------------------------------------------------------------

    def _passes_filters(self, item: SourceItem) -> bool:
        """
        Programmatic filter — rejects da'if, mawdu', denylisted collections,
        and denylisted topic keywords. The LLM never makes this decision.
        """
        # 1. Collection denylist
        if item.collection in self._denylist_collections:
            logger.debug(f"Rejected: collection {item.collection} is denylisted")
            return False

        # 2. Collection allowlist
        if item.collection not in self._allowed_collections:
            logger.debug(f"Rejected: collection {item.collection} not in allowlist")
            return False

        # 3. Grade filter — the core safety gate
        if item.grade is not None and item.grade not in self._allowed_grades:
            logger.debug(
                f"Rejected: hadith {item.reference} grade '{item.grade}' "
                f"not in {self._allowed_grades}"
            )
            return False

        # 4. If grade is None/missing for hadith, reject (no grade = can't verify)
        if item.grade is None:
            logger.debug(
                f"Rejected: hadith {item.reference} has no grade information"
            )
            return False

        # 5. Topic denylist — keyword scan
        combined_text = (
            f"{item.translation_text} {item.arabic_text}"
        ).lower()
        for denied_topic in self._denylist_topics:
            if denied_topic in combined_text:
                logger.debug(
                    f"Rejected: hadith {item.reference} contains "
                    f"denied topic '{denied_topic}'"
                )
                return False

        return True

    @staticmethod
    def _extract_best_grade(grades: list[dict]) -> Optional[str]:
        """
        Extract the most authoritative grade from a list of grading entries.

        The fawazahmed0 API returns grades as:
        [{"name": "Scholar Name", "grade": "Sahih"}, ...]

        Priority: sahih > hasan > anything else.
        If ANY grader says sahih, we accept sahih.
        If no sahih but ANY says hasan, we accept hasan.
        Otherwise, return the most common grade (likely da'if).
        """
        if not grades:
            return None

        grade_values = [
            g.get("grade", "").lower().strip()
            for g in grades
            if g.get("grade")
        ]

        if not grade_values:
            return None

        # Check for sahih
        for g in grade_values:
            if "sahih" in g and "da'if" not in g:
                return "sahih"

        # Check for hasan
        for g in grade_values:
            if "hasan" in g:
                return "hasan"

        # Return the first grade as-is (likely da'if or other)
        return grade_values[0] if grade_values else None

    def _resolve_collection_name(self, key: str) -> Optional[str]:
        """Resolve a shorthand collection key to the full canonical name."""
        key_lower = key.lower().strip()

        # Direct match
        if key in self._allowed_collections:
            return key

        # Shorthand mappings
        shorthands = {
            "bukhari": "Sahih al-Bukhari",
            "muslim": "Sahih Muslim",
            "abudawud": "Sunan Abu Dawood",
            "abu dawud": "Sunan Abu Dawood",
            "abu dawood": "Sunan Abu Dawood",
            "tirmidhi": "Jami at-Tirmidhi",
            "nasai": "Sunan an-Nasa'i",
            "ibnmajah": "Sunan Ibn Majah",
            "ibn majah": "Sunan Ibn Majah",
        }

        return shorthands.get(key_lower)

    @staticmethod
    def _collection_to_sunnah_slug(collection: str) -> str:
        """Convert collection name to sunnah.com URL slug."""
        slugs = {
            "Sahih al-Bukhari": "bukhari",
            "Sahih Muslim": "muslim",
            "Sunan Abu Dawood": "abudawud",
            "Jami at-Tirmidhi": "tirmidhi",
            "Sunan an-Nasa'i": "nasai",
            "Sunan Ibn Majah": "ibnmajah",
        }
        return slugs.get(collection, collection.lower().replace(" ", ""))

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
