"""
Base source provider interface and SourceItem dataclass.

All source providers (Quran, Hadith, Seerah) implement BaseSourceProvider.
SourceItem is the canonical data structure that flows through the entire pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceItem:
    """A single verified source item from the Quran, Hadith, or Seerah."""

    source_type: str        # "quran" | "hadith" | "seerah"
    collection: str         # e.g. "Sahih al-Bukhari", "Quran"
    reference: str          # e.g. "Bukhari 3616" or "18:9-26"
    grade: Optional[str]    # "sahih" | "hasan" | None for Quran
    arabic_text: str
    translation_text: str
    translator: str
    source_url: str
    metadata: dict = field(default_factory=dict)  # Extra context (tafsir, chapter name, etc.)

    def __post_init__(self):
        """Normalize grade to lowercase if present."""
        if self.grade is not None:
            self.grade = self.grade.lower().strip()

    @property
    def is_quran(self) -> bool:
        return self.source_type == "quran"

    @property
    def is_hadith(self) -> bool:
        return self.source_type == "hadith"

    @property
    def is_seerah(self) -> bool:
        return self.source_type == "seerah"

    @property
    def citation_display(self) -> str:
        """Human-readable citation for on-screen overlay."""
        if self.is_quran:
            return f"Surah {self.metadata.get('surah_name', '')}, {self.reference}"
        return f"{self.collection}, Hadith {self.reference}"


class BaseSourceProvider(ABC):
    """Abstract base class for all content source providers."""

    @abstractmethod
    async def fetch_candidates(
        self, topic: Optional[str] = None, limit: int = 5
    ) -> list[SourceItem]:
        """
        Return items matching the topic. Return [] rather than inventing anything.

        Args:
            topic: Optional topic string to search for.
            limit: Maximum number of items to return.

        Returns:
            List of SourceItem instances. Empty list if no matches found.
            NEVER fabricates content — returns [] on any ambiguity.
        """

    @abstractmethod
    async def verify(self, item: SourceItem) -> bool:
        """
        Re-check grade/collection live before use. False on any mismatch.

        This is a secondary verification step that re-fetches the source
        to confirm the grade and collection are still valid. Called before
        the item enters the narration pipeline.
        """

    @abstractmethod
    async def fetch_by_reference(self, reference: str) -> Optional[SourceItem]:
        """
        Fetch a specific item by its canonical reference.

        Args:
            reference: e.g. "18:9" for Quran, "bukhari:3616" for hadith.

        Returns:
            SourceItem if found and valid, None otherwise.
        """
