"""
Seerah source provider.

Wraps HadithSourceProvider to fetch hadith related to Seerah (Prophetic biography)
episodes. Every Seerah narrative must be cross-referenced against at least one
hadith from the Kutub al-Sittah — no standalone biographical claims.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from src.sourcing.base_source import BaseSourceProvider, SourceItem
from src.sourcing.hadith_source import HadithSourceProvider

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"
_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "content_policy.yaml"


class SeerahSourceProvider(BaseSourceProvider):
    """
    Fetches Seerah-related hadith with mandatory cross-referencing.

    Every Seerah episode must have at least one backing hadith graded
    sahih or hasan from the Kutub al-Sittah. This is enforced by
    seerah_requires_hadith_crossref in content_policy.yaml.
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

        self._seerah_cfg = sources_cfg.get("seerah", {})
        self._episode_mappings = self._seerah_cfg.get("episode_mappings", {})
        self._requires_crossref = policy_cfg["sources"].get(
            "seerah_requires_hadith_crossref", True
        )

        # Wrap the hadith provider — all filtering is delegated to it
        self._hadith_provider = HadithSourceProvider(
            config_path=config_path, policy_path=policy_path
        )

    async def fetch_candidates(
        self, topic: Optional[str] = None, limit: int = 5
    ) -> list[SourceItem]:
        """
        Fetch hadith related to a Seerah topic.

        Looks up the topic in episode_mappings to get relevant search terms,
        then delegates to HadithSourceProvider. Returns items re-tagged as
        source_type="seerah" with the backing hadith reference preserved.
        """
        if topic is None:
            logger.warning("No topic provided for Seerah source fetch.")
            return []

        topic_key = topic.lower().strip()

        # Find matching episode mapping
        mapping = self._episode_mappings.get(topic_key)
        if mapping is None:
            # Try partial match
            for key, val in self._episode_mappings.items():
                if key in topic_key or topic_key in key:
                    mapping = val
                    topic_key = key
                    break

        if mapping is None:
            # No specific mapping — try a direct hadith search
            logger.info(
                f"No Seerah episode mapping for '{topic}', "
                f"trying direct hadith search."
            )
            search_terms = [topic]
            mapping = {"search_terms": search_terms, "collections": []}

        search_terms = mapping.get("search_terms", [])
        preferred_collections = mapping.get("collections", [])

        all_items: list[SourceItem] = []

        # Search with each term
        for term in search_terms:
            hadith_items = await self._hadith_provider.fetch_candidates(
                topic=term, limit=limit
            )

            for item in hadith_items:
                # Prefer items from the specified collections for this episode
                if preferred_collections and item.collection not in preferred_collections:
                    continue

                # Re-tag as seerah source type
                seerah_item = SourceItem(
                    source_type="seerah",
                    collection=item.collection,
                    reference=item.reference,
                    grade=item.grade,
                    arabic_text=item.arabic_text,
                    translation_text=item.translation_text,
                    translator=item.translator,
                    source_url=item.source_url,
                    metadata={
                        **item.metadata,
                        "seerah_episode": topic_key,
                        "backing_hadith_ref": f"{item.collection}:{item.reference}",
                    },
                )
                all_items.append(seerah_item)

                if len(all_items) >= limit:
                    return all_items

        if self._requires_crossref and not all_items:
            logger.warning(
                f"No hadith cross-reference found for Seerah topic: {topic}. "
                f"Returning empty list (seerah_requires_hadith_crossref=true)."
            )
            return []

        logger.info(f"Fetched {len(all_items)} Seerah items for topic: {topic}")
        return all_items

    async def fetch_by_reference(self, reference: str) -> Optional[SourceItem]:
        """
        Fetch a specific Seerah item by its backing hadith reference.

        Delegates to HadithSourceProvider and re-tags as seerah.
        """
        item = await self._hadith_provider.fetch_by_reference(reference)
        if item is None:
            return None

        return SourceItem(
            source_type="seerah",
            collection=item.collection,
            reference=item.reference,
            grade=item.grade,
            arabic_text=item.arabic_text,
            translation_text=item.translation_text,
            translator=item.translator,
            source_url=item.source_url,
            metadata={
                **item.metadata,
                "backing_hadith_ref": f"{item.collection}:{item.reference}",
            },
        )

    async def verify(self, item: SourceItem) -> bool:
        """
        Verify a Seerah item by verifying its backing hadith.

        If seerah_requires_hadith_crossref is true (default), the item
        must have a valid backing hadith reference.
        """
        if not item.is_seerah:
            return False

        # Verify the backing hadith exists and is valid
        backing_ref = item.metadata.get("backing_hadith_ref")
        if self._requires_crossref and not backing_ref:
            logger.warning(
                f"Verification failed: Seerah item has no backing hadith reference"
            )
            return False

        # Delegate to hadith verification
        hadith_item = SourceItem(
            source_type="hadith",
            collection=item.collection,
            reference=item.reference,
            grade=item.grade,
            arabic_text=item.arabic_text,
            translation_text=item.translation_text,
            translator=item.translator,
            source_url=item.source_url,
            metadata=item.metadata,
        )
        return await self._hadith_provider.verify(hadith_item)

    async def close(self):
        """Close underlying providers."""
        await self._hadith_provider.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
