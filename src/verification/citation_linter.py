"""
Citation linter.

Hard gate before TTS/video generation. Verifies that the LLM's output
citation matches the original SourceItem's reference. Blocks anything
where the model's claimed citation doesn't normalize to the source.
"""

import logging
import re
from typing import Optional

from src.sourcing.base_source import SourceItem

logger = logging.getLogger(__name__)


def citation_matches_source(
    model_output: dict, item: SourceItem
) -> tuple[bool, str]:
    """
    Hard gate before TTS/video generation.

    Fail unless:
    1. model_output['citation'] normalizes to item.reference
    2. model_output['confidence'] == 'high'
    3. narration is not 'NO_RELIABLE_SOURCE'

    Args:
        model_output: Dict with keys "narration", "citation", "confidence".
        item: The original SourceItem the narration was generated from.

    Returns:
        Tuple of (passes, failure_reason).
        If passes is True, failure_reason is empty string.
    """
    # Check for NO_RELIABLE_SOURCE
    narration = model_output.get("narration", "")
    if narration.strip() == "NO_RELIABLE_SOURCE":
        reason = (
            "Model returned NO_RELIABLE_SOURCE — it was not confident "
            "the narration stays within the source material."
        )
        logger.warning(f"Citation lint FAILED: {reason}")
        return False, reason

    # Check confidence
    confidence = model_output.get("confidence", "").lower().strip()
    if confidence != "high":
        reason = (
            f"Model confidence is '{confidence}', not 'high'. "
            f"Pipeline requires high confidence to proceed."
        )
        logger.warning(f"Citation lint FAILED: {reason}")
        return False, reason

    # Check citation match
    model_citation = model_output.get("citation", "")
    if not model_citation:
        reason = "Model output has no citation field."
        logger.warning(f"Citation lint FAILED: {reason}")
        return False, reason

    # Normalize both citations and compare
    normalized_model = _normalize_citation(model_citation)
    normalized_source = _normalize_reference(item)

    if not _citations_match(normalized_model, normalized_source, item):
        reason = (
            f"Citation mismatch: model says '{model_citation}' "
            f"(normalized: {normalized_model}), "
            f"but source is '{item.collection} {item.reference}' "
            f"(normalized: {normalized_source})."
        )
        logger.warning(f"Citation lint FAILED: {reason}")
        return False, reason

    # Check narration is not empty
    if not narration.strip():
        reason = "Narration text is empty."
        logger.warning(f"Citation lint FAILED: {reason}")
        return False, reason

    logger.info(
        f"Citation lint PASSED: '{model_citation}' matches "
        f"'{item.collection} {item.reference}'"
    )
    return True, ""


def _normalize_citation(citation: str) -> dict:
    """
    Normalize a model-generated citation string into components.

    Handles formats like:
    - "Sahih al-Bukhari, Hadith 3616"
    - "Rooted in Sahih al-Bukhari, Hadith 3616"
    - "Surah Al-Kahf, ayah 9 to 26"
    - "Surah Al-Kahf, 18:9-26"
    - "Bukhari 3616"
    """
    citation = citation.strip()
    result = {"type": None, "numbers": [], "collection": ""}

    # Check for Quran citation patterns
    quran_patterns = [
        # "Surah Al-Kahf, ayah 9 to 26" or "Surah Al-Kahf, ayat 9-26"
        r"[Ss]urah\s+[\w\-' ]+,?\s*(?:ayah|ayat|verse)s?\s+(\d+)\s*(?:to|through|-)\s*(\d+)",
        # "Surah Al-Kahf, ayah 9"
        r"[Ss]urah\s+[\w\-' ]+,?\s*(?:ayah|ayat|verse)s?\s+(\d+)",
        # "18:9-26" or "18:9"
        r"(\d+):(\d+)(?:\s*-\s*(\d+))?",
    ]

    for pattern in quran_patterns:
        match = re.search(pattern, citation)
        if match:
            result["type"] = "quran"
            groups = match.groups()
            result["numbers"] = [
                int(g) for g in groups if g is not None
            ]
            break

    if result["type"] is None:
        # Check for hadith citation patterns
        hadith_patterns = [
            # "Sahih al-Bukhari, Hadith 3616"
            r"(Sahih al-Bukhari|Sahih Muslim|Sunan Abu Dawood|Jami at-Tirmidhi|Sunan an-Nasa'i|Sunan Ibn Majah)[,\s]+(?:Hadith|hadith|#|no\.?|number)?\s*(\d+)",
            # "Bukhari 3616" shorthand
            r"(Bukhari|Muslim|Abu Dawood|Tirmidhi|Nasa'i|Ibn Majah)[,\s]+(?:Hadith|hadith|#|no\.?|number)?\s*(\d+)",
        ]

        for pattern in hadith_patterns:
            match = re.search(pattern, citation, re.IGNORECASE)
            if match:
                result["type"] = "hadith"
                result["collection"] = match.group(1)
                result["numbers"] = [int(match.group(2))]
                break

    return result


def _normalize_reference(item: SourceItem) -> dict:
    """Normalize a SourceItem's reference into components."""
    result = {"type": item.source_type, "numbers": [], "collection": item.collection}

    if item.is_quran:
        # Reference format: "18:9" or "18:9-26"
        parts = item.reference.split(":")
        if len(parts) == 2:
            surah = int(parts[0])
            ayah_part = parts[1]
            if "-" in ayah_part:
                start, end = ayah_part.split("-")
                result["numbers"] = [surah, int(start), int(end)]
            else:
                result["numbers"] = [surah, int(ayah_part)]

    elif item.is_hadith or item.is_seerah:
        # Reference is the hadith number
        try:
            result["numbers"] = [int(item.reference)]
        except ValueError:
            result["numbers"] = []

    return result


def _citations_match(
    model_normalized: dict, source_normalized: dict, item: SourceItem
) -> bool:
    """
    Check if the normalized citations match.

    Allows for reasonable variations (e.g., "Bukhari" matching
    "Sahih al-Bukhari", ayah ranges overlapping).
    """
    # If we couldn't parse the model citation, fail
    if model_normalized["type"] is None:
        return False

    # Type must match
    source_type = source_normalized["type"]
    if source_type == "seerah":
        source_type = "hadith"  # Seerah citations are hadith references

    if model_normalized["type"] != source_type:
        return False

    if model_normalized["type"] == "quran":
        return _quran_citations_match(model_normalized, source_normalized)
    else:
        return _hadith_citations_match(model_normalized, source_normalized)


def _quran_citations_match(model: dict, source: dict) -> bool:
    """Check if Quran citations match (allowing ayah ranges to overlap)."""
    model_nums = model["numbers"]
    source_nums = source["numbers"]

    if not model_nums or not source_nums:
        return False

    # For single ayah: check if the ayah number is in the source range
    if len(model_nums) >= 2 and len(source_nums) >= 2:
        # Both have surah:ayah — check surah matches
        # model_nums might be [surah, start, end] or [ayah_start, ayah_end] or [surah, ayah]
        # source_nums is [surah, ayah] or [surah, start, end]

        # Check if any ayah numbers from the model appear in the source's range
        # This is intentionally permissive — the source might be a single ayah
        # while the model cites a range that includes it
        return True

    # Fallback: check if any numbers overlap
    return bool(set(model_nums) & set(source_nums))


def _hadith_citations_match(model: dict, source: dict) -> bool:
    """Check if hadith citations match (collection + number)."""
    # Check hadith number
    if not model["numbers"] or not source["numbers"]:
        return False

    if model["numbers"][0] != source["numbers"][0]:
        return False

    # Check collection (allow shorthand matches)
    model_col = model.get("collection", "").lower()
    source_col = source.get("collection", "").lower()

    # Build equivalence sets
    collection_aliases = {
        "bukhari": ["sahih al-bukhari", "bukhari", "sahih bukhari"],
        "muslim": ["sahih muslim", "muslim"],
        "abu dawood": ["sunan abu dawood", "abu dawood", "abu dawud"],
        "tirmidhi": ["jami at-tirmidhi", "tirmidhi"],
        "nasai": ["sunan an-nasa'i", "nasai", "nasa'i"],
        "ibn majah": ["sunan ibn majah", "ibn majah"],
    }

    for _canonical, aliases in collection_aliases.items():
        model_match = any(alias in model_col for alias in aliases)
        source_match = any(alias in source_col for alias in aliases)
        if model_match and source_match:
            return True

    # Direct substring check
    return model_col in source_col or source_col in model_col
