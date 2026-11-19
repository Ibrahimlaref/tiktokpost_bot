"""
Tests for citation_linter.

Verifies that:
- Matching citations pass
- Mismatched citations fail
- NO_RELIABLE_SOURCE fails
- Low confidence fails
- Various citation formats are handled correctly
"""

import pytest

from src.sourcing.base_source import SourceItem
from src.verification.citation_linter import (
    _normalize_citation,
    _normalize_reference,
    citation_matches_source,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_quran_item(reference: str = "18:9") -> SourceItem:
    return SourceItem(
        source_type="quran",
        collection="Quran",
        reference=reference,
        grade=None,
        arabic_text="test",
        translation_text="test",
        translator="Sahih International",
        source_url=f"https://quran.com/{reference}",
        metadata={"surah_name": "Al-Kahf"},
    )


def make_hadith_item(
    collection: str = "Sahih al-Bukhari",
    reference: str = "3616",
) -> SourceItem:
    return SourceItem(
        source_type="hadith",
        collection=collection,
        reference=reference,
        grade="sahih",
        arabic_text="test",
        translation_text="test",
        translator="test",
        source_url=f"https://sunnah.com/bukhari/{reference}",
    )


# ---------------------------------------------------------------------------
# Tests: Hard gate conditions
# ---------------------------------------------------------------------------

class TestHardGateConditions:

    def test_no_reliable_source_fails(self):
        """NO_RELIABLE_SOURCE must be a hard failure."""
        model_output = {
            "narration": "NO_RELIABLE_SOURCE",
            "citation": "",
            "confidence": "low",
        }
        item = make_hadith_item()

        passes, reason = citation_matches_source(model_output, item)

        assert passes is False
        assert "NO_RELIABLE_SOURCE" in reason

    def test_low_confidence_fails(self):
        """Confidence 'low' must fail — pipeline requires 'high'."""
        model_output = {
            "narration": "A narration about the Prophet...",
            "citation": "Sahih al-Bukhari, Hadith 3616",
            "confidence": "low",
        }
        item = make_hadith_item()

        passes, reason = citation_matches_source(model_output, item)

        assert passes is False
        assert "confidence" in reason.lower()

    def test_empty_narration_fails(self):
        """Empty narration text must fail."""
        model_output = {
            "narration": "",
            "citation": "Sahih al-Bukhari, Hadith 3616",
            "confidence": "high",
        }
        item = make_hadith_item()

        passes, reason = citation_matches_source(model_output, item)

        assert passes is False
        assert "empty" in reason.lower()

    def test_missing_citation_fails(self):
        """Missing citation field must fail."""
        model_output = {
            "narration": "A narration...",
            "citation": "",
            "confidence": "high",
        }
        item = make_hadith_item()

        passes, reason = citation_matches_source(model_output, item)

        assert passes is False


# ---------------------------------------------------------------------------
# Tests: Citation matching — hadith
# ---------------------------------------------------------------------------

class TestHadithCitationMatching:

    def test_exact_hadith_citation_passes(self):
        """Exact citation match should pass."""
        model_output = {
            "narration": "The Prophet (ﷺ) said...",
            "citation": "Sahih al-Bukhari, Hadith 3616",
            "confidence": "high",
        }
        item = make_hadith_item(collection="Sahih al-Bukhari", reference="3616")

        passes, reason = citation_matches_source(model_output, item)

        assert passes is True
        assert reason == ""

    def test_shorthand_collection_passes(self):
        """'Bukhari 3616' should match 'Sahih al-Bukhari' reference 3616."""
        model_output = {
            "narration": "The Prophet (ﷺ) said...",
            "citation": "Bukhari 3616",
            "confidence": "high",
        }
        item = make_hadith_item(collection="Sahih al-Bukhari", reference="3616")

        passes, reason = citation_matches_source(model_output, item)

        assert passes is True

    def test_rooted_in_prefix_passes(self):
        """'Rooted in Sahih al-Bukhari, Hadith 3616' should still match."""
        model_output = {
            "narration": "The Prophet (ﷺ) said...",
            "citation": "Rooted in Sahih al-Bukhari, Hadith 3616",
            "confidence": "high",
        }
        item = make_hadith_item(collection="Sahih al-Bukhari", reference="3616")

        passes, reason = citation_matches_source(model_output, item)

        assert passes is True

    def test_wrong_hadith_number_fails(self):
        """Different hadith number must fail."""
        model_output = {
            "narration": "The Prophet (ﷺ) said...",
            "citation": "Sahih al-Bukhari, Hadith 9999",
            "confidence": "high",
        }
        item = make_hadith_item(collection="Sahih al-Bukhari", reference="3616")

        passes, reason = citation_matches_source(model_output, item)

        assert passes is False
        assert "mismatch" in reason.lower()

    def test_wrong_collection_fails(self):
        """Different collection must fail."""
        model_output = {
            "narration": "The Prophet (ﷺ) said...",
            "citation": "Sahih Muslim, Hadith 3616",
            "confidence": "high",
        }
        item = make_hadith_item(collection="Sahih al-Bukhari", reference="3616")

        passes, reason = citation_matches_source(model_output, item)

        assert passes is False


# ---------------------------------------------------------------------------
# Tests: Citation matching — Quran
# ---------------------------------------------------------------------------

class TestQuranCitationMatching:

    def test_surah_ayah_citation_passes(self):
        """'Surah Al-Kahf, ayah 9 to 26' should match reference '18:9'."""
        model_output = {
            "narration": "In the Quran, Allah tells us...",
            "citation": "Surah Al-Kahf, ayah 9 to 26",
            "confidence": "high",
        }
        item = make_quran_item(reference="18:9")

        passes, reason = citation_matches_source(model_output, item)

        assert passes is True

    def test_verse_key_citation_passes(self):
        """'18:9' format should match."""
        model_output = {
            "narration": "In the Quran...",
            "citation": "Quran 18:9-26",
            "confidence": "high",
        }
        item = make_quran_item(reference="18:9")

        passes, reason = citation_matches_source(model_output, item)

        assert passes is True


# ---------------------------------------------------------------------------
# Tests: Citation normalization
# ---------------------------------------------------------------------------

class TestCitationNormalization:

    def test_normalize_hadith_citation(self):
        result = _normalize_citation("Sahih al-Bukhari, Hadith 3616")
        assert result["type"] == "hadith"
        assert 3616 in result["numbers"]

    def test_normalize_quran_ayah_range(self):
        result = _normalize_citation("Surah Al-Kahf, ayah 9 to 26")
        assert result["type"] == "quran"
        assert 9 in result["numbers"]
        assert 26 in result["numbers"]

    def test_normalize_quran_verse_key(self):
        result = _normalize_citation("18:9-26")
        assert result["type"] == "quran"
        assert result["numbers"] == [18, 9, 26]

    def test_normalize_shorthand_bukhari(self):
        result = _normalize_citation("Bukhari 3616")
        assert result["type"] == "hadith"
        assert 3616 in result["numbers"]

    def test_normalize_unparseable_returns_none_type(self):
        result = _normalize_citation("Some random text with no citation")
        assert result["type"] is None
