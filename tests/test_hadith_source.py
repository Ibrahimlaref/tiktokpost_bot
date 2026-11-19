"""
Tests for HadithSourceProvider.

Verifies that:
- Da'if (weak) hadith are programmatically rejected
- Denylisted collections never reach the caller
- Sahih hadith pass through the filter
- Grade extraction works correctly
"""

import pytest

from src.sourcing.base_source import SourceItem
from src.sourcing.hadith_source import HadithSourceProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_hadith_item(
    collection: str = "Sahih al-Bukhari",
    reference: str = "3616",
    grade: str = "sahih",
    translation_text: str = "The Prophet (ﷺ) said...",
    arabic_text: str = "قال النبي ﷺ...",
) -> SourceItem:
    """Create a test SourceItem for hadith."""
    return SourceItem(
        source_type="hadith",
        collection=collection,
        reference=reference,
        grade=grade,
        arabic_text=arabic_text,
        translation_text=translation_text,
        translator="test",
        source_url=f"https://sunnah.com/bukhari/{reference}",
        metadata={"api_source": "test"},
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestHadithGradeFilter:
    """Tests for programmatic grade filtering."""

    def test_daif_hadith_is_rejected(self):
        """A da'if (weak) graded hadith must be rejected by the filter."""
        provider = HadithSourceProvider()
        item = make_hadith_item(grade="da'if")

        assert provider._passes_filters(item) is False

    def test_mawdu_hadith_is_rejected(self):
        """A mawdu' (fabricated) hadith must be rejected."""
        provider = HadithSourceProvider()
        item = make_hadith_item(grade="mawdu'")

        assert provider._passes_filters(item) is False

    def test_sahih_hadith_passes(self):
        """A sahih hadith from an allowed collection must pass."""
        provider = HadithSourceProvider()
        item = make_hadith_item(grade="sahih", collection="Sahih al-Bukhari")

        assert provider._passes_filters(item) is True

    def test_hasan_hadith_passes(self):
        """A hasan hadith from an allowed collection must pass."""
        provider = HadithSourceProvider()
        item = make_hadith_item(grade="hasan", collection="Jami at-Tirmidhi")

        assert provider._passes_filters(item) is True

    def test_no_grade_is_rejected(self):
        """A hadith with no grade must be rejected (can't verify authenticity)."""
        provider = HadithSourceProvider()
        item = make_hadith_item(grade=None)

        # grade is None, so it won't be in allowed_grades
        assert provider._passes_filters(item) is False

    def test_case_insensitive_grade(self):
        """Grade matching should be case-insensitive."""
        provider = HadithSourceProvider()
        item = make_hadith_item(grade="Sahih")  # Capital S

        assert provider._passes_filters(item) is True


class TestCollectionDenylist:
    """Tests for collection denylist enforcement."""

    def test_al_kafi_is_rejected(self):
        """Al-Kafi (Shia collection) must be rejected."""
        provider = HadithSourceProvider()
        item = make_hadith_item(collection="Al-Kafi", grade="sahih")

        assert provider._passes_filters(item) is False

    def test_bihar_al_anwar_is_rejected(self):
        """Bihar al-Anwar must be rejected."""
        provider = HadithSourceProvider()
        item = make_hadith_item(collection="Bihar al-Anwar", grade="sahih")

        assert provider._passes_filters(item) is False

    def test_nahj_al_balagha_is_rejected(self):
        """Nahj al-Balagha must be rejected."""
        provider = HadithSourceProvider()
        item = make_hadith_item(collection="Nahj al-Balagha", grade="sahih")

        assert provider._passes_filters(item) is False

    def test_unknown_collection_is_rejected(self):
        """A collection not in the allowlist must be rejected."""
        provider = HadithSourceProvider()
        item = make_hadith_item(collection="Unknown Collection", grade="sahih")

        assert provider._passes_filters(item) is False


class TestTopicDenylist:
    """Tests for denied topic keyword detection."""

    def test_political_content_rejected(self):
        """Content containing political keywords must be rejected."""
        provider = HadithSourceProvider()
        item = make_hadith_item(
            translation_text="The political party gathered to discuss..."
        )

        assert provider._passes_filters(item) is False

    def test_succession_dispute_rejected(self):
        """Content about succession disputes must be rejected."""
        provider = HadithSourceProvider()
        item = make_hadith_item(
            translation_text="The succession dispute after the Prophet..."
        )

        assert provider._passes_filters(item) is False

    def test_clean_content_passes(self):
        """Normal Islamic content without denied keywords should pass."""
        provider = HadithSourceProvider()
        item = make_hadith_item(
            translation_text=(
                "The Prophet (ﷺ) said: 'The best of you are those "
                "who learn the Quran and teach it.'"
            )
        )

        assert provider._passes_filters(item) is True


class TestGradeExtraction:
    """Tests for grade extraction from API responses."""

    def test_extract_sahih_grade(self):
        """Should extract 'sahih' when present in grades list."""
        grades = [
            {"name": "Al-Albani", "grade": "Sahih"},
            {"name": "Darussalam", "grade": "Sahih"},
        ]
        result = HadithSourceProvider._extract_best_grade(grades)
        assert result == "sahih"

    def test_extract_hasan_grade(self):
        """Should extract 'hasan' when no sahih is present."""
        grades = [
            {"name": "Al-Albani", "grade": "Hasan"},
        ]
        result = HadithSourceProvider._extract_best_grade(grades)
        assert result == "hasan"

    def test_extract_daif_grade(self):
        """Should return da'if when only weak grades present."""
        grades = [
            {"name": "Al-Albani", "grade": "Da'if"},
        ]
        result = HadithSourceProvider._extract_best_grade(grades)
        assert result == "da'if"

    def test_empty_grades_returns_none(self):
        """Should return None for empty grades list."""
        result = HadithSourceProvider._extract_best_grade([])
        assert result is None

    def test_sahih_takes_priority_over_daif(self):
        """If any grader says sahih, it should be accepted as sahih."""
        grades = [
            {"name": "Scholar A", "grade": "Da'if"},
            {"name": "Scholar B", "grade": "Sahih"},
        ]
        result = HadithSourceProvider._extract_best_grade(grades)
        assert result == "sahih"


class TestCollectionResolution:
    """Tests for collection name resolution."""

    def test_resolve_bukhari_shorthand(self):
        provider = HadithSourceProvider()
        assert provider._resolve_collection_name("bukhari") == "Sahih al-Bukhari"

    def test_resolve_full_name(self):
        provider = HadithSourceProvider()
        assert provider._resolve_collection_name("Sahih al-Bukhari") == "Sahih al-Bukhari"

    def test_resolve_unknown_returns_none(self):
        provider = HadithSourceProvider()
        assert provider._resolve_collection_name("unknown") is None
