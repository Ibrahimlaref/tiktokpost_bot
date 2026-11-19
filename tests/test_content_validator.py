"""
Tests for ContentValidator.

Verifies that:
- Denied topic keywords are detected and rejected
- Valid source items pass validation
- Collection allowlist/denylist are enforced
- Grade filtering works for hadith
- Narration text validation catches denied content
"""

import pytest

from src.sourcing.base_source import SourceItem
from src.verification.content_validator import ContentValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_quran_item(
    reference: str = "18:9",
    translation_text: str = "Or do you think that the companions of the cave...",
) -> SourceItem:
    return SourceItem(
        source_type="quran",
        collection="Quran",
        reference=reference,
        grade=None,
        arabic_text="أَمْ حَسِبْتَ أَنَّ أَصْحَابَ الْكَهْفِ",
        translation_text=translation_text,
        translator="Sahih International",
        source_url=f"https://quran.com/{reference}",
        metadata={"surah_name": "Al-Kahf"},
    )


def make_hadith_item(
    collection: str = "Sahih al-Bukhari",
    reference: str = "3616",
    grade: str = "sahih",
    translation_text: str = "The Prophet (ﷺ) said...",
) -> SourceItem:
    return SourceItem(
        source_type="hadith",
        collection=collection,
        reference=reference,
        grade=grade,
        arabic_text="قال النبي ﷺ",
        translation_text=translation_text,
        translator="test",
        source_url=f"https://sunnah.com/bukhari/{reference}",
    )


# ---------------------------------------------------------------------------
# Tests: Topic denylist
# ---------------------------------------------------------------------------

class TestTopicDenylist:

    def test_succession_dispute_detected(self):
        """Content mentioning 'succession dispute' must fail validation."""
        validator = ContentValidator()
        item = make_hadith_item(
            translation_text="The succession dispute among the companions..."
        )
        is_valid, violations = validator.validate(item)

        assert is_valid is False
        assert any("succession dispute" in v for v in violations)

    def test_imamate_detected(self):
        """Content mentioning 'imamate' must fail validation."""
        validator = ContentValidator()
        item = make_hadith_item(
            translation_text="The doctrine of imamate is central to..."
        )
        is_valid, violations = validator.validate(item)

        assert is_valid is False
        assert any("imamate" in v for v in violations)

    def test_political_party_detected(self):
        """Content mentioning 'political party' must fail validation."""
        validator = ContentValidator()
        item = make_hadith_item(
            translation_text="The political party announced their stance..."
        )
        is_valid, violations = validator.validate(item)

        assert is_valid is False
        assert any("political party" in v for v in violations)

    def test_current_events_detected(self):
        """Content mentioning 'current events' must fail validation."""
        validator = ContentValidator()
        item = make_hadith_item(
            translation_text="In light of current events, scholars say..."
        )
        is_valid, violations = validator.validate(item)

        assert is_valid is False

    def test_clean_content_passes(self):
        """Normal Islamic content without denied keywords should pass."""
        validator = ContentValidator()
        item = make_hadith_item(
            translation_text=(
                "The Prophet (ﷺ) said: 'The best of you are those "
                "who learn the Quran and teach it.'"
            )
        )
        is_valid, violations = validator.validate(item)

        assert is_valid is True
        assert violations == []


# ---------------------------------------------------------------------------
# Tests: Collection allowlist/denylist
# ---------------------------------------------------------------------------

class TestCollectionValidation:

    def test_allowed_collection_passes(self):
        validator = ContentValidator()
        item = make_hadith_item(collection="Sahih al-Bukhari")
        is_valid, _ = validator.validate(item)
        assert is_valid is True

    def test_denylisted_collection_fails(self):
        validator = ContentValidator()
        item = make_hadith_item(collection="Al-Kafi")
        is_valid, violations = validator.validate(item)
        assert is_valid is False
        assert any("denylist" in v.lower() for v in violations)

    def test_unknown_collection_fails(self):
        validator = ContentValidator()
        item = make_hadith_item(collection="Random Book")
        is_valid, violations = validator.validate(item)
        assert is_valid is False
        assert any("not in the allowed list" in v for v in violations)


# ---------------------------------------------------------------------------
# Tests: Grade validation
# ---------------------------------------------------------------------------

class TestGradeValidation:

    def test_sahih_grade_passes(self):
        validator = ContentValidator()
        item = make_hadith_item(grade="sahih")
        is_valid, _ = validator.validate(item)
        assert is_valid is True

    def test_hasan_grade_passes(self):
        validator = ContentValidator()
        item = make_hadith_item(grade="hasan")
        is_valid, _ = validator.validate(item)
        assert is_valid is True

    def test_daif_grade_fails(self):
        validator = ContentValidator()
        item = make_hadith_item(grade="da'if")
        is_valid, violations = validator.validate(item)
        assert is_valid is False
        assert any("not allowed" in v.lower() for v in violations)

    def test_no_grade_fails_for_hadith(self):
        validator = ContentValidator()
        item = make_hadith_item(grade=None)
        is_valid, violations = validator.validate(item)
        assert is_valid is False
        assert any("no grade" in v.lower() for v in violations)


# ---------------------------------------------------------------------------
# Tests: Quran-specific
# ---------------------------------------------------------------------------

class TestQuranValidation:

    def test_valid_quran_item_passes(self):
        validator = ContentValidator()
        item = make_quran_item()
        is_valid, violations = validator.validate(item)
        assert is_valid is True
        assert violations == []

    def test_quran_no_grade_is_fine(self):
        """Quran items don't have grades — None is acceptable."""
        validator = ContentValidator()
        item = make_quran_item()
        assert item.grade is None
        is_valid, _ = validator.validate(item)
        assert is_valid is True

    def test_quran_wrong_collection_fails(self):
        """Quran items must have collection='Quran'."""
        validator = ContentValidator()
        item = SourceItem(
            source_type="quran",
            collection="Some Other Book",
            reference="18:9",
            grade=None,
            arabic_text="test",
            translation_text="test",
            translator="test",
            source_url="test",
        )
        is_valid, violations = validator.validate(item)
        assert is_valid is False


# ---------------------------------------------------------------------------
# Tests: Narration text validation
# ---------------------------------------------------------------------------

class TestNarrationValidation:

    def test_clean_narration_passes(self):
        validator = ContentValidator()
        narration = (
            "In the Quran, Allah tells us the story of the People of the Cave. "
            "A group of young believers fled from persecution."
        )
        is_valid, violations = validator.validate_narration_text(narration)
        assert is_valid is True
        assert violations == []

    def test_narration_with_denied_topic_fails(self):
        validator = ContentValidator()
        narration = (
            "The succession dispute after the Prophet's passing "
            "led to significant political changes."
        )
        is_valid, violations = validator.validate_narration_text(narration)
        assert is_valid is False

    def test_empty_content_fails(self):
        """Items with no text content should fail."""
        validator = ContentValidator()
        item = SourceItem(
            source_type="hadith",
            collection="Sahih al-Bukhari",
            reference="1",
            grade="sahih",
            arabic_text="",
            translation_text="",
            translator="test",
            source_url="test",
        )
        is_valid, violations = validator.validate(item)
        assert is_valid is False
        assert any("empty" in v.lower() for v in violations)
