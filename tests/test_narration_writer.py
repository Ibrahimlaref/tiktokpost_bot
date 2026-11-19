"""
Tests for NarrationWriter.

Verifies that:
- The system prompt is present and correct
- NO_RELIABLE_SOURCE triggers a hard failure
- Low confidence triggers a hard failure
- User message is built correctly from SourceItem
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.sourcing.base_source import SourceItem
from src.scripting.narration_writer import (
    NARRATION_SYSTEM_PROMPT,
    NarrationFailure,
    NarrationWriter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_source_item() -> SourceItem:
    return SourceItem(
        source_type="hadith",
        collection="Sahih al-Bukhari",
        reference="3616",
        grade="sahih",
        arabic_text="قال النبي ﷺ: إن الدين يسر",
        translation_text=(
            "The Prophet (ﷺ) said: 'Religion is easy, and no one "
            "overburdens himself in religion but it overwhelms him.'"
        ),
        translator="Sahih International",
        source_url="https://sunnah.com/bukhari/3616",
        metadata={"book_name": "Sahih al-Bukhari"},
    )


# ---------------------------------------------------------------------------
# Tests: System prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:

    def test_system_prompt_contains_key_instructions(self):
        """The system prompt must contain all critical instructions."""
        assert "NO_RELIABLE_SOURCE" in NARRATION_SYSTEM_PROMPT
        assert "60-90 second" in NARRATION_SYSTEM_PROMPT
        assert "150-220 words" in NARRATION_SYSTEM_PROMPT
        assert "confidence" in NARRATION_SYSTEM_PROMPT
        assert "political party" in NARRATION_SYSTEM_PROMPT
        assert "physical appearance" in NARRATION_SYSTEM_PROMPT

    def test_system_prompt_requires_json_output(self):
        """Output format must be JSON."""
        assert '"narration"' in NARRATION_SYSTEM_PROMPT
        assert '"citation"' in NARRATION_SYSTEM_PROMPT
        assert '"confidence"' in NARRATION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tests: Hard failures
# ---------------------------------------------------------------------------

class TestNarrationHardFailures:

    @pytest.mark.asyncio
    async def test_no_reliable_source_raises(self):
        """NO_RELIABLE_SOURCE must raise NarrationFailure, not retry."""
        writer = NarrationWriter.__new__(NarrationWriter)
        writer._provider = "openai"
        writer._model = "test"

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "narration": "NO_RELIABLE_SOURCE",
            "citation": "",
            "confidence": "low",
        })

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response
        writer._llm = mock_llm

        with pytest.raises(NarrationFailure) as exc_info:
            await writer.generate_narration(make_source_item())

        assert "NO_RELIABLE_SOURCE" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_low_confidence_raises(self):
        """Low confidence must raise NarrationFailure, not retry."""
        writer = NarrationWriter.__new__(NarrationWriter)
        writer._provider = "openai"
        writer._model = "test"

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "narration": "Some narration text here...",
            "citation": "Sahih al-Bukhari, Hadith 3616",
            "confidence": "low",
        })

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response
        writer._llm = mock_llm

        with pytest.raises(NarrationFailure) as exc_info:
            await writer.generate_narration(make_source_item())

        assert "confidence" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_valid_response_succeeds(self):
        """A valid high-confidence response should return successfully."""
        writer = NarrationWriter.__new__(NarrationWriter)
        writer._provider = "openai"
        writer._model = "test"

        expected = {
            "narration": (
                "The Prophet, peace be upon him, taught us that religion "
                "is meant to be easy. He said that no one overburdens "
                "himself in religion but it overwhelms him."
            ),
            "citation": "Sahih al-Bukhari, Hadith 3616",
            "confidence": "high",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(expected)

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response
        writer._llm = mock_llm

        result = await writer.generate_narration(make_source_item())

        assert result["narration"] == expected["narration"]
        assert result["citation"] == expected["citation"]
        assert result["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_missing_citation_raises(self):
        """Missing citation in LLM output should raise NarrationFailure."""
        writer = NarrationWriter.__new__(NarrationWriter)
        writer._provider = "openai"
        writer._model = "test"

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "narration": "Some narration...",
            "citation": "",
            "confidence": "high",
        })

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response
        writer._llm = mock_llm

        with pytest.raises(NarrationFailure):
            await writer.generate_narration(make_source_item())


# ---------------------------------------------------------------------------
# Tests: User message building
# ---------------------------------------------------------------------------

class TestUserMessageBuilding:

    def test_user_message_contains_source_data(self):
        """The user message must contain all SourceItem fields."""
        writer = NarrationWriter.__new__(NarrationWriter)
        writer._provider = "openai"
        writer._model = "test"

        item = make_source_item()
        message = writer._build_user_message(item)

        assert "hadith" in message.lower()
        assert "Sahih al-Bukhari" in message
        assert "3616" in message
        assert "sahih" in message.lower()
        assert item.translation_text in message
        assert item.arabic_text in message

    def test_user_message_includes_metadata(self):
        """Relevant metadata should appear in the user message."""
        writer = NarrationWriter.__new__(NarrationWriter)
        writer._provider = "openai"
        writer._model = "test"

        item = make_source_item()
        message = writer._build_user_message(item)

        assert "book_name" in message


# ---------------------------------------------------------------------------
# Tests: JSON extraction fallback
# ---------------------------------------------------------------------------

class TestJsonExtraction:

    def test_extract_json_from_markdown(self):
        """Should extract JSON from markdown-wrapped response."""
        writer = NarrationWriter.__new__(NarrationWriter)

        text = '''Here is the output:
```json
{"narration": "test", "citation": "test", "confidence": "high"}
```'''

        result = writer._extract_json(text)
        assert result is not None
        assert result["narration"] == "test"

    def test_extract_json_returns_none_for_garbage(self):
        """Should return None for non-JSON text."""
        writer = NarrationWriter.__new__(NarrationWriter)
        result = writer._extract_json("This is just plain text.")
        assert result is None
