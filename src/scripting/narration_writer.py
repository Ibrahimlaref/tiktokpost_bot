"""
Narration writer.

Uses a strong hosted LLM (GPT-4-class or Claude) to write 60-90 second
narration scripts from verified SourceItems. The system prompt is from
the spec — it constrains the model to ONLY use content present in the
SourceItem and output structured JSON.

Hard-fails on NO_RELIABLE_SOURCE or confidence: "low" — logs, does not
retry-and-hope.
"""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from src.sourcing.base_source import SourceItem

load_dotenv()
logger = logging.getLogger(__name__)

# System prompt from spec §5 — pasted verbatim
NARRATION_SYSTEM_PROMPT = """You are the narration writer for a Sunni Quran/Seerah storytelling channel.
You will receive exactly one verified SourceItem. Your job:

1. Write a 60-90 second spoken narration (~150-220 words) using ONLY the
   content present in the SourceItem. Add no plot details, dialogue, or
   facts that aren't in the source text or its immediate context.
2. End with a spoken citation, e.g. "Rooted in Sahih al-Bukhari, Hadith 3616"
   or "Surah Al-Kahf, ayah 9 to 26."
3. Never mention a modern political party, movement, or current event.
4. Never describe a Prophet's or Sahabi's physical appearance in a way that
   could be used to generate a likeness of them.
5. If the SourceItem is empty, ambiguous, or you're not fully confident the
   narration stays inside it, output exactly: NO_RELIABLE_SOURCE.
   Do not fill the gap from general knowledge.

Output JSON: {"narration": str, "citation": str, "confidence": "high"|"low"}"""


class NarrationWriter:
    """
    Generates narration scripts from verified SourceItems using an LLM.

    The model is constrained to use ONLY the content in the SourceItem.
    Hard-fails on any sign of uncertainty rather than generating unreliable content.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the narration writer.

        Args:
            provider: "openai" or "anthropic". Defaults to LLM_PROVIDER env var.
            model: Model name. Defaults to LLM_MODEL env var.
        """
        self._provider = provider or os.getenv("LLM_PROVIDER", "openai")
        self._model = model or os.getenv("LLM_MODEL", "gpt-4o")
        self._llm = self._init_llm()

    def _init_llm(self):
        """Initialize the LLM client based on provider."""
        if self._provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self._model,
                temperature=0.3,  # Low temperature for factual narration
                max_tokens=1000,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        elif self._provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self._model,
                temperature=0.3,
                max_tokens=1000,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self._provider}")

    async def generate_narration(self, item: SourceItem) -> dict:
        """
        Generate a narration script from a verified SourceItem.

        Args:
            item: A verified SourceItem (Quran, Hadith, or Seerah).

        Returns:
            Dict with keys: "narration", "citation", "confidence".

        Raises:
            NarrationFailure: If the model returns NO_RELIABLE_SOURCE
                or confidence is "low".
        """
        # Build the user message with all source context
        user_message = self._build_user_message(item)

        logger.info(
            f"Generating narration for {item.source_type} "
            f"ref={item.reference} ({item.collection})"
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=NARRATION_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]

            response = await self._llm.ainvoke(messages)
            raw_output = response.content

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise NarrationFailure(
                f"LLM call failed: {e}",
                source_ref=item.reference,
            )

        # Parse the JSON output
        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            result = self._extract_json(raw_output)
            if result is None:
                logger.error(f"Failed to parse LLM output as JSON: {raw_output[:200]}")
                raise NarrationFailure(
                    "LLM output is not valid JSON.",
                    source_ref=item.reference,
                    raw_output=raw_output,
                )

        # Validate required fields
        if "narration" not in result:
            raise NarrationFailure(
                "LLM output missing 'narration' field.",
                source_ref=item.reference,
                raw_output=raw_output,
            )

        # Hard-fail on NO_RELIABLE_SOURCE
        narration = result.get("narration", "")
        if narration.strip() == "NO_RELIABLE_SOURCE":
            logger.warning(
                f"Model returned NO_RELIABLE_SOURCE for {item.reference}. "
                f"Pipeline stops here — will NOT retry."
            )
            raise NarrationFailure(
                "Model could not write a reliable narration from the source material. "
                "NO_RELIABLE_SOURCE — this is a hard stop, not a retry-able error.",
                source_ref=item.reference,
            )

        # Hard-fail on low confidence
        confidence = result.get("confidence", "low").lower().strip()
        if confidence != "high":
            logger.warning(
                f"Model confidence is '{confidence}' for {item.reference}. "
                f"Pipeline stops — will NOT retry."
            )
            raise NarrationFailure(
                f"Model confidence is '{confidence}', not 'high'. "
                f"This is a hard stop to prevent unreliable narrations.",
                source_ref=item.reference,
            )

        # Ensure citation is present
        if not result.get("citation", "").strip():
            raise NarrationFailure(
                "Model output has no citation.",
                source_ref=item.reference,
            )

        logger.info(
            f"Narration generated successfully for {item.reference}: "
            f"{len(narration.split())} words, confidence={confidence}"
        )

        return {
            "narration": narration,
            "citation": result["citation"],
            "confidence": confidence,
        }

    def _build_user_message(self, item: SourceItem) -> str:
        """Build the user message containing the SourceItem data."""
        parts = [
            f"Source Type: {item.source_type}",
            f"Collection: {item.collection}",
            f"Reference: {item.reference}",
        ]

        if item.grade:
            parts.append(f"Grade: {item.grade}")

        if item.arabic_text:
            parts.append(f"\nArabic Text:\n{item.arabic_text}")

        if item.translation_text:
            parts.append(f"\nTranslation ({item.translator}):\n{item.translation_text}")

        if item.metadata:
            # Add relevant metadata (surah name, chapter info, etc.)
            relevant_keys = [
                "surah_name", "surah_name_arabic", "book_name",
                "chapter", "seerah_episode",
            ]
            for key in relevant_keys:
                if key in item.metadata:
                    parts.append(f"{key}: {item.metadata[key]}")

        return "\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Try to extract JSON from a text response that may contain other content."""
        # Look for JSON between curly braces
        import re

        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None


class NarrationFailure(Exception):
    """
    Raised when narration generation fails.

    This is a HARD failure — the pipeline should log it and move on,
    not retry. Retrying a NO_RELIABLE_SOURCE won't make the source
    more reliable.
    """

    def __init__(
        self,
        message: str,
        source_ref: str = "",
        raw_output: str = "",
    ):
        super().__init__(message)
        self.source_ref = source_ref
        self.raw_output = raw_output
