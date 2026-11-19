"""
Content validator.

Runs the full content_policy.yaml rule set against every SourceItem.
This is a programmatic gate — the LLM never decides if content is valid.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from src.sourcing.base_source import SourceItem

logger = logging.getLogger(__name__)

_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "content_policy.yaml"


class ContentValidator:
    """
    Validates SourceItems against the content policy configuration.

    Every check is deterministic and based on config — no LLM involvement.
    """

    def __init__(self, policy_path: Optional[Path] = None):
        policy_path = policy_path or _POLICY_PATH
        with open(policy_path, "r", encoding="utf-8") as f:
            self._policy = yaml.safe_load(f)

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
        self._reject_if_no_source = self._policy["exclusions"][
            "reject_if_no_source_match"
        ]

    def validate(self, item: SourceItem) -> tuple[bool, list[str]]:
        """
        Validate a SourceItem against all content policy rules.

        Args:
            item: The SourceItem to validate.

        Returns:
            Tuple of (is_valid, list_of_violation_reasons).
            If is_valid is True, the violations list is empty.
        """
        violations: list[str] = []

        # 1. Source type check
        if item.source_type not in ("quran", "hadith", "seerah"):
            violations.append(f"Unknown source type: {item.source_type}")

        # 2. Collection denylist
        if item.collection in self._denylist_collections:
            violations.append(
                f"Collection '{item.collection}' is in the denylist. "
                f"Denylisted: {self._denylist_collections}"
            )

        # 3. Collection allowlist (for hadith/seerah only)
        if item.source_type in ("hadith", "seerah"):
            if item.collection not in self._allowed_collections:
                violations.append(
                    f"Collection '{item.collection}' is not in the allowed list. "
                    f"Allowed: {self._allowed_collections}"
                )

        # 4. Grade filter (for hadith/seerah — Quran has no grade)
        if item.source_type in ("hadith", "seerah"):
            if item.grade is None:
                violations.append(
                    "Hadith/Seerah item has no grade — cannot verify authenticity."
                )
            elif item.grade not in self._allowed_grades:
                violations.append(
                    f"Grade '{item.grade}' is not allowed. "
                    f"Allowed grades: {self._allowed_grades}"
                )

        # 5. Topic denylist — keyword scan on all text fields
        combined_text = " ".join([
            item.translation_text or "",
            item.arabic_text or "",
            item.collection or "",
            str(item.metadata) if item.metadata else "",
        ]).lower()

        for denied_topic in self._denylist_topics:
            if denied_topic in combined_text:
                violations.append(
                    f"Content contains denied topic keyword: '{denied_topic}'"
                )

        # 6. Empty content check
        if not item.translation_text and not item.arabic_text:
            violations.append("Both translation_text and arabic_text are empty.")

        # 7. Missing reference
        if not item.reference:
            violations.append("Item has no reference (citation).")

        # 8. Quran-specific: must have collection == "Quran"
        if item.source_type == "quran" and item.collection != "Quran":
            violations.append(
                f"Quran item has unexpected collection: '{item.collection}'"
            )

        is_valid = len(violations) == 0

        if not is_valid:
            logger.warning(
                f"Content validation FAILED for {item.source_type} "
                f"ref={item.reference}: {violations}"
            )
        else:
            logger.info(
                f"Content validation PASSED for {item.source_type} "
                f"ref={item.reference}"
            )

        return is_valid, violations

    def validate_narration_text(self, narration: str) -> tuple[bool, list[str]]:
        """
        Validate the LLM-generated narration text against content policy.

        Checks for denied topic keywords in the narration itself.
        This is a secondary safety net after the sourcing-level filter.

        Args:
            narration: The generated narration text.

        Returns:
            Tuple of (is_valid, list_of_violation_reasons).
        """
        violations: list[str] = []
        narration_lower = narration.lower()

        # Check denied topics
        for denied_topic in self._denylist_topics:
            if denied_topic in narration_lower:
                violations.append(
                    f"Narration contains denied topic keyword: '{denied_topic}'"
                )

        # Check for physical descriptions of prophets/sahaba
        # (basic heuristic — the system prompt also prevents this)
        appearance_patterns = [
            r"\b(looked like|appeared as|had (?:a )?\w+ face|physical(?:ly)?)\b",
            r"\b(tall|short|fair|dark|handsome|beautiful) (?:man|woman|person)\b",
        ]
        for pattern in appearance_patterns:
            if re.search(pattern, narration_lower):
                violations.append(
                    f"Narration may contain physical description "
                    f"(matched pattern: {pattern})"
                )

        is_valid = len(violations) == 0

        if not is_valid:
            logger.warning(f"Narration validation FAILED: {violations}")

        return is_valid, violations
