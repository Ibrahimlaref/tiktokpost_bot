"""
Main pipeline orchestrator.

Wires the full LangGraph graph end-to-end:
  source → validate → narrate → lint citation → generate media → assemble video
  → human review → publish

Each node is a discrete step with conditional edges:
- Validation failure → log + stop
- Citation mismatch → log + stop
- Review rejection → log + stop
- CAPTCHA detected → log + skip publishing
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

import yaml
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.media.tts_engine import TTSEngine
from src.media.video_assembler import VideoAssembler
from src.media.visual_generator import VisualGenerator
from src.publishing.tiktok_api_publisher import TikTokAPIPublisher, PublishError
from src.publishing.tiktok_browser_publisher import (
    BrowserPublishError,
    TikTokBrowserPublisher,
)
from src.review.human_review_queue import HumanReviewQueue
from src.scripting.narration_writer import NarrationFailure, NarrationWriter
from src.sourcing.base_source import SourceItem
from src.sourcing.hadith_source import HadithSourceProvider
from src.sourcing.quran_source import QuranSourceProvider
from src.sourcing.seerah_source import SeerahSourceProvider
from src.verification.citation_linter import citation_matches_source
from src.verification.content_validator import ContentValidator

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

_POLICY_PATH = Path("config/content_policy.yaml")


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """State passed through the LangGraph pipeline."""

    # Input
    topic: str
    source_type: str                    # "quran" | "hadith" | "seerah"

    # Sourcing
    source_item: Optional[dict]         # Serialized SourceItem
    source_items: list[dict]            # All candidates

    # Narration
    narration: Optional[dict]           # {"narration": str, "citation": str, "confidence": str}

    # Verification
    validation_passed: bool
    validation_errors: list[str]
    citation_passed: bool
    citation_error: str

    # Media
    audio_path: Optional[str]
    visual_path: Optional[str]
    video_path: Optional[str]

    # Review
    review_id: Optional[int]
    review_status: str                  # "pending" | "approved" | "rejected"

    # Publishing
    publish_id: Optional[str]
    publish_status: str

    # Pipeline control
    error: Optional[str]
    stage: str                          # Current stage name
    dry_run: bool


# ---------------------------------------------------------------------------
# Helper: SourceItem serialization
# ---------------------------------------------------------------------------

def _item_to_dict(item: SourceItem) -> dict:
    return {
        "source_type": item.source_type,
        "collection": item.collection,
        "reference": item.reference,
        "grade": item.grade,
        "arabic_text": item.arabic_text,
        "translation_text": item.translation_text,
        "translator": item.translator,
        "source_url": item.source_url,
        "metadata": item.metadata,
    }


def _dict_to_item(d: dict) -> SourceItem:
    return SourceItem(**d)


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------

async def source_content(state: PipelineState) -> PipelineState:
    """Node 1: Fetch source content based on topic and source type."""
    topic = state["topic"]
    source_type = state["source_type"]

    logger.info(f"📖 Sourcing content: topic='{topic}', type='{source_type}'")

    providers = {
        "quran": QuranSourceProvider,
        "hadith": HadithSourceProvider,
        "seerah": SeerahSourceProvider,
    }

    provider_cls = providers.get(source_type)
    if provider_cls is None:
        return {
            **state,
            "error": f"Unknown source type: {source_type}",
            "stage": "source_content",
        }

    try:
        async with provider_cls() as provider:
            items = await provider.fetch_candidates(topic=topic, limit=5)

            if not items:
                return {
                    **state,
                    "error": (
                        f"No source items found for topic '{topic}' "
                        f"(type={source_type}). Pipeline stops — "
                        f"will not fabricate content."
                    ),
                    "stage": "source_content",
                    "source_items": [],
                }

            # Use the first (best) item
            best_item = items[0]

            # Verify the item
            verified = await provider.verify(best_item)
            if not verified:
                return {
                    **state,
                    "error": (
                        f"Source verification failed for {best_item.reference}. "
                        f"Re-fetch did not confirm the item."
                    ),
                    "stage": "source_content",
                }

            logger.info(
                f"✅ Source found: {best_item.collection} {best_item.reference} "
                f"(grade={best_item.grade})"
            )

            return {
                **state,
                "source_item": _item_to_dict(best_item),
                "source_items": [_item_to_dict(i) for i in items],
                "stage": "source_content",
                "error": None,
            }

    except Exception as e:
        logger.error(f"Sourcing error: {e}")
        return {
            **state,
            "error": f"Sourcing failed: {e}",
            "stage": "source_content",
        }


async def validate_content(state: PipelineState) -> PipelineState:
    """Node 2: Run content validator against the source item."""
    if state.get("error"):
        return state

    source_dict = state.get("source_item")
    if not source_dict:
        return {**state, "error": "No source item to validate.", "stage": "validate"}

    item = _dict_to_item(source_dict)
    validator = ContentValidator()

    is_valid, violations = validator.validate(item)

    logger.info(
        f"{'✅' if is_valid else '❌'} Content validation: "
        f"{'PASSED' if is_valid else 'FAILED'}"
    )
    if violations:
        for v in violations:
            logger.warning(f"  Violation: {v}")

    return {
        **state,
        "validation_passed": is_valid,
        "validation_errors": violations,
        "stage": "validate",
        "error": None if is_valid else f"Validation failed: {'; '.join(violations)}",
    }


async def write_narration(state: PipelineState) -> PipelineState:
    """Node 3: Generate narration via LLM."""
    if state.get("error"):
        return state

    source_dict = state.get("source_item")
    if not source_dict:
        return {**state, "error": "No source item for narration.", "stage": "narrate"}

    item = _dict_to_item(source_dict)
    writer = NarrationWriter()

    try:
        result = await writer.generate_narration(item)

        logger.info(
            f"✅ Narration generated: {len(result['narration'].split())} words, "
            f"citation='{result['citation']}'"
        )

        return {
            **state,
            "narration": result,
            "stage": "narrate",
            "error": None,
        }

    except NarrationFailure as e:
        logger.error(f"❌ Narration HARD FAIL: {e}")
        return {
            **state,
            "error": f"Narration failed (hard stop): {e}",
            "stage": "narrate",
        }


async def lint_citation(state: PipelineState) -> PipelineState:
    """Node 4: Citation linter — hard gate."""
    if state.get("error"):
        return state

    narration = state.get("narration")
    source_dict = state.get("source_item")

    if not narration or not source_dict:
        return {
            **state,
            "error": "Missing narration or source for citation lint.",
            "stage": "lint_citation",
        }

    item = _dict_to_item(source_dict)
    passes, failure_reason = citation_matches_source(narration, item)

    # Also validate the narration text itself
    validator = ContentValidator()
    text_valid, text_violations = validator.validate_narration_text(
        narration.get("narration", "")
    )

    all_passed = passes and text_valid
    all_errors = []
    if not passes:
        all_errors.append(failure_reason)
    all_errors.extend(text_violations)

    logger.info(
        f"{'✅' if all_passed else '❌'} Citation lint: "
        f"{'PASSED' if all_passed else 'FAILED'}"
    )

    return {
        **state,
        "citation_passed": all_passed,
        "citation_error": "; ".join(all_errors) if all_errors else "",
        "stage": "lint_citation",
        "error": None if all_passed else f"Citation lint failed: {'; '.join(all_errors)}",
    }


async def generate_media(state: PipelineState) -> PipelineState:
    """Node 5: Generate TTS audio and visual background."""
    if state.get("error"):
        return state
    if state.get("dry_run"):
        logger.info("🏃 Dry run — skipping media generation.")
        return {**state, "stage": "generate_media"}

    narration = state.get("narration", {})
    source_dict = state.get("source_item", {})
    item = _dict_to_item(source_dict)

    # Load visual config
    with open(_POLICY_PATH, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    visual_config = policy.get("visuals", {})

    try:
        # Generate TTS audio
        async with TTSEngine() as tts:
            audio_path = await tts.synthesize_narration(narration["narration"])

        # Generate visual background
        visual_gen = VisualGenerator(
            style=visual_config.get("style", "calligraphy_and_nature"),
            depict_prophets=visual_config.get("depict_prophets", False),
            depict_sahaba_faces=visual_config.get("depict_sahaba_faces", False),
        )
        visual_path = await visual_gen.generate_background(item)

        logger.info(f"✅ Media generated: audio={audio_path}, visual={visual_path}")

        return {
            **state,
            "audio_path": str(audio_path),
            "visual_path": str(visual_path),
            "stage": "generate_media",
            "error": None,
        }

    except Exception as e:
        logger.error(f"❌ Media generation failed: {e}")
        return {
            **state,
            "error": f"Media generation failed: {e}",
            "stage": "generate_media",
        }


async def assemble_video(state: PipelineState) -> PipelineState:
    """Node 6: Assemble the final video."""
    if state.get("error"):
        return state
    if state.get("dry_run"):
        logger.info("🏃 Dry run — skipping video assembly.")
        return {**state, "stage": "assemble_video"}

    audio_path = state.get("audio_path")
    visual_path = state.get("visual_path")
    narration = state.get("narration", {})
    source_dict = state.get("source_item", {})

    if not audio_path or not visual_path:
        return {
            **state,
            "error": "Missing audio or visual paths for assembly.",
            "stage": "assemble_video",
        }

    item = _dict_to_item(source_dict)
    assembler = VideoAssembler()

    try:
        video_path = await assembler.assemble(
            background_path=Path(visual_path),
            audio_path=Path(audio_path),
            source_item=item,
            narration_text=narration.get("narration", ""),
            citation_text=narration.get("citation", item.citation_display),
        )

        logger.info(f"✅ Video assembled: {video_path}")

        return {
            **state,
            "video_path": str(video_path),
            "stage": "assemble_video",
            "error": None,
        }

    except Exception as e:
        logger.error(f"❌ Video assembly failed: {e}")
        return {
            **state,
            "error": f"Video assembly failed: {e}",
            "stage": "assemble_video",
        }


async def human_review(state: PipelineState) -> PipelineState:
    """Node 7: Submit to human review queue."""
    if state.get("error"):
        return state
    if state.get("dry_run"):
        logger.info("🏃 Dry run — skipping review.")
        return {**state, "stage": "human_review", "review_status": "dry_run"}

    # Check if review is required
    with open(_POLICY_PATH, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    review_required = policy.get("review", {}).get("human_approval_required", True)

    narration = state.get("narration", {})
    source_dict = state.get("source_item", {})
    video_path = state.get("video_path", "")

    queue = HumanReviewQueue()

    review_id = queue.submit_for_review(
        video_path=video_path,
        source_type=source_dict.get("source_type", ""),
        reference=source_dict.get("reference", ""),
        collection=source_dict.get("collection", ""),
        citation=narration.get("citation", ""),
        narration=narration.get("narration", ""),
        metadata={
            "grade": source_dict.get("grade"),
            "source_url": source_dict.get("source_url", ""),
        },
    )

    if review_required:
        logger.info(
            f"📋 Submitted to review queue: review_id={review_id}. "
            f"Run 'python -m src.review.human_review_queue' to approve/reject."
        )
        return {
            **state,
            "review_id": review_id,
            "review_status": "pending",
            "stage": "human_review",
            "error": None,
        }
    else:
        # Auto-approve (only when human_approval_required is explicitly false)
        queue.approve(review_id)
        logger.info(f"✅ Auto-approved: review_id={review_id}")
        return {
            **state,
            "review_id": review_id,
            "review_status": "approved",
            "stage": "human_review",
            "error": None,
        }


async def publish(state: PipelineState) -> PipelineState:
    """Node 8: Publish to TikTok."""
    if state.get("error"):
        return state
    if state.get("dry_run"):
        logger.info("🏃 Dry run — skipping publish.")
        return {**state, "stage": "publish", "publish_status": "dry_run"}

    review_status = state.get("review_status", "pending")
    if review_status == "pending":
        logger.info(
            "⏸️ Review pending — not publishing yet. "
            "Approve via the review CLI, then re-run with --publish-approved."
        )
        return {
            **state,
            "stage": "publish",
            "publish_status": "awaiting_review",
        }

    if review_status == "rejected":
        logger.info("❌ Review rejected — not publishing.")
        return {
            **state,
            "stage": "publish",
            "publish_status": "rejected",
        }

    video_path = state.get("video_path", "")
    narration = state.get("narration", {})
    source_dict = state.get("source_item", {})

    # Build caption
    citation = narration.get("citation", "")
    caption = (
        f"{narration.get('narration', '')[:100]}...\n\n"
        f"📖 {citation}\n\n"
        f"#IslamicStories #Quran #Hadith #Islam #Seerah"
    )

    # Load publishing config
    with open(_POLICY_PATH, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)
    pub_config = policy.get("publishing", {})
    max_posts = pub_config.get("max_posts_per_day", 2)

    # Try API publisher first
    try:
        async with TikTokAPIPublisher(max_posts_per_day=max_posts) as publisher:
            publish_id = await publisher.upload(
                video_path=video_path,
                caption=caption,
                ai_disclosure=True,  # Always True — non-negotiable
            )

            logger.info(f"✅ Published via API: publish_id={publish_id}")

            # Mark as published in review queue
            if state.get("review_id"):
                queue = HumanReviewQueue()
                queue.mark_published(state["review_id"], publish_id)

            return {
                **state,
                "publish_id": publish_id,
                "publish_status": "published",
                "stage": "publish",
                "error": None,
            }

    except PublishError as e:
        logger.warning(f"API publish failed: {e}. Trying browser fallback...")

        # Try browser fallback
        try:
            browser_pub = TikTokBrowserPublisher(max_posts_per_day=max_posts)
            publish_id = await browser_pub.upload(
                video_path=video_path,
                caption=caption,
                ai_disclosure=True,
            )

            logger.info(f"✅ Published via browser fallback: publish_id={publish_id}")

            if state.get("review_id"):
                queue = HumanReviewQueue()
                queue.mark_published(state["review_id"], publish_id)

            return {
                **state,
                "publish_id": publish_id,
                "publish_status": "published_browser",
                "stage": "publish",
                "error": None,
            }

        except BrowserPublishError as be:
            logger.error(
                f"❌ Both API and browser publish failed. "
                f"Manual posting required. API error: {e}. Browser error: {be}"
            )
            return {
                **state,
                "publish_status": "failed_manual_required",
                "stage": "publish",
                "error": (
                    f"Publishing failed. API: {e}. Browser: {be}. "
                    f"Manual posting required for: {video_path}"
                ),
            }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def should_continue(state: PipelineState) -> str:
    """Route to next node or END based on error state."""
    if state.get("error"):
        return "stop"
    return "continue"


def after_review(state: PipelineState) -> str:
    """Route after review: publish if approved, stop otherwise."""
    if state.get("error"):
        return "stop"
    review_status = state.get("review_status", "pending")
    if review_status in ("approved", "dry_run"):
        return "publish"
    return "stop"  # pending or rejected — stop here


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    """Build the LangGraph pipeline."""

    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("source_content", source_content)
    graph.add_node("validate_content", validate_content)
    graph.add_node("write_narration", write_narration)
    graph.add_node("lint_citation", lint_citation)
    graph.add_node("generate_media", generate_media)
    graph.add_node("assemble_video", assemble_video)
    graph.add_node("human_review", human_review)
    graph.add_node("publish", publish)

    # Set entry point
    graph.set_entry_point("source_content")

    # Add conditional edges
    graph.add_conditional_edges(
        "source_content",
        should_continue,
        {"continue": "validate_content", "stop": END},
    )
    graph.add_conditional_edges(
        "validate_content",
        should_continue,
        {"continue": "write_narration", "stop": END},
    )
    graph.add_conditional_edges(
        "write_narration",
        should_continue,
        {"continue": "lint_citation", "stop": END},
    )
    graph.add_conditional_edges(
        "lint_citation",
        should_continue,
        {"continue": "generate_media", "stop": END},
    )
    graph.add_conditional_edges(
        "generate_media",
        should_continue,
        {"continue": "assemble_video", "stop": END},
    )
    graph.add_conditional_edges(
        "assemble_video",
        should_continue,
        {"continue": "human_review", "stop": END},
    )
    graph.add_conditional_edges(
        "human_review",
        after_review,
        {"publish": "publish", "stop": END},
    )
    graph.add_edge("publish", END)

    return graph


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def run_pipeline(
    topic: str,
    source_type: str = "quran",
    dry_run: bool = False,
) -> PipelineState:
    """Run the full pipeline for a given topic."""

    logger.info(
        f"\n{'='*60}\n"
        f"🚀 Islamic Stories Bot Pipeline\n"
        f"   Topic: {topic}\n"
        f"   Source: {source_type}\n"
        f"   Dry run: {dry_run}\n"
        f"   Time: {datetime.now(timezone.utc).isoformat()}\n"
        f"{'='*60}\n"
    )

    initial_state: PipelineState = {
        "topic": topic,
        "source_type": source_type,
        "source_item": None,
        "source_items": [],
        "narration": None,
        "validation_passed": False,
        "validation_errors": [],
        "citation_passed": False,
        "citation_error": "",
        "audio_path": None,
        "visual_path": None,
        "video_path": None,
        "review_id": None,
        "review_status": "",
        "publish_id": None,
        "publish_status": "",
        "error": None,
        "stage": "",
        "dry_run": dry_run,
    }

    graph = build_pipeline()
    compiled = graph.compile()

    # Run the pipeline
    final_state = await compiled.ainvoke(initial_state)

    # Log final status
    if final_state.get("error"):
        logger.error(
            f"\n❌ Pipeline stopped at stage '{final_state.get('stage', '?')}': "
            f"{final_state['error']}\n"
        )
    else:
        logger.info(
            f"\n✅ Pipeline completed successfully!\n"
            f"   Stage: {final_state.get('stage', '?')}\n"
            f"   Video: {final_state.get('video_path', 'N/A')}\n"
            f"   Review: {final_state.get('review_status', 'N/A')}\n"
            f"   Publish: {final_state.get('publish_status', 'N/A')}\n"
        )

    return final_state


async def publish_approved():
    """Publish all approved items in the review queue."""
    queue = HumanReviewQueue()
    approved = queue.get_approved()

    if not approved:
        logger.info("No approved items to publish.")
        return

    with open(_POLICY_PATH, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)
    max_posts = policy.get("publishing", {}).get("max_posts_per_day", 2)

    async with TikTokAPIPublisher(max_posts_per_day=max_posts) as publisher:
        for item in approved:
            daily_count = await publisher.get_daily_post_count()
            if daily_count >= max_posts:
                logger.info(
                    f"Daily post limit reached ({daily_count}/{max_posts}). "
                    f"Remaining items will be published tomorrow."
                )
                break

            narration_text = item.get("narration", "")
            citation = item.get("citation", "")
            caption = (
                f"{narration_text[:100]}...\n\n"
                f"📖 {citation}\n\n"
                f"#IslamicStories #Quran #Hadith #Islam"
            )

            try:
                publish_id = await publisher.upload(
                    video_path=item["video_path"],
                    caption=caption,
                    ai_disclosure=True,
                )
                queue.mark_published(item["id"], publish_id)
                logger.info(f"✅ Published review_id={item['id']}: {publish_id}")

            except PublishError as e:
                logger.error(f"Failed to publish review_id={item['id']}: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Islamic Stories Content Bot Pipeline"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Topic to generate content for (e.g., 'People of the Cave')",
    )
    parser.add_argument(
        "--source-type",
        choices=["quran", "hadith", "seerah"],
        default="quran",
        help="Source type to use (default: quran)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run through sourcing, narration, and linting only (skip media/publish)",
    )
    parser.add_argument(
        "--publish-approved",
        action="store_true",
        help="Publish all approved items in the review queue",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Open the interactive review CLI",
    )

    args = parser.parse_args()

    if args.review:
        from src.review.human_review_queue import run_cli_review
        run_cli_review()
        return

    if args.publish_approved:
        asyncio.run(publish_approved())
        return

    if not args.topic:
        parser.print_help()
        print("\nExamples:")
        print("  python main_pipeline.py 'People of the Cave' --source-type quran")
        print("  python main_pipeline.py 'hijra to medina' --source-type seerah")
        print("  python main_pipeline.py 'People of the Cave' --dry-run")
        print("  python main_pipeline.py --review")
        print("  python main_pipeline.py --publish-approved")
        return

    result = asyncio.run(
        run_pipeline(
            topic=args.topic,
            source_type=args.source_type,
            dry_run=args.dry_run,
        )
    )

    # Exit with error code if pipeline failed
    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
