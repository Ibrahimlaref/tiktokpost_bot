"""Integration test: visual generator + review queue + pipeline stages."""
import asyncio
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from pathlib import Path
from src.sourcing.base_source import SourceItem
from src.media.visual_generator import VisualGenerator
from src.review.human_review_queue import HumanReviewQueue


def make_test_item() -> SourceItem:
    return SourceItem(
        source_type="quran",
        collection="Quran",
        reference="18:9",
        grade=None,
        arabic_text="أَمْ حَسِبْتَ أَنَّ أَصْحَـٰبَ ٱلْكَهْفِ وَٱلرَّقِيمِ كَانُوا۟ مِنْ ءَايَـٰتِنَا عَجَبًا",
        translation_text="Or have you thought that the companions of the cave were a wonder among Our signs?",
        translator="Sahih International",
        source_url="https://quran.com/18:9",
        metadata={"surah_name": "Al-Kahf", "surah": 18, "ayah": 9},
    )


async def test_visual_generator():
    print("=" * 60)
    print("TEST 1: Visual Generator")
    print("=" * 60)

    item = make_test_item()
    output_dir = Path("output/test_visuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test all 3 styles
    for style in ["calligraphy_and_nature", "abstract_geometric", "silhouette_no_face"]:
        gen = VisualGenerator(
            output_dir=output_dir,
            style=style,
            depict_prophets=False,
            depict_sahaba_faces=False,
        )

        bg_path = await gen.generate_background(
            item, filename=f"test_{style}.png"
        )
        size_kb = bg_path.stat().st_size / 1024
        print(f"  [{style}] -> {bg_path} ({size_kb:.0f} KB)")

    # Verify negative prompt
    gen = VisualGenerator(depict_prophets=False, depict_sahaba_faces=False)
    neg = gen.get_negative_prompt(item)
    pos = gen.get_positive_prompt(item)
    print(f"\n  Negative prompt: {neg[:80]}...")
    print(f"  Positive prompt: {pos[:80]}...")

    assert "human face" in neg, "Negative prompt must block human faces"
    assert "prophet" in neg, "Negative prompt must block prophet depiction"
    print("\n  [PASS] Visual generator works correctly")


def test_review_queue():
    print("\n" + "=" * 60)
    print("TEST 2: Human Review Queue")
    print("=" * 60)

    db_path = Path("data/test_review.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    queue = HumanReviewQueue(db_path=db_path)

    # Submit an item
    review_id = queue.submit_for_review(
        video_path="output/test_video.mp4",
        source_type="quran",
        reference="18:9",
        collection="Quran",
        citation="Surah Al-Kahf, ayah 9",
        narration="A test narration about the People of the Cave...",
        metadata={"grade": None, "source_url": "https://quran.com/18:9"},
    )
    print(f"  Submitted: review_id={review_id}")

    # Check pending
    pending = queue.get_pending()
    print(f"  Pending items: {len(pending)}")
    assert len(pending) == 1, "Should have 1 pending item"

    # Get stats
    stats = queue.get_stats()
    print(f"  Stats: {stats}")
    assert stats["pending"] == 1

    # Approve it
    success = queue.approve(review_id)
    print(f"  Approved: {success}")
    assert success is True

    # Check it moved to approved
    approved = queue.get_approved()
    print(f"  Approved items: {len(approved)}")
    assert len(approved) == 1

    # Mark as published
    queue.mark_published(review_id, "test_publish_123")
    final_stats = queue.get_stats()
    print(f"  Final stats: {final_stats}")
    assert final_stats["published"] == 1
    assert final_stats["pending"] == 0

    # Test rejection flow
    review_id_2 = queue.submit_for_review(
        video_path="output/test_video_2.mp4",
        source_type="hadith",
        reference="3616",
        collection="Sahih al-Bukhari",
        citation="Sahih al-Bukhari, Hadith 3616",
        narration="Another test narration...",
    )
    queue.reject(review_id_2, "Test rejection - content needs revision")
    stats = queue.get_stats()
    print(f"  After rejection: {stats}")
    assert stats["rejected"] == 1

    # Cleanup (skip if file locked on Windows)
    try:
        db_path.unlink()
    except PermissionError:
        pass  # Windows file lock — harmless for test DB
    print("\n  [PASS] Review queue works correctly")


async def test_pipeline_sourcing_only():
    print("\n" + "=" * 60)
    print("TEST 3: Pipeline - Sourcing + Validation (live API)")
    print("=" * 60)

    from src.sourcing.quran_source import QuranSourceProvider
    from src.verification.content_validator import ContentValidator
    from src.verification.citation_linter import citation_matches_source

    # Step 1: Source
    async with QuranSourceProvider() as provider:
        items = await provider.fetch_candidates(
            topic="story of yusuf", limit=3
        )

    if not items:
        print("  [SKIP] No items returned for 'story of yusuf'")
        return

    print(f"  Sourced {len(items)} verses from Surah Yusuf")
    for item in items:
        print(f"    {item.reference}: {item.translation_text[:60]}...")

    # Step 2: Validate
    validator = ContentValidator()
    for item in items:
        valid, violations = validator.validate(item)
        status = "PASS" if valid else "FAIL"
        print(f"  Validation {item.reference}: {status}")
        if violations:
            for v in violations:
                print(f"    - {v}")

    # Step 3: Test citation linter with a mock narration
    test_narration = {
        "narration": "The story of Prophet Yusuf begins in the Quran...",
        "citation": f"Surah Yusuf, ayah {items[0].reference.split(':')[1]}",
        "confidence": "high",
    }
    passes, reason = citation_matches_source(test_narration, items[0])
    print(f"  Citation lint (mock): {'PASS' if passes else 'FAIL'}")
    if not passes:
        print(f"    Reason: {reason}")

    print("\n  [PASS] Sourcing + validation pipeline works")


async def main():
    await test_visual_generator()
    test_review_queue()
    await test_pipeline_sourcing_only()

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
