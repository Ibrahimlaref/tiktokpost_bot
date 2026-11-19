"""Quick test: fetch Quran verses from live API and validate them."""
import asyncio
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.sourcing.quran_source import QuranSourceProvider
from src.verification.content_validator import ContentValidator


async def test_live_quran():
    print("=" * 60)
    print("LIVE TEST: Quran Source Provider + Content Validator")
    print("=" * 60)

    async with QuranSourceProvider() as provider:
        # Test 1: Fetch "People of the Cave" (Surah 18:9-26)
        print("\n[Test 1] Fetching 'People of the Cave'...")
        items = await provider.fetch_candidates(topic="People of the Cave", limit=3)

        if not items:
            print("[FAIL] No items returned - API may be down or mapping not found.")
            return

        print(f"[PASS] Got {len(items)} verse(s)\n")

        validator = ContentValidator()

        for i, item in enumerate(items):
            print(f"--- Verse {i+1} ---")
            print(f"  Reference: {item.reference}")
            print(f"  Collection: {item.collection}")
            print(f"  Grade: {item.grade}")
            if item.arabic_text:
                print(f"  Arabic: {item.arabic_text[:80]}...")
            print(f"  Translation: {item.translation_text[:120]}...")
            print(f"  Surah name: {item.metadata.get('surah_name', 'N/A')}")
            print(f"  Source URL: {item.source_url}")
            print(f"  Citation display: {item.citation_display}")

            # Validate
            is_valid, violations = validator.validate(item)
            print(f"  Validation: {'PASS' if is_valid else 'FAIL'}")
            if violations:
                for v in violations:
                    print(f"    WARNING: {v}")
            print()

        # Test 2: Verify a single verse
        print("\n[Test 2] Verifying verse {0}...".format(items[0].reference))
        verified = await provider.verify(items[0])
        print(f"  Verification: {'PASS' if verified else 'FAIL'}")

        # Test 3: Fetch by reference
        print("\n[Test 3] Fetch by reference '18:10'...")
        single = await provider.fetch_by_reference("18:10")
        if single:
            print(f"  [PASS] Got verse: {single.reference}")
            print(f"  Translation: {single.translation_text[:120]}...")
        else:
            print("  [FAIL] Could not fetch verse by reference")

        # Test 4: Fetch tafsir
        print("\n[Test 4] Fetching tafsir for 18:9...")
        tafsir = await provider.fetch_tafsir("18:9")
        if tafsir:
            print(f"  [PASS] Tafsir length: {len(tafsir)} chars")
            print(f"  Preview: {tafsir[:200]}...")
        else:
            print("  [INFO] Tafsir not available (API may not support this endpoint)")

    print("\n" + "=" * 60)
    print("LIVE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_live_quran())
