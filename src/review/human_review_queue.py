"""
Human review queue.

Gates all content before publishing. Nothing reaches the publisher
without explicit human approval while human_approval_required=true.

Uses SQLite for persistence. Can be operated via CLI.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(os.getenv("DATA_DIR", "data")) / "review_queue.db"


class HumanReviewQueue:
    """
    Review queue that gates content before publishing.

    Items must be explicitly approved before they can be published.
    This is a non-negotiable safety layer.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    reference TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    citation TEXT NOT NULL,
                    narration TEXT NOT NULL,
                    metadata_json TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    rejection_reason TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    published_at TEXT
                )
            """)
            conn.commit()

    def submit_for_review(
        self,
        video_path: str,
        source_type: str,
        reference: str,
        collection: str,
        citation: str,
        narration: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Submit a video for human review.

        Returns:
            The review_id for tracking.
        """
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"

        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO review_queue
                (video_path, source_type, reference, collection, citation,
                 narration, metadata_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    video_path,
                    source_type,
                    reference,
                    collection,
                    citation,
                    narration,
                    metadata_json,
                    now,
                ),
            )
            conn.commit()
            review_id = cursor.lastrowid

        logger.info(
            f"Submitted for review: id={review_id}, "
            f"ref={reference}, video={video_path}"
        )
        return review_id

    def get_pending(self) -> list[dict]:
        """Get all items pending review."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM review_queue WHERE status = 'pending' "
                "ORDER BY created_at ASC"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_approved(self) -> list[dict]:
        """Get all approved items that haven't been published yet."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM review_queue WHERE status = 'approved' "
                "AND published_at IS NULL "
                "ORDER BY reviewed_at ASC"
            ).fetchall()
            return [dict(row) for row in rows]

    def approve(self, review_id: int) -> bool:
        """Approve an item for publishing."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(self._db_path)) as conn:
            result = conn.execute(
                "UPDATE review_queue SET status = 'approved', reviewed_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (now, review_id),
            )
            conn.commit()
            success = result.rowcount > 0

        if success:
            logger.info(f"Review item {review_id} APPROVED.")
        else:
            logger.warning(f"Failed to approve review item {review_id}.")
        return success

    def reject(self, review_id: int, reason: str = "") -> bool:
        """Reject an item with an optional reason."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(self._db_path)) as conn:
            result = conn.execute(
                "UPDATE review_queue SET status = 'rejected', "
                "rejection_reason = ?, reviewed_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (reason, now, review_id),
            )
            conn.commit()
            success = result.rowcount > 0

        if success:
            logger.info(f"Review item {review_id} REJECTED: {reason}")
        else:
            logger.warning(f"Failed to reject review item {review_id}.")
        return success

    def mark_published(self, review_id: int, publish_id: str = "") -> bool:
        """Mark an approved item as published."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(self._db_path)) as conn:
            result = conn.execute(
                "UPDATE review_queue SET status = 'published', published_at = ?, "
                "metadata_json = json_set(COALESCE(metadata_json, '{}'), '$.publish_id', ?) "
                "WHERE id = ? AND status = 'approved'",
                (now, publish_id, review_id),
            )
            conn.commit()
            return result.rowcount > 0

    def get_item(self, review_id: int) -> Optional[dict]:
        """Get a specific review item by ID."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM review_queue WHERE id = ?", (review_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_stats(self) -> dict:
        """Get review queue statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            stats = {}
            for status in ["pending", "approved", "rejected", "published"]:
                row = conn.execute(
                    "SELECT COUNT(*) FROM review_queue WHERE status = ?",
                    (status,),
                ).fetchone()
                stats[status] = row[0]
            stats["total"] = sum(stats.values())
            return stats


def run_cli_review():
    """
    Interactive CLI for reviewing pending items.

    Usage: python -m src.review.human_review_queue
    """
    queue = HumanReviewQueue()
    pending = queue.get_pending()

    if not pending:
        print("\n✅ No items pending review.\n")
        stats = queue.get_stats()
        print(f"Queue stats: {stats}")
        return

    print(f"\n📋 {len(pending)} item(s) pending review:\n")
    print("=" * 70)

    for item in pending:
        print(f"\n🆔 Review ID: {item['id']}")
        print(f"📖 Source: {item['source_type']} — {item['collection']}")
        print(f"📌 Reference: {item['reference']}")
        print(f"🎬 Video: {item['video_path']}")
        print(f"📝 Citation: {item['citation']}")
        print(f"\n📜 Narration Preview:")
        narration = item["narration"]
        # Show first 300 chars
        preview = narration[:300] + "..." if len(narration) > 300 else narration
        print(f"   {preview}")
        print(f"\n⏰ Submitted: {item['created_at']}")
        print("-" * 70)

        while True:
            action = input("\n[A]pprove / [R]eject / [S]kip / [Q]uit? ").strip().lower()
            if action == "a":
                queue.approve(item["id"])
                print("✅ Approved!")
                break
            elif action == "r":
                reason = input("Rejection reason: ").strip()
                queue.reject(item["id"], reason)
                print(f"❌ Rejected: {reason}")
                break
            elif action == "s":
                print("⏭️ Skipped.")
                break
            elif action == "q":
                print("\n👋 Exiting review.")
                return
            else:
                print("Please enter A, R, S, or Q.")

    print("\n📊 Updated stats:")
    print(queue.get_stats())


if __name__ == "__main__":
    run_cli_review()
