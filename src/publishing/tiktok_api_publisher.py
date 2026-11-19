"""
TikTok Content Posting API publisher (Direct Post).

Uses the official TikTok Content Posting API for publishing.
- Sets AI-disclosure flag on every upload
- Respects max_posts_per_day
- Defaults to SELF_ONLY until TikTok audit passes
- Tracks daily post count in SQLite
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

from src.publishing.base_publisher import BasePublisher

load_dotenv()
logger = logging.getLogger(__name__)


class TikTokAPIPublisher(BasePublisher):
    """
    TikTok Content Posting API (Direct Post) publisher.

    Workflow:
    1. Query creator info for available privacy levels
    2. Init post with video + AI disclosure flag
    3. Upload video binary
    4. Poll status until PUBLISH_COMPLETE
    """

    def __init__(
        self,
        max_posts_per_day: int = 2,
        default_privacy: str = "SELF_ONLY",
        db_path: Optional[Path] = None,
    ):
        self._api_base = "https://open.tiktokapis.com/v2"
        self._client_key = os.getenv("TIKTOK_CLIENT_KEY", "")
        self._client_secret = os.getenv("TIKTOK_CLIENT_SECRET", "")
        self._access_token = os.getenv("TIKTOK_ACCESS_TOKEN", "")

        self._max_posts_per_day = max_posts_per_day
        self._default_privacy = default_privacy

        self._db_path = db_path or Path(
            os.getenv("DATA_DIR", "data")
        ) / "publish_log.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        self._client = httpx.AsyncClient(timeout=120.0)

    def _init_db(self):
        """Initialize the publish tracking database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS publish_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    publish_id TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    caption TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    privacy_level TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT
                )
            """)
            conn.commit()

    async def upload(
        self,
        video_path: str,
        caption: str,
        ai_disclosure: bool = True,
    ) -> str:
        """
        Upload and publish a video via TikTok Content Posting API.

        ai_disclosure is always True — enforced by the pipeline,
        required by TikTok policy for AI-generated content.
        """
        # Always enforce AI disclosure
        if not ai_disclosure:
            logger.warning(
                "ai_disclosure was False but is being forced to True. "
                "AI disclosure is mandatory for this project."
            )
            ai_disclosure = True

        # Check daily post limit
        daily_count = await self.get_daily_post_count()
        if daily_count >= self._max_posts_per_day:
            raise PublishError(
                f"Daily post limit reached ({daily_count}/{self._max_posts_per_day}). "
                f"Try again tomorrow."
            )

        if not self._access_token:
            raise PublishError(
                "No TikTok access token configured. "
                "Set TIKTOK_ACCESS_TOKEN in .env"
            )

        # Step 1: Query creator info
        creator_info = await self._query_creator_info()
        privacy_level = self._select_privacy_level(creator_info)

        # Step 2: Initialize the post
        publish_id = await self._init_post(
            video_path=video_path,
            caption=caption,
            privacy_level=privacy_level,
            ai_disclosure=ai_disclosure,
        )

        # Step 3: Log the publish attempt
        self._log_publish(publish_id, video_path, caption, privacy_level)

        logger.info(
            f"TikTok post initiated: publish_id={publish_id}, "
            f"privacy={privacy_level}, ai_disclosure=True"
        )
        return publish_id

    async def check_status(self, publish_id: str) -> str:
        """Poll the publish status endpoint."""
        try:
            response = await self._client.post(
                f"{self._api_base}/post/publish/status/fetch/",
                headers=self._auth_headers(),
                json={"publish_id": publish_id},
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("data", {}).get("status", "unknown")

            # Update local log
            self._update_publish_status(publish_id, status)

            return status

        except httpx.HTTPError as e:
            logger.error(f"Failed to check publish status: {e}")
            return "error"

    async def get_daily_post_count(self) -> int:
        """Get the number of posts made today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM publish_log "
                "WHERE created_at LIKE ? AND status != 'failed'",
                (f"{today}%",),
            ).fetchone()
            return row[0] if row else 0

    async def _query_creator_info(self) -> dict:
        """Query the creator info endpoint for available privacy levels."""
        try:
            response = await self._client.post(
                f"{self._api_base}/post/publish/creator_info/query/",
                headers=self._auth_headers(),
            )
            response.raise_for_status()
            return response.json().get("data", {})
        except httpx.HTTPError as e:
            logger.warning(f"Failed to query creator info: {e}")
            return {}

    async def _init_post(
        self,
        video_path: str,
        caption: str,
        privacy_level: str,
        ai_disclosure: bool,
    ) -> str:
        """Initialize a Direct Post upload."""
        video_file = Path(video_path)
        if not video_file.exists():
            raise PublishError(f"Video file not found: {video_path}")

        file_size = video_file.stat().st_size

        post_info = {
            "title": caption[:150],  # TikTok title limit
            "privacy_level": privacy_level,
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
            "video_cover_timestamp_ms": 1000,
        }

        # Set AI disclosure — required for all content from this pipeline
        if ai_disclosure:
            post_info["is_ai_generated"] = True

        source_info = {
            "source": "FILE_UPLOAD",
            "video_size": file_size,
            "chunk_size": min(file_size, 10_000_000),  # 10MB chunks
            "total_chunk_count": max(1, file_size // 10_000_000 + 1),
        }

        try:
            response = await self._client.post(
                f"{self._api_base}/post/publish/video/init/",
                headers=self._auth_headers(),
                json={
                    "post_info": post_info,
                    "source_info": source_info,
                },
            )
            response.raise_for_status()
            data = response.json()

            publish_id = data.get("data", {}).get("publish_id", "")
            upload_url = data.get("data", {}).get("upload_url", "")

            if not publish_id:
                raise PublishError(
                    f"No publish_id in response: {data}"
                )

            # Upload the video file
            if upload_url:
                await self._upload_video_file(upload_url, video_path)

            return publish_id

        except httpx.HTTPError as e:
            raise PublishError(f"Failed to init TikTok post: {e}")

    async def _upload_video_file(self, upload_url: str, video_path: str):
        """Upload the video binary to TikTok's upload URL."""
        video_file = Path(video_path)

        try:
            with open(video_file, "rb") as f:
                video_data = f.read()

            response = await self._client.put(
                upload_url,
                content=video_data,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Range": f"bytes 0-{len(video_data) - 1}/{len(video_data)}",
                },
                timeout=300.0,
            )
            response.raise_for_status()
            logger.info(f"Video uploaded successfully: {len(video_data)} bytes")

        except httpx.HTTPError as e:
            raise PublishError(f"Failed to upload video: {e}")

    def _select_privacy_level(self, creator_info: dict) -> str:
        """
        Select privacy level based on available options.

        Defaults to SELF_ONLY until TikTok audit passes.
        """
        available_levels = creator_info.get("privacy_level_options", [])

        if self._default_privacy in available_levels:
            return self._default_privacy

        # If default isn't available, use SELF_ONLY (safest)
        if "SELF_ONLY" in available_levels:
            return "SELF_ONLY"

        # Fallback to whatever is available, preferring private
        for level in ["SELF_ONLY", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR", "PUBLIC_TO_EVERYONE"]:
            if level in available_levels:
                return level

        return self._default_privacy

    def _auth_headers(self) -> dict:
        """Build authorization headers."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

    def _log_publish(
        self, publish_id: str, video_path: str, caption: str, privacy: str
    ):
        """Log a publish attempt."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO publish_log
                (publish_id, video_path, caption, status, privacy_level, created_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (publish_id, video_path, caption, privacy, now),
            )
            conn.commit()

    def _update_publish_status(self, publish_id: str, status: str):
        """Update the status of a publish attempt in the log."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE publish_log SET status = ?, completed_at = ? "
                "WHERE publish_id = ?",
                (status, now, publish_id),
            )
            conn.commit()

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class PublishError(Exception):
    """Raised when publishing fails."""
    pass
