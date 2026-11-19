"""
Base publisher interface.

All publishers (TikTok API, TikTok browser fallback) implement this ABC.
ai_disclosure is always True in this project — it's not optional.
"""

from abc import ABC, abstractmethod


class BasePublisher(ABC):
    """Abstract base class for all content publishers."""

    @abstractmethod
    async def upload(
        self,
        video_path: str,
        caption: str,
        ai_disclosure: bool = True,
    ) -> str:
        """
        Upload and publish a video.

        Args:
            video_path: Path to the video file.
            caption: Post caption/description.
            ai_disclosure: Always True in this project. Marks the
                content as AI-generated per platform policy.

        Returns:
            publish_id: A unique identifier for the published post.
        """

    @abstractmethod
    async def check_status(self, publish_id: str) -> str:
        """
        Check the status of a published post.

        Args:
            publish_id: The ID returned from upload().

        Returns:
            Status string: "pending", "published", "failed", etc.
        """

    @abstractmethod
    async def get_daily_post_count(self) -> int:
        """
        Get the number of posts made today.

        Used to enforce max_posts_per_day limit.
        """
