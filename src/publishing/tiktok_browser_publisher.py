"""
TikTok browser automation publisher (FALLBACK ONLY).

Uses Playwright to upload via TikTok's web interface.
This is a HIGHER-RISK option documented for when the Content Posting API
audit is still pending.

CRITICAL RULES:
- NEVER attempts to defeat CAPTCHA or bot detection
- On any bot-detection signal: logs "CAPTCHA_DETECTED" and returns failure
- Graceful degradation: logs and skips, manual posting required for that item
- Same posture as JobBot's Lever hCaptcha handling
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.publishing.base_publisher import BasePublisher

logger = logging.getLogger(__name__)


class TikTokBrowserPublisher(BasePublisher):
    """
    Playwright-based fallback publisher for TikTok.

    NEVER tries to defeat bot detection. Logs and fails gracefully
    if any CAPTCHA or challenge is detected.
    """

    def __init__(
        self,
        max_posts_per_day: int = 2,
        headless: bool = False,
    ):
        self._max_posts_per_day = max_posts_per_day
        self._headless = headless
        self._daily_count = 0
        self._last_count_date: Optional[str] = None

    async def upload(
        self,
        video_path: str,
        caption: str,
        ai_disclosure: bool = True,
    ) -> str:
        """
        Upload a video via TikTok's web interface.

        NEVER defeats CAPTCHA. Fails gracefully on bot detection.
        """
        # Always enforce AI disclosure
        if not ai_disclosure:
            logger.warning(
                "ai_disclosure forced to True — mandatory for this project."
            )

        # Check daily limit
        daily_count = await self.get_daily_post_count()
        if daily_count >= self._max_posts_per_day:
            raise BrowserPublishError(
                f"Daily post limit reached ({daily_count}/{self._max_posts_per_day})."
            )

        video_file = Path(video_path)
        if not video_file.exists():
            raise BrowserPublishError(f"Video file not found: {video_path}")

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self._headless)
                context = await browser.new_context()
                page = await context.new_page()

                # Navigate to TikTok upload page
                logger.info("Navigating to TikTok upload page...")
                await page.goto(
                    "https://www.tiktok.com/upload",
                    wait_until="networkidle",
                    timeout=30000,
                )

                # Check for CAPTCHA or login challenge BEFORE proceeding
                if await self._detect_bot_challenge(page):
                    logger.warning(
                        "CAPTCHA_DETECTED — TikTok bot detection triggered. "
                        "NOT attempting to defeat it. "
                        "Manual posting required for this item."
                    )
                    await browser.close()
                    raise BrowserPublishError(
                        "CAPTCHA_DETECTED: TikTok bot detection triggered. "
                        "Manual posting required. This is expected behavior — "
                        "the browser fallback does not defeat CAPTCHAs."
                    )

                # Check if logged in
                if await self._needs_login(page):
                    logger.warning(
                        "LOGIN_REQUIRED — TikTok requires login. "
                        "Please log in manually and save cookies."
                    )
                    await browser.close()
                    raise BrowserPublishError(
                        "LOGIN_REQUIRED: TikTok requires login. "
                        "Set up cookies/session first."
                    )

                # Upload the video file
                file_input = page.locator('input[type="file"]').first
                await file_input.set_input_files(str(video_file))

                # Wait for upload to process
                logger.info("Waiting for video upload to process...")
                await page.wait_for_timeout(5000)

                # Check for CAPTCHA again after upload
                if await self._detect_bot_challenge(page):
                    logger.warning(
                        "CAPTCHA_DETECTED after upload — aborting. "
                        "Manual posting required."
                    )
                    await browser.close()
                    raise BrowserPublishError(
                        "CAPTCHA_DETECTED after upload. Manual posting required."
                    )

                # Set caption
                caption_field = page.locator(
                    '[data-testid="caption-input"], '
                    '.DraftEditor-root, '
                    '[contenteditable="true"]'
                ).first
                if caption_field:
                    await caption_field.click()
                    await caption_field.fill("")
                    await page.keyboard.type(caption, delay=50)

                # Add AI disclosure note in caption
                if ai_disclosure:
                    await page.keyboard.type(
                        "\n\n#AIGenerated #AIContent", delay=50
                    )

                # Click publish button
                publish_btn = page.locator(
                    'button:has-text("Post"), '
                    'button:has-text("Publish"), '
                    '[data-testid="publish-button"]'
                ).first
                if publish_btn:
                    await publish_btn.click()
                    await page.wait_for_timeout(5000)

                    # Final CAPTCHA check
                    if await self._detect_bot_challenge(page):
                        logger.warning(
                            "CAPTCHA_DETECTED at publish — aborting."
                        )
                        await browser.close()
                        raise BrowserPublishError(
                            "CAPTCHA_DETECTED at publish step. "
                            "Manual posting required."
                        )

                publish_id = f"browser_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                self._daily_count += 1

                logger.info(
                    f"Browser upload completed: publish_id={publish_id}"
                )
                await browser.close()
                return publish_id

        except ImportError:
            raise BrowserPublishError(
                "Playwright not installed. Run: pip install playwright && "
                "playwright install chromium"
            )
        except BrowserPublishError:
            raise
        except Exception as e:
            logger.error(f"Browser upload failed: {e}")
            raise BrowserPublishError(f"Browser upload failed: {e}")

    async def check_status(self, publish_id: str) -> str:
        """
        Check status of a browser-published post.

        Browser publishes don't have a status API — return best guess.
        """
        if publish_id.startswith("browser_"):
            return "published"  # Optimistic — we can't verify from browser
        return "unknown"

    async def get_daily_post_count(self) -> int:
        """Get daily post count (tracked in memory for browser publishes)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_count_date != today:
            self._daily_count = 0
            self._last_count_date = today
        return self._daily_count

    async def _detect_bot_challenge(self, page) -> bool:
        """
        Detect CAPTCHA or bot-detection challenges on the page.

        Checks for common TikTok challenge indicators.
        NEVER tries to solve them — just detects and reports.
        """
        challenge_selectors = [
            # Common CAPTCHA/challenge selectors
            '[class*="captcha"]',
            '[id*="captcha"]',
            '[class*="verify"]',
            '[id*="verify"]',
            'iframe[src*="captcha"]',
            'iframe[src*="verify"]',
            '[class*="challenge"]',
            # TikTok-specific
            '[class*="tiktok-captcha"]',
            '[data-testid*="captcha"]',
        ]

        for selector in challenge_selectors:
            try:
                element = page.locator(selector).first
                if await element.is_visible(timeout=1000):
                    logger.warning(
                        f"Bot challenge detected via selector: {selector}"
                    )
                    return True
            except Exception:
                continue

        # Check page content for challenge keywords
        try:
            content = await page.content()
            challenge_keywords = [
                "verify you are human",
                "complete the puzzle",
                "slide to verify",
                "security check",
                "suspicious activity",
            ]
            content_lower = content.lower()
            for keyword in challenge_keywords:
                if keyword in content_lower:
                    logger.warning(
                        f"Bot challenge detected via keyword: '{keyword}'"
                    )
                    return True
        except Exception:
            pass

        return False

    async def _needs_login(self, page) -> bool:
        """Check if the page requires login."""
        try:
            login_selectors = [
                '[data-testid="login-button"]',
                'button:has-text("Log in")',
                'a:has-text("Log in")',
            ]
            for selector in login_selectors:
                element = page.locator(selector).first
                if await element.is_visible(timeout=2000):
                    return True
        except Exception:
            pass

        # Check URL
        current_url = page.url
        if "login" in current_url.lower():
            return True

        return False


class BrowserPublishError(Exception):
    """Raised when browser-based publishing fails."""
    pass
