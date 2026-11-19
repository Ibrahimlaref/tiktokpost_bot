"""
Visual generator.

Creates non-figurative background visuals for video assembly.
Respects content policy flags (depict_prophets, depict_sahaba_faces)
by defaulting to calligraphy, geometric patterns, and nature/architecture
silhouettes.

Ships with built-in gradient/geometric templates. Optional image-gen API
integration adds a negative prompt against human faces for Prophet/Sahaba content.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from src.sourcing.base_source import SourceItem

logger = logging.getLogger(__name__)

# Color palettes for different visual styles
PALETTES = {
    "calligraphy_and_nature": {
        "backgrounds": [
            [(15, 25, 45), (25, 45, 75)],       # Deep navy gradient
            [(20, 30, 20), (35, 55, 35)],        # Forest dark green
            [(45, 25, 15), (75, 45, 25)],        # Warm earth tones
            [(25, 20, 40), (45, 35, 70)],        # Deep purple
            [(10, 15, 30), (20, 35, 55)],        # Midnight blue
        ],
        "accent": (218, 165, 32),  # Gold
        "text": (245, 240, 230),   # Warm white
    },
    "abstract_geometric": {
        "backgrounds": [
            [(10, 10, 25), (20, 20, 45)],
            [(25, 10, 10), (50, 20, 20)],
            [(10, 25, 25), (20, 50, 50)],
        ],
        "accent": (100, 200, 200),  # Teal
        "text": (240, 240, 240),
    },
    "silhouette_no_face": {
        "backgrounds": [
            [(30, 20, 10), (60, 40, 20)],  # Sunset amber
            [(10, 15, 25), (25, 35, 55)],  # Dusk blue
        ],
        "accent": (255, 180, 50),  # Amber
        "text": (255, 250, 240),
    },
}


class VisualGenerator:
    """
    Generates non-figurative background visuals for Islamic content videos.

    When depict_prophets=false and depict_sahaba_faces=false:
    - Uses calligraphy, geometric patterns, nature/architecture silhouettes
    - If image generation API is used: adds negative prompt against human faces
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: str = "calligraphy_and_nature",
        depict_prophets: bool = False,
        depict_sahaba_faces: bool = False,
        width: int = 1080,
        height: int = 1920,
    ):
        self._output_dir = output_dir or Path(
            os.getenv("OUTPUT_DIR", "output")
        ) / "visuals"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._style = style
        self._depict_prophets = depict_prophets
        self._depict_sahaba_faces = depict_sahaba_faces
        self._width = width
        self._height = height
        self._palette = PALETTES.get(style, PALETTES["calligraphy_and_nature"])

    async def generate_background(
        self,
        source_item: SourceItem,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate a background image for the video.

        Uses built-in templates (gradients + geometric patterns).
        Future: integrate with image generation API.

        Args:
            source_item: The source item for context.
            filename: Optional output filename.

        Returns:
            Path to the generated background image (PNG).
        """
        if not filename:
            filename = f"bg_{source_item.source_type}_{source_item.reference.replace(':', '_')}.png"

        output_path = self._output_dir / filename

        # Select a random gradient from the palette
        bg_colors = random.choice(self._palette["backgrounds"])

        # Create the background image
        img = self._create_gradient_background(bg_colors[0], bg_colors[1])

        # Add geometric/Islamic patterns
        img = self._add_geometric_pattern(img)

        # Add Arabic calligraphy text overlay (if available)
        if source_item.arabic_text:
            img = self._add_arabic_text_overlay(img, source_item.arabic_text)

        img.save(str(output_path), "PNG", quality=95)
        logger.info(f"Background generated: {output_path}")
        return output_path

    def _create_gradient_background(
        self, color_top: tuple, color_bottom: tuple
    ) -> Image.Image:
        """Create a vertical gradient background."""
        img = Image.new("RGB", (self._width, self._height))
        draw = ImageDraw.Draw(img)

        for y in range(self._height):
            ratio = y / self._height
            r = int(color_top[0] + (color_bottom[0] - color_top[0]) * ratio)
            g = int(color_top[1] + (color_bottom[1] - color_top[1]) * ratio)
            b = int(color_top[2] + (color_bottom[2] - color_top[2]) * ratio)
            draw.line([(0, y), (self._width, y)], fill=(r, g, b))

        return img

    def _add_geometric_pattern(self, img: Image.Image) -> Image.Image:
        """
        Add Islamic geometric pattern overlay.

        Draws non-figurative geometric shapes inspired by Islamic art:
        interlocking circles, star patterns, and arabesques.
        """
        draw = ImageDraw.Draw(img, "RGBA")
        accent = self._palette["accent"]

        # Draw subtle geometric grid
        grid_size = 120
        line_color = (*accent, 25)  # Very subtle

        # Interlocking circles pattern
        for x in range(0, self._width + grid_size, grid_size):
            for y in range(0, self._height + grid_size, grid_size):
                radius = grid_size // 3
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    outline=line_color,
                    width=1,
                )

        # Add corner decorative elements
        corner_size = 200
        corner_color = (*accent, 40)

        # Top-left corner decoration
        for i in range(5):
            offset = i * 15
            draw.arc(
                [offset, offset, corner_size - offset, corner_size - offset],
                start=0,
                end=90,
                fill=corner_color,
                width=2,
            )

        # Bottom-right corner decoration
        for i in range(5):
            offset = i * 15
            draw.arc(
                [
                    self._width - corner_size + offset,
                    self._height - corner_size + offset,
                    self._width - offset,
                    self._height - offset,
                ],
                start=180,
                end=270,
                fill=corner_color,
                width=2,
            )

        # Central star pattern (8-pointed star)
        center_x = self._width // 2
        center_y = self._height // 2
        star_color = (*accent, 15)

        import math

        for angle_offset in range(0, 360, 45):
            for radius in range(100, 400, 50):
                angle_rad = math.radians(angle_offset)
                x1 = center_x + int(radius * math.cos(angle_rad))
                y1 = center_y + int(radius * math.sin(angle_rad))
                x2 = center_x + int((radius + 50) * math.cos(angle_rad + 0.3))
                y2 = center_y + int((radius + 50) * math.sin(angle_rad + 0.3))
                draw.line([(x1, y1), (x2, y2)], fill=star_color, width=1)

        return img

    def _add_arabic_text_overlay(
        self, img: Image.Image, arabic_text: str
    ) -> Image.Image:
        """
        Add a subtle Arabic calligraphy text overlay in the center.

        Uses large, semi-transparent text as a decorative background element.
        """
        draw = ImageDraw.Draw(img, "RGBA")
        text_color = (*self._palette["text"], 20)  # Very subtle

        # Use default font (would use Amiri/Noto Naskh Arabic in production)
        try:
            # Try to load a system Arabic font
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
                "/System/Library/Fonts/GeezaPro.ttc",
            ]
            font = None
            for fp in font_paths:
                if Path(fp).exists():
                    font = ImageFont.truetype(fp, 72)
                    break
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Truncate text for overlay (use first ~50 chars as decorative element)
        display_text = arabic_text[:100] if len(arabic_text) > 100 else arabic_text

        # Center the text
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self._width - text_width) // 2
        y = (self._height - text_height) // 2

        draw.text((x, y), display_text, fill=text_color, font=font)

        return img

    def get_negative_prompt(self, source_item: SourceItem) -> str:
        """
        Build negative prompt for image generation API (if used).

        When depict_prophets=false and depict_sahaba_faces=false, the negative
        prompt explicitly blocks human likenesses.
        """
        negative_parts = []

        if not self._depict_prophets:
            negative_parts.extend([
                "human face", "portrait", "person", "prophet",
                "man", "figure", "figurative",
            ])

        if not self._depict_sahaba_faces:
            negative_parts.extend([
                "companion", "sahaba", "sahabi",
            ])

        # Always exclude for safety
        negative_parts.extend([
            "nsfw", "violent", "blood", "weapon",
            "modern", "contemporary", "political",
        ])

        return ", ".join(negative_parts)

    def get_positive_prompt(self, source_item: SourceItem) -> str:
        """
        Build positive prompt for image generation API (if used).

        Focuses on non-figurative Islamic art styles.
        """
        style_prompts = {
            "calligraphy_and_nature": (
                "Islamic calligraphy art, arabesque patterns, "
                "ornamental geometric design, nature landscape, "
                "mountains, desert sunset, starry sky, "
                "gold and deep blue color palette, "
                "high quality, elegant, serene"
            ),
            "abstract_geometric": (
                "Islamic geometric pattern, tessellation, "
                "abstract mathematical art, zellige tiles, "
                "muqarnas, intricate patterns, "
                "dark background, luminous lines"
            ),
            "silhouette_no_face": (
                "Mosque silhouette, minaret against sunset, "
                "architectural silhouette, palm trees, "
                "desert landscape, golden hour photography, "
                "no people, no faces"
            ),
        }

        return style_prompts.get(self._style, style_prompts["calligraphy_and_nature"])
