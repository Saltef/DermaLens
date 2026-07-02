from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageStat

MAX_SIDE = 1280


@dataclass(frozen=True)
class Region:
    name: str
    box: tuple[int, int, int, int]
    mean_rgb: tuple[float, float, float]
    brightness: float
    redness_index: float
    contrast: float

    def to_public_dict(self) -> dict:
        return {
            "name": self.name,
            "box": self.box,
            "brightness": round(self.brightness, 3),
            "redness_index": round(self.redness_index, 3),
            "contrast": round(self.contrast, 3),
        }


def clean_image(image: Image.Image) -> tuple[bytes, Image.Image]:
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=94, optimize=True)
    cleaned_bytes = out.getvalue()
    cleaned = Image.open(io.BytesIO(cleaned_bytes)).convert("RGB")
    return cleaned_bytes, cleaned


def prepare_regions(image: Image.Image) -> list[Region]:
    width, height = image.size

    # Simple face-photo assumptions for MVP. Replace with MediaPipe/RetinaFace later.
    boxes = {
        "forehead": _box(width, height, 0.28, 0.10, 0.72, 0.30),
        "left_cheek": _box(width, height, 0.18, 0.36, 0.44, 0.66),
        "right_cheek": _box(width, height, 0.56, 0.36, 0.82, 0.66),
        "nose": _box(width, height, 0.42, 0.32, 0.58, 0.62),
        "chin": _box(width, height, 0.34, 0.66, 0.66, 0.88),
    }
    return [_summarize_region(name, image.crop(box), box) for name, box in boxes.items()]


def image_quality(image: Image.Image) -> dict:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = stat.mean[0] / 255.0
    contrast = stat.stddev[0] / 128.0
    warnings = []
    if brightness < 0.22:
        warnings.append("Image appears underexposed.")
    if brightness > 0.88:
        warnings.append("Image appears overexposed.")
    if contrast < 0.18:
        warnings.append("Image appears low contrast or soft.")
    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "warnings": warnings,
    }


def _box(width: int, height: int, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    return (
        max(0, round(width * x1)),
        max(0, round(height * y1)),
        min(width, round(width * x2)),
        min(height, round(height * y2)),
    )


def _summarize_region(name: str, crop: Image.Image, box: tuple[int, int, int, int]) -> Region:
    stat = ImageStat.Stat(crop)
    r, g, b = stat.mean
    std = sum(stat.stddev) / (3 * 128.0)
    brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    redness = max(0.0, (r - ((g + b) / 2.0)) / 255.0)
    return Region(
        name=name,
        box=box,
        mean_rgb=(r, g, b),
        brightness=brightness,
        redness_index=redness,
        contrast=std,
    )

