from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
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
    face_box = detect_face_box(image)
    left, top, right, bottom = face_box
    face_width = right - left
    face_height = bottom - top

    boxes = {
        "forehead": _relative_box(left, top, face_width, face_height, 0.28, 0.08, 0.72, 0.28, width, height),
        "left_cheek": _relative_box(left, top, face_width, face_height, 0.16, 0.34, 0.44, 0.65, width, height),
        "right_cheek": _relative_box(left, top, face_width, face_height, 0.56, 0.34, 0.84, 0.65, width, height),
        "nose": _relative_box(left, top, face_width, face_height, 0.42, 0.30, 0.58, 0.62, width, height),
        "chin": _relative_box(left, top, face_width, face_height, 0.34, 0.64, 0.66, 0.88, width, height),
    }
    return [_summarize_region(name, image.crop(box), box) for name, box in boxes.items()]


def detect_face_box(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    fallback = _box(width, height, 0.10, 0.02, 0.90, 0.98)

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return fallback

        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        min_side = max(32, min(width, height) // 6)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
    except Exception:
        return fallback

    if len(faces) == 0:
        return fallback

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    pad_x = round(w * 0.18)
    pad_y = round(h * 0.25)
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(width, x + w + pad_x),
        min(height, y + h + pad_y),
    )


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


def _relative_box(
    left: int,
    top: int,
    width: int,
    height: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, round(left + width * x1)),
        max(0, round(top + height * y1)),
        min(image_width, round(left + width * x2)),
        min(image_height, round(top + height * y2)),
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
