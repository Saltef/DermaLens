from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps
from PIL.Image import DecompressionBombError

from api.classifier import classify_regions
from api.preprocessing import clean_image, prepare_regions
from api.privacy import safe_storage_name

Image.MAX_IMAGE_PIXELS = 24_000_000


def analyze_upload(
    raw: bytes,
    *,
    original_filename: str,
    save_uploads: bool,
    private_data_root: Path,
) -> dict:
    image = _load_image(raw)
    cleaned_bytes, cleaned = clean_image(image)
    regions = prepare_regions(cleaned)
    result = classify_regions(cleaned, regions)

    image_hash = hashlib.sha256(cleaned_bytes).hexdigest()
    stored = None
    if save_uploads:
        stored = _persist_clean_image(
            cleaned_bytes,
            result,
            original_filename=original_filename,
            image_hash=image_hash,
            private_data_root=private_data_root,
        )

    return {
        "privacy": {
            "exif_stripped": True,
            "image_saved": save_uploads,
            "stored_record": stored,
            "sha256": image_hash,
        },
        "image": {
            "width": cleaned.width,
            "height": cleaned.height,
            "mode": cleaned.mode,
        },
        "quality": result["quality"],
        "model": result["model"],
        "regions": [region.to_public_dict() for region in regions],
        "findings": result["findings"],
        "review_flags": result["review_flags"],
        "limitations": _limitations_for_model(result["model"]),
    }


def _limitations_for_model(model: dict) -> list[str]:
    limitations = [
        "Results are screening observations, not a medical diagnosis.",
        "Lighting, makeup, filters, camera sharpening, and skin tone representation can affect output.",
    ]
    if model["source"] == "heuristic":
        limitations.insert(0, "This MVP is using a heuristic fallback because no trained ONNX model was found.")
    else:
        limitations.insert(0, "This MVP is using the local ONNX classifier found in the models directory.")
    return limitations


def _load_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except DecompressionBombError as exc:
        raise ValueError("Image dimensions are too large. Use a normal phone photo under 24 megapixels.") from exc
    except Exception as exc:
        raise ValueError("Could not decode image. Try a JPEG or PNG photo.") from exc


def _persist_clean_image(
    cleaned_bytes: bytes,
    result: dict,
    *,
    original_filename: str,
    image_hash: str,
    private_data_root: Path,
) -> str:
    private_data_root.mkdir(parents=True, exist_ok=True)
    stem = safe_storage_name(original_filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_dir = private_data_root / f"{timestamp}-{stem}-{image_hash[:10]}"
    record_dir.mkdir(parents=True, exist_ok=False)
    (record_dir / "image.exif-stripped.jpg").write_bytes(cleaned_bytes)
    (record_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(record_dir.name)
