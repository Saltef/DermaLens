from __future__ import annotations

import io

from PIL import Image

from api.preprocessing import clean_image, prepare_regions
from api.privacy import safe_storage_name


def test_clean_image_strips_exif() -> None:
    image = Image.new("RGB", (32, 32), "white")
    exif = Image.Exif()
    exif[271] = "camera-maker"
    raw = io.BytesIO()
    image.save(raw, format="JPEG", exif=exif)

    source = Image.open(io.BytesIO(raw.getvalue()))
    cleaned_bytes, _ = clean_image(source)
    cleaned = Image.open(io.BytesIO(cleaned_bytes))

    assert not cleaned.getexif()


def test_safe_storage_name_blocks_path_traversal() -> None:
    assert safe_storage_name("../../My Face Photo!!.jpg") == "my-face-photo"


def test_prepare_regions_stays_inside_bounds() -> None:
    image = Image.new("RGB", (37, 911), "white")

    for region in prepare_regions(image):
        left, top, right, bottom = region.box
        assert 0 <= left < right <= image.width
        assert 0 <= top < bottom <= image.height
