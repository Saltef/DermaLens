from __future__ import annotations

import random

import pytest

from scripts.prepare_imagefolder import _assert_no_group_overlap, _grouped_train_val_split


def test_grouped_split_keeps_case_images_together() -> None:
    rows = [
        {"image_path": "a1.jpg", "label": "acne", "group_id": "case-a"},
        {"image_path": "a2.jpg", "label": "acne", "group_id": "case-a"},
        {"image_path": "b1.jpg", "label": "acne", "group_id": "case-b"},
        {"image_path": "c1.jpg", "label": "acne", "group_id": "case-c"},
    ]

    train, val = _grouped_train_val_split(
        rows,
        val_ratio=0.34,
        rng=random.Random(7),
        allow_image_level_split=False,
    )

    train_groups = {row["group_id"] for row in train}
    val_groups = {row["group_id"] for row in val}
    assert not train_groups & val_groups


def test_group_overlap_assertion_fails_fast() -> None:
    rows = [
        {"image_path": "a1.jpg", "label": "acne", "group_id": "case-a", "split": "train"},
        {"image_path": "a2.jpg", "label": "acne", "group_id": "case-a", "split": "val"},
    ]

    with pytest.raises(ValueError, match="Group leakage detected"):
        _assert_no_group_overlap(rows)
