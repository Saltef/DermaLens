from __future__ import annotations

from scripts.evaluate_subgroups import _exclude_trained_cases, _grouped_split, _metrics


def test_evaluate_subgroups_split_keeps_case_ids_disjoint() -> None:
    rows = [
        {"image_path": "a1.png", "label": "acne", "case_id": "case-a"},
        {"image_path": "a2.png", "label": "acne", "case_id": "case-a"},
        {"image_path": "b1.png", "label": "acne", "case_id": "case-b"},
        {"image_path": "c1.png", "label": "acne", "case_id": "case-c"},
        {"image_path": "d1.png", "label": "dermatitis", "case_id": "case-d"},
        {"image_path": "e1.png", "label": "dermatitis", "case_id": "case-e"},
    ]

    train, val = _grouped_split(rows, val_ratio=0.34, seed=11)

    assert {row["case_id"] for row in train}.isdisjoint({row["case_id"] for row in val})


def test_exclude_trained_cases_removes_seen_model_cases() -> None:
    rows = [
        {"image_path": "a.png", "label": "acne", "case_id": "seen-case"},
        {"image_path": "b.png", "label": "acne", "case_id": "new-case"},
    ]

    kept, excluded = _exclude_trained_cases(rows, {"seen-case"})

    assert excluded == 1
    assert kept == [{"image_path": "b.png", "label": "acne", "case_id": "new-case"}]


def test_metrics_reports_low_per_class_support() -> None:
    rows = [
        {"actual": "tail", "predicted": "head"},
        {"actual": "head", "predicted": "head"},
        {"actual": "head", "predicted": "head"},
        {"actual": "head", "predicted": "tail"},
    ]

    metrics = _metrics(rows, ["head", "tail"])

    assert metrics["per_class_support"] == {"head": 3, "tail": 1}
    assert metrics["low_support_labels"] == ["head", "tail"]
