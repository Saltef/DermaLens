from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.analyze_mobilenet_checkpoint_selection import build_report as build_checkpoint_report
from scripts.compare_grouped_seed_counts import _exact_sign_test
from scripts.evaluate_onnx import _build_metrics as _build_onnx_metrics
from scripts.evaluate_subgroups import _exclude_trained_cases, _grouped_split, _metrics
from scripts.run_grouped_mobilenet_baseline import _summary


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


def test_evaluate_onnx_metrics_reports_support_and_low_support_labels() -> None:
    rows = [
        {"actual": "common", "predicted": "common"},
        {"actual": "common", "predicted": "rare"},
        {"actual": "rare", "predicted": "common"},
    ]

    metrics = _build_onnx_metrics(rows, ["common", "rare"])

    assert metrics["total"] == 3
    assert metrics["accuracy"] == 1 / 3
    assert metrics["per_class_support"] == {"common": 2, "rare": 1}
    assert metrics["low_support_labels"] == ["common", "rare"]


def test_grouped_mobilenet_summary_aggregates_metrics() -> None:
    class Args:
        manifest = "manifest.csv"
        image_root = "images"
        model = "mobilenet_v3_small"
        epochs = 1
        batch_size = 8
        class_weights = "balanced"
        select_metric = "macro_recall"
        seeds = [1, 2]

    split_results = [
        {
            "seed": 1,
            "metrics": {
                "accuracy": 0.5,
                "macro_recall": 0.4,
                "per_class_recall": {"head": 0.8, "tail": 0.0},
                "per_class_support": {"head": 20, "tail": 2},
                "low_support_labels": ["tail"],
            },
        },
        {
            "seed": 2,
            "metrics": {
                "accuracy": 0.7,
                "macro_recall": 0.6,
                "per_class_recall": {"head": 0.9, "tail": 0.3},
                "per_class_support": {"head": 18, "tail": 3},
                "low_support_labels": ["tail"],
            },
        },
    ]

    summary = _summary(Args(), split_results)

    assert summary["summary"]["accuracy"]["mean"] == 0.6
    assert summary["summary"]["macro_recall"]["values"] == [0.4, 0.6]
    assert summary["summary"]["per_class_support"]["tail"]["values"] == [2, 3]
    assert summary["summary"]["low_support_by_seed"] == {"1": ["tail"], "2": ["tail"]}


def test_train_export_best_state_copy_does_not_share_storage() -> None:
    torch = pytest.importorskip("torch")
    from scripts.train_export_onnx import _copy_state_dict

    model = torch.nn.Linear(2, 1)
    copied = _copy_state_dict(model)
    original_weight = copied["weight"].clone()

    with torch.no_grad():
        model.weight.add_(10)

    assert torch.equal(copied["weight"], original_weight)
    assert not torch.equal(copied["weight"], model.state_dict()["weight"])


def test_five_seed_exact_sign_test_cannot_clear_point_05() -> None:
    result = _exact_sign_test([0.3, 0.2, 0.1, 0.4, 0.5])

    assert result["positive"] == 5
    assert result["negative"] == 0
    assert result["p_value_two_sided"] == 0.0625


def test_checkpoint_selection_diagnostic_uses_accuracy_history(tmp_path: Path) -> None:
    mobile_path = tmp_path / "mobile.json"
    derm_path = tmp_path / "derm.json"
    run_root = tmp_path / "runs"
    for seed in [1, 2]:
        seed_dir = run_root / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "training_metrics.json").write_text(
            json.dumps(
                {
                    "history": [
                        {"epoch": 1, "val_accuracy": 0.4, "macro_recall": 0.7},
                        {"epoch": 2, "val_accuracy": 0.8, "macro_recall": 0.2},
                    ]
                }
            ),
            encoding="utf-8",
        )
    mobile_path.write_text(
        json.dumps(
            {
                "config": {"seeds": [1, 2]},
                "splits": [
                    {"metrics": {"accuracy": 0.4, "macro_recall": 0.7}},
                    {"metrics": {"accuracy": 0.5, "macro_recall": 0.6}},
                ],
            }
        ),
        encoding="utf-8",
    )
    derm_path.write_text(
        json.dumps(
            {
                "summary": {
                    "derm_foundation_linear_probe": {
                        "accuracy": {"values": [0.9, 0.7], "mean": 0.8, "std": 0.1},
                        "macro_recall": {"values": [0.4, 0.3], "mean": 0.35, "std": 0.05},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        mobile=str(mobile_path),
        derm=str(derm_path),
        run_root=str(run_root),
        output=str(tmp_path / "out.json"),
    )

    report = build_checkpoint_report(args)

    assert report["mobile_history_accuracy_selected_diagnostic"]["accuracy"]["mean"] == 0.8
    assert report["mobile_history_accuracy_selected_diagnostic"]["macro_recall"]["mean"] == 0.2
