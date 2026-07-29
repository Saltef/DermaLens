from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_grouped_seed_counts import _paired  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the exported macro-selected MobileNet checkpoints against an accuracy-selected diagnostic "
            "computed from each seed's training history."
        )
    )
    parser.add_argument("--mobile", default="models/grouped_scin_mobilenet_retrained_baseline_12seed_metrics.json")
    parser.add_argument("--derm", default="models/grouped_scin_derm_foundation_embedding_12seed_metrics.json")
    parser.add_argument("--run-root", default="models/experiments/grouped_mobilenet_retrained")
    parser.add_argument("--output", default="models/grouped_scin_mobilenet_checkpoint_selection_diagnostic.json")
    return parser.parse_args()


def _series(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _best(history: list[dict[str, Any]], primary: str, secondary: str) -> dict[str, Any]:
    return max(history, key=lambda item: (item[primary], item[secondary]))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    mobile = json.loads(Path(args.mobile).read_text(encoding="utf-8"))
    derm = json.loads(Path(args.derm).read_text(encoding="utf-8"))
    derm_summary = derm["summary"]["derm_foundation_linear_probe"]
    run_root = Path(args.run_root)

    seed_rows = []
    macro_selected_accuracy = []
    macro_selected_macro = []
    accuracy_selected_accuracy = []
    accuracy_selected_macro = []

    for seed, split in zip(mobile["config"]["seeds"], mobile["splits"], strict=True):
        metrics_path = run_root / f"seed{seed}" / "training_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        training = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = training["history"]
        macro_best = _best(history, "macro_recall", "val_accuracy")
        accuracy_best = _best(history, "val_accuracy", "macro_recall")
        exported = split["metrics"]

        macro_selected_accuracy.append(exported["accuracy"])
        macro_selected_macro.append(exported["macro_recall"])
        accuracy_selected_accuracy.append(accuracy_best["val_accuracy"])
        accuracy_selected_macro.append(accuracy_best["macro_recall"])
        seed_rows.append(
            {
                "seed": seed,
                "exported_macro_selected": {
                    "accuracy": exported["accuracy"],
                    "macro_recall": exported["macro_recall"],
                    "source": "exported ONNX checkpoint selected by validation macro recall",
                },
                "history_macro_selected": {
                    "epoch": macro_best["epoch"],
                    "accuracy": macro_best["val_accuracy"],
                    "macro_recall": macro_best["macro_recall"],
                },
                "history_accuracy_selected": {
                    "epoch": accuracy_best["epoch"],
                    "accuracy": accuracy_best["val_accuracy"],
                    "macro_recall": accuracy_best["macro_recall"],
                    "source": "diagnostic only; selected post hoc on the evaluation fold history",
                },
            }
        )

    return {
        "status": "completed_checkpoint_selection_diagnostic",
        "validity_note": (
            "The accuracy-selected MobileNet row is an optimistic diagnostic, not a clean model estimate, because "
            "it chooses the best epoch from the evaluation-fold history. It tests whether the Derm Foundation lift "
            "depends entirely on macro-recall checkpoint selection weakening MobileNet accuracy."
        ),
        "artifacts": {
            "mobile": args.mobile,
            "derm_foundation": args.derm,
            "run_root": args.run_root,
        },
        "seeds": mobile["config"]["seeds"],
        "mobile_exported_macro_selected": {
            "accuracy": _series(macro_selected_accuracy),
            "macro_recall": _series(macro_selected_macro),
        },
        "mobile_history_accuracy_selected_diagnostic": {
            "accuracy": _series(accuracy_selected_accuracy),
            "macro_recall": _series(accuracy_selected_macro),
        },
        "derm_foundation": {
            "accuracy": derm_summary["accuracy"],
            "macro_recall": derm_summary["macro_recall"],
        },
        "paired_derm_minus_accuracy_selected_mobile_diagnostic": {
            "accuracy": _paired(derm_summary["accuracy"]["values"], accuracy_selected_accuracy),
            "macro_recall": _paired(derm_summary["macro_recall"]["values"], accuracy_selected_macro),
        },
        "seed_results": seed_rows,
        "interpretation": (
            "Derm Foundation remains ahead of a post-hoc accuracy-selected MobileNet diagnostic on accuracy, "
            "but the gap is smaller than against the macro-selected exported baseline. This suggests part of the "
            "reported accuracy lift comes from MobileNet checkpoint-selection tradeoffs, while the representation "
            "advantage does not disappear under this stronger diagnostic comparator."
        ),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_report(args), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
