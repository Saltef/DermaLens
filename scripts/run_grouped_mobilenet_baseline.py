from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

DEFAULT_SEEDS = [42, 7, 13, 21, 84]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain MobileNetV3 on grouped SCIN train folds and evaluate each untouched validation fold. "
            "This is the fair baseline for comparing Derm Foundation or decoupled-head experiments."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", default="models/experiments/grouped_mobilenet_retrained")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--summary-output", default="models/grouped_scin_mobilenet_retrained_baseline_metrics.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", choices=["mobilenet_v3_small", "efficientnet_b0"], default="mobilenet_v3_small")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--select-metric", choices=["val_accuracy", "macro_recall"], default="macro_recall")
    parser.add_argument("--overwrite-splits", action="store_true")
    parser.add_argument("--overwrite-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    processed_root = Path(args.processed_root)
    output_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    split_results = []
    for seed in args.seeds:
        split_dir = processed_root / f"scin_grouped_seed{seed}"
        run_dir = output_root / f"seed{seed}"
        eval_path = run_dir / "eval_metrics.json"

        if args.overwrite_splits and split_dir.exists():
            shutil.rmtree(split_dir)
        if args.overwrite_runs and run_dir.exists():
            shutil.rmtree(run_dir)

        if not (split_dir / "split_audit.json").exists():
            _run(
                [
                    sys.executable,
                    "scripts/prepare_imagefolder.py",
                    "--manifest",
                    args.manifest,
                    "--image-root",
                    args.image_root,
                    "--output",
                    str(split_dir),
                    "--seed",
                    str(seed),
                ]
            )

        if not (run_dir / "skin_classifier.onnx").exists():
            _run(
                [
                    sys.executable,
                    "scripts/train_export_onnx.py",
                    "--data-dir",
                    str(split_dir),
                    "--output-dir",
                    str(run_dir),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--model",
                    args.model,
                    "--class-weights",
                    args.class_weights,
                    "--select-metric",
                    args.select_metric,
                    "--seed",
                    str(seed),
                ]
            )

        if not eval_path.exists():
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_onnx.py",
                    "--model",
                    str(run_dir / "skin_classifier.onnx"),
                    "--label-map",
                    str(run_dir / "label_map.json"),
                    "--data-dir",
                    str(split_dir / "val"),
                    "--output",
                    str(eval_path),
                ]
            )

        metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        split_audit = json.loads((split_dir / "split_audit.json").read_text(encoding="utf-8"))
        split_results.append(
            {
                "seed": seed,
                "split_dir": str(split_dir),
                "run_dir": str(run_dir),
                "metrics": metrics,
                "split_audit": split_audit,
            }
        )
        print(
            f"seed={seed} accuracy={metrics['accuracy']:.4f} "
            f"macro_recall={metrics['macro_recall']:.4f} total={metrics['total']}"
        )

    summary = _summary(args, split_results)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "summary "
        f"accuracy={summary['summary']['accuracy']['mean']:.4f}+/-{summary['summary']['accuracy']['std']:.4f} "
        f"macro_recall={summary['summary']['macro_recall']['mean']:.4f}+/-{summary['summary']['macro_recall']['std']:.4f}"
    )
    print(f"wrote {summary_path}")


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def _summary(args: argparse.Namespace, split_results: list[dict]) -> dict:
    metrics = [item["metrics"] for item in split_results]
    low_support_by_seed = {
        str(item["seed"]): item["metrics"].get("low_support_labels", [])
        for item in split_results
    }
    labels = sorted({label for metric in metrics for label in metric.get("per_class_recall", {})})
    return {
        "protocol": (
            "Fair grouped MobileNetV3 baseline: for each seed, build a case-level grouped SCIN split, "
            "train the image model only on that seed's training cases, export ONNX, and evaluate once on "
            "that seed's untouched validation cases."
        ),
        "validity_note": (
            "This artifact is the intended baseline for comparing Derm Foundation and decoupled-head experiments. "
            "It should replace fixed-model grouped diagnostics as the clean modeling comparison once all configured "
            "seeds finish."
        ),
        "config": {
            "manifest": args.manifest,
            "image_root": args.image_root,
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "class_weights": args.class_weights,
            "select_metric": args.select_metric,
            "seeds": args.seeds,
        },
        "summary": {
            "accuracy": _mean_std([metric["accuracy"] for metric in metrics]),
            "macro_recall": _mean_std([metric["macro_recall"] for metric in metrics]),
            "per_class_recall": {
                label: _mean_std(
                    [
                        metric["per_class_recall"][label]
                        for metric in metrics
                        if label in metric.get("per_class_recall", {})
                    ]
                )
                for label in labels
            },
            "per_class_support": {
                label: _mean_std(
                    [
                        metric["per_class_support"][label]
                        for metric in metrics
                        if label in metric.get("per_class_support", {})
                    ]
                )
                for label in labels
            },
            "low_support_by_seed": low_support_by_seed,
        },
        "splits": split_results,
    }


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "values": values,
    }


if __name__ == "__main__":
    main()
