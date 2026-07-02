from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize training and evaluation JSON files.")
    parser.add_argument("--experiments-dir", default="models/experiments")
    parser.add_argument("--output", default="models/experiments/summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.experiments_dir)
    rows = []
    for metrics_path in sorted(root.glob("**/training_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        best = max(payload.get("history", []), key=lambda item: item.get("val_accuracy", 0), default={})
        rows.append(
            {
                "experiment": metrics_path.parent.name,
                "source": "training",
                "accuracy": best.get("val_accuracy", 0),
                "macro_recall": best.get("macro_recall", 0),
                "epoch": best.get("epoch", ""),
            }
        )
    for eval_path in sorted(root.glob("**/eval_*.json")):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": eval_path.parent.name,
                "source": eval_path.stem,
                "accuracy": payload.get("accuracy", 0),
                "macro_recall": payload.get("macro_recall", 0),
                "epoch": "",
            }
        )

    lines = [
        "# Experiment Summary",
        "",
        "| Experiment | Source | Accuracy | Macro Recall | Epoch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: item["accuracy"], reverse=True):
        lines.append(
            f"| {row['experiment']} | {row['source']} | "
            f"{row['accuracy']:.4f} | {row['macro_recall']:.4f} | {row['epoch']} |"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
