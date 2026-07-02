from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Derm Foundation embeddings with a nested linear probe on grouped SCIN splits."
    )
    parser.add_argument("--manifest", default="data/raw/scin/face_skin_manifest.csv")
    parser.add_argument("--image-root", default="data/raw/scin/images")
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--foundation-model", default="google/derm-foundation")
    parser.add_argument("--seeds", default="42,7,13,21,84")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--cache", default="models/experiments/derm_foundation_embeddings.npz")
    parser.add_argument("--output", default="models/grouped_scin_derm_foundation_embedding_metrics.json")
    return parser.parse_args()


def main() -> None:
    from scripts.evaluate_decoupled_logit_head import (
        _dedup_like_imagefolder,
        _grouped_split,
        _metrics_from_indices,
        _read_labels,
        _read_manifest,
        _summarize,
    )

    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = _read_labels(Path(args.label_map))
    manifest = Path(args.manifest)
    image_root = Path(args.image_root)
    if not manifest.exists() or not image_root.exists():
        _write_blocked(
            output,
            args,
            reason=(
                "Local SCIN manifest/images are not present in this checkout. A direct access attempt to "
                "google/derm-foundation also returned a gated-repo 401 without authenticated terms acceptance."
            ),
            next_step=(
                "Restore data/raw/scin locally, accept the google/derm-foundation terms on Hugging Face, "
                "run `huggingface-cli login`, then rerun this script."
            ),
        )
        return

    rows = _read_manifest(manifest)
    try:
        embeddings = _load_or_build_embeddings(args, rows, image_root)
    except Exception as exc:
        _write_blocked(
            output,
            args,
            reason=f"{exc.__class__.__name__}: {exc}",
            next_step=(
                "Accept the google/derm-foundation terms on Hugging Face, run `huggingface-cli login`, "
                "then rerun this script."
            ),
        )
        return

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    split_results = []
    for seed in seeds:
        train_rows, val_rows = _grouped_split(rows, val_ratio=args.val_ratio, seed=seed)
        train_rows, val_rows = _dedup_like_imagefolder(train_rows, val_rows, image_root)
        train_rows = [row for row in train_rows if row["image_path"] in embeddings]
        val_rows = [row for row in val_rows if row["image_path"] in embeddings]
        head_train_rows, calibration_rows = _grouped_split(
            train_rows,
            val_ratio=args.calibration_ratio,
            seed=seed + 10_000,
        )

        x_train = np.vstack([embeddings[row["image_path"]] for row in train_rows])
        y_train = np.array([labels.index(row["label"]) for row in train_rows])
        x_head_train = np.vstack([embeddings[row["image_path"]] for row in head_train_rows])
        y_head_train = np.array([labels.index(row["label"]) for row in head_train_rows])
        x_calibration = np.vstack([embeddings[row["image_path"]] for row in calibration_rows])
        y_calibration = np.array([labels.index(row["label"]) for row in calibration_rows])
        x_val = np.vstack([embeddings[row["image_path"]] for row in val_rows])
        y_val = np.array([labels.index(row["label"]) for row in val_rows])

        best_calibration = None
        calibration_sweep = []
        for c_value in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=seed),
            )
            probe.fit(x_head_train, y_head_train)
            metrics = _metrics_from_indices(y_calibration, probe.predict(x_calibration), labels)
            metrics["c"] = c_value
            calibration_sweep.append(metrics)
            if best_calibration is None or (metrics["macro_recall"], metrics["accuracy"]) > (
                best_calibration["macro_recall"],
                best_calibration["accuracy"],
            ):
                best_calibration = metrics

        assert best_calibration is not None
        final_probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_calibration["c"], class_weight="balanced", max_iter=3000, random_state=seed),
        )
        final_probe.fit(x_train, y_train)
        final_metrics = _metrics_from_indices(y_val, final_probe.predict(x_val), labels)
        final_metrics["c"] = best_calibration["c"]
        split_results.append(
            {
                "seed": seed,
                "train_images": len(train_rows),
                "head_train_images": len(head_train_rows),
                "calibration_images": len(calibration_rows),
                "val_images": len(val_rows),
                "derm_foundation_linear_probe": final_metrics,
                "best_calibration_result": best_calibration,
                "calibration_sweep": calibration_sweep,
            }
        )
        print(
            f"seed={seed} acc={final_metrics['accuracy']:.4f} macro={final_metrics['macro_recall']:.4f} "
            f"c={final_metrics['c']}"
        )

    payload = {
        "status": "completed",
        "protocol": (
            "Derm Foundation embedding experiment: extract 6144-dimensional google/derm-foundation embeddings, "
            "train a class-balanced linear probe, select C on nested grouped calibration data, and report the "
            "held-out grouped SCIN fold once."
        ),
        "foundation_model": args.foundation_model,
        "group_key": "case_id",
        "selection_protocol": "Nested C selection; evaluation fold is not used for hyperparameter selection.",
        "seeds": seeds,
        "summary": {"derm_foundation_linear_probe": _summarize(split_results, "derm_foundation_linear_probe", labels)},
        "split_results": split_results,
        "reporting_note": _reporting_note(split_results, "derm_foundation_linear_probe"),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


def _load_or_build_embeddings(args: argparse.Namespace, rows: list[dict], image_root: Path) -> dict[str, np.ndarray]:
    cache_path = Path(args.cache)
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        image_paths = payload["image_paths"].tolist()
        values = payload["embeddings"]
        return {image_path: values[idx] for idx, image_path in enumerate(image_paths)}

    import tensorflow as tf
    from huggingface_hub import from_pretrained_keras

    model = from_pretrained_keras(args.foundation_model)
    infer = model.signatures["serving_default"]
    image_paths = []
    values = []
    seen = set()
    for row in rows:
        image_path = row["image_path"]
        if image_path in seen:
            continue
        path = image_root / image_path
        if not path.exists():
            continue
        seen.add(image_path)
        example = _image_example(path, tf)
        output = infer(inputs=tf.constant([example]))
        image_paths.append(image_path)
        values.append(output["embedding"].numpy().reshape(-1).astype(np.float32))
        if len(values) % 50 == 0:
            print(f"embedded {len(values)} images")

    embeddings = np.vstack(values)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, image_paths=np.array(image_paths), embeddings=embeddings)
    return {image_path: embeddings[idx] for idx, image_path in enumerate(image_paths)}


def _image_example(path: Path, tf) -> bytes:
    with Image.open(path) as image:
        buf = io.BytesIO()
        image.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC).save(buf, format="PNG")
    return tf.train.Example(
        features=tf.train.Features(
            feature={"image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[buf.getvalue()]))}
        )
    ).SerializeToString()


def _reporting_note(split_results: list[dict], key: str) -> str:
    accuracy = [result[key]["accuracy"] for result in split_results]
    macro = [result[key]["macro_recall"] for result in split_results]
    return (
        f"accuracy={statistics.mean(accuracy):.4f}+/-{statistics.stdev(accuracy):.4f}, "
        f"macro_recall={statistics.mean(macro):.4f}+/-{statistics.stdev(macro):.4f}"
    )


def _write_blocked(output: Path, args: argparse.Namespace, *, reason: str, next_step: str) -> None:
    payload = {
        "status": "blocked",
        "protocol": "Derm Foundation embedding experiment under grouped SCIN protocol.",
        "foundation_model": args.foundation_model,
        "reason": reason,
        "next_step": next_step,
        "source": "https://huggingface.co/google/derm-foundation",
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote blocked run artifact to {output}")


if __name__ == "__main__":
    main()
