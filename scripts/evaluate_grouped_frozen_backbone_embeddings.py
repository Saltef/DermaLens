from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_decoupled_logit_head import (  # noqa: E402
    _dedup_like_imagefolder,
    _grouped_split,
    _metrics_from_indices,
    _read_labels,
    _read_manifest,
    _summarize,
)

BACKBONES = [
    "convnext_tiny",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "swin_t",
    "vit_b_16",
    "bit_m_r101x3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generic frozen torchvision embeddings with a nested linear probe on grouped SCIN splits. "
            "This is a control for Derm Foundation that keeps the grouped protocol and probe logic aligned."
        )
    )
    parser.add_argument("--manifest", default="data/raw/scin/face_skin_manifest.csv")
    parser.add_argument("--image-root", default="data/raw/scin/images")
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--backbone", choices=BACKBONES, default="convnext_tiny")
    parser.add_argument("--seeds", default="42,7,13,21,84")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache", default="models/experiments/grouped_convnext_tiny_embeddings.npz")
    parser.add_argument("--output", default="models/grouped_scin_convnext_tiny_embedding_12seed_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = _read_labels(Path(args.label_map))
    rows = _read_manifest(Path(args.manifest))
    try:
        embeddings = _load_or_build_embeddings(args, rows, Path(args.image_root))
    except Exception as exc:
        _write_blocked(output, args, reason=f"{exc.__class__.__name__}: {exc}")
        return
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    split_results = []
    for seed in seeds:
        train_rows, val_rows = _grouped_split(rows, val_ratio=args.val_ratio, seed=seed)
        train_rows, val_rows = _dedup_like_imagefolder(train_rows, val_rows, Path(args.image_root))
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
                f"{args.backbone}_linear_probe": final_metrics,
                "best_calibration_result": best_calibration,
                "calibration_sweep": calibration_sweep,
            }
        )
        print(
            f"seed={seed} acc={final_metrics['accuracy']:.4f} macro={final_metrics['macro_recall']:.4f} "
            f"c={final_metrics['c']}"
        )

    key = f"{args.backbone}_linear_probe"
    payload = {
        "status": "completed",
        "protocol": (
            "Generic frozen torchvision embedding control: extract one embedding per image, train a class-balanced "
            "linear probe, select C on nested grouped calibration data, and report the held-out grouped SCIN fold once."
        ),
        "backbone": args.backbone,
        "embedding_source": _embedding_source(args.backbone),
        "group_key": "case_id",
        "selection_protocol": "Nested C selection; evaluation fold is not used for hyperparameter selection.",
        "seeds": seeds,
        "summary": {key: _summarize(split_results, key, labels)},
        "split_results": split_results,
        "reporting_note": _reporting_note(split_results, key),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def _load_or_build_embeddings(args: argparse.Namespace, rows: list[dict], image_root: Path) -> dict[str, np.ndarray]:
    cache_path = Path(args.cache)
    cached_paths: list[str] = []
    cached_values: list[np.ndarray] = []
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        cached_paths = payload["image_paths"].tolist()
        cached_values = [value.astype(np.float32) for value in payload["embeddings"]]
        cached = {image_path: cached_values[idx] for idx, image_path in enumerate(cached_paths)}
        row_paths = {row["image_path"] for row in rows}
        if row_paths.issubset(cached):
            return cached
        print(f"resuming from {len(cached_paths)} cached embeddings")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = _build_feature_model(args.backbone)
    model = model.to(device).eval()
    image_paths = list(cached_paths)
    values = list(cached_values)
    seen = set(cached_paths)
    batch_images = []
    batch_paths = []
    for row in rows:
        image_path = row["image_path"]
        if image_path in seen:
            continue
        path = image_root / image_path
        if not path.exists():
            continue
        seen.add(image_path)
        with Image.open(path) as image:
            batch_images.append(transform(image.convert("RGB")))
        batch_paths.append(image_path)
        if len(batch_images) >= args.batch_size:
            _append_batch(model, device, batch_images, batch_paths, image_paths, values)
            _write_embedding_cache(cache_path, image_paths, values)
            batch_images = []
            batch_paths = []
            print(f"embedded {len(values)} images")
    if batch_images:
        _append_batch(model, device, batch_images, batch_paths, image_paths, values)
        _write_embedding_cache(cache_path, image_paths, values)

    embeddings = np.vstack(values).astype(np.float32)
    return {image_path: embeddings[idx] for idx, image_path in enumerate(image_paths)}


def _write_embedding_cache(cache_path: Path, image_paths: list[str], values: list[np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image_paths=np.array(image_paths),
        embeddings=np.vstack(values).astype(np.float32),
    )


@torch.no_grad()
def _append_batch(
    model: nn.Module,
    device: torch.device,
    batch_images: list[torch.Tensor],
    batch_paths: list[str],
    image_paths: list[str],
    values: list[np.ndarray],
) -> None:
    batch = torch.stack(batch_images).to(device)
    output = model(batch)
    if output.ndim > 2:
        output = torch.flatten(output, start_dim=1)
    for image_path, embedding in zip(batch_paths, output.detach().cpu().numpy(), strict=True):
        image_paths.append(image_path)
        values.append(embedding.astype(np.float32))


def _build_feature_model(name: str) -> tuple[nn.Module, object]:
    if name == "bit_m_r101x3":
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError as exc:
            raise ImportError(
                "The BiT-M R101x3 control requires timm. Install with `python -m pip install timm`."
            ) from exc

        model = timm.create_model("resnetv2_101x3_bit.goog_in21k_ft_in1k", pretrained=True, num_classes=0)
        config = resolve_data_config({"input_size": (3, 448, 448)}, model=model)
        transform = create_transform(**config)
        return model, transform
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Identity()
        return model, weights.transforms()
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Identity()
        return model, weights.transforms()
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Identity()
        return model, weights.transforms()
    if name == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1
        model = models.swin_t(weights=weights)
        model.head = nn.Identity()
        return model, weights.transforms()
    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        model.heads = nn.Identity()
        return model, weights.transforms()
    raise ValueError(name)


def _embedding_source(backbone: str) -> str:
    if backbone == "bit_m_r101x3":
        return "timm_resnetv2_101x3_bit_goog_in21k_ft_in1k"
    return "torchvision_imagenet1k_weights"


def _reporting_note(split_results: list[dict], key: str) -> str:
    accuracy = [result[key]["accuracy"] for result in split_results]
    macro = [result[key]["macro_recall"] for result in split_results]
    return (
        f"accuracy={statistics.mean(accuracy):.4f}+/-{statistics.stdev(accuracy):.4f}, "
        f"macro_recall={statistics.mean(macro):.4f}+/-{statistics.stdev(macro):.4f}"
    )


def _write_blocked(output: Path, args: argparse.Namespace, *, reason: str) -> None:
    payload = {
        "status": "blocked",
        "protocol": "Generic frozen-backbone grouped SCIN embedding control.",
        "backbone": args.backbone,
        "embedding_source": _embedding_source(args.backbone),
        "reason": reason,
        "next_step": (
            "Install any missing model dependency and ensure the pretrained weights are available locally, "
            "then rerun this script with the same grouped seeds."
        ),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote blocked run artifact to {output}")


if __name__ == "__main__":
    main()
