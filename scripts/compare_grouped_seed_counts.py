from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fair grouped MobileNet and Derm Foundation results across matched seed counts. "
            "The output includes majority-class floors, paired t intervals, exact sign tests, and exact "
            "sign-flip permutation tests."
        )
    )
    parser.add_argument("--mobile-5", default="models/grouped_scin_mobilenet_retrained_baseline_metrics.json")
    parser.add_argument("--derm-5", default="models/grouped_scin_derm_foundation_embedding_metrics.json")
    parser.add_argument("--mobile-10", default="models/grouped_scin_mobilenet_retrained_baseline_10seed_metrics.json")
    parser.add_argument("--derm-10", default="models/grouped_scin_derm_foundation_embedding_10seed_metrics.json")
    parser.add_argument("--mobile-12", default="models/grouped_scin_mobilenet_retrained_baseline_12seed_metrics.json")
    parser.add_argument("--derm-12", default="models/grouped_scin_derm_foundation_embedding_12seed_metrics.json")
    parser.add_argument("--output", default="models/grouped_scin_seed_count_sensitivity_metrics.json")
    return parser.parse_args()


def _series(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _summary_block(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics["summary"]["accuracy"],
        "macro_recall": metrics["summary"]["macro_recall"],
    }


def _majority_baseline(mobile_metrics: dict[str, Any]) -> dict[str, Any]:
    supports = mobile_metrics["summary"]["per_class_support"]["dermatitis_like_irritation"]["values"]
    split_rows = mobile_metrics.get("splits", mobile_metrics.get("split_results", []))
    totals = [sum(split["metrics"]["per_class_support"].values()) for split in split_rows]
    accuracies = [support / total for support, total in zip(supports, totals, strict=True)]
    return {
        "accuracy": _series(accuracies),
        "macro_recall": _series([1.0 / 6.0 for _ in supports]),
    }


def _exact_sign_test(deltas: list[float]) -> dict[str, Any]:
    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    ties = len(deltas) - positive - negative
    n = positive + negative
    extreme = min(positive, negative)
    p_value = min(1.0, 2 * sum(math.comb(n, k) for k in range(extreme + 1)) / (2**n)) if n else 1.0
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "p_value_two_sided": p_value,
    }


def _exact_sign_flip_permutation(deltas: list[float]) -> dict[str, Any]:
    observed = abs(sum(deltas))
    totals = []
    for signs in itertools.product((-1, 1), repeat=len(deltas)):
        totals.append(abs(sum(sign * delta for sign, delta in zip(signs, deltas, strict=True))))
    extreme_or_equal = sum(total >= observed - 1e-12 for total in totals)
    return {
        "p_value_two_sided": extreme_or_equal / len(totals),
        "extreme_or_equal": extreme_or_equal,
        "total_sign_flips": len(totals),
    }


def _paired(derm_values: list[float], mobile_values: list[float]) -> dict[str, Any]:
    deltas = [derm - mobile for derm, mobile in zip(derm_values, mobile_values, strict=True)]
    mean_delta = statistics.mean(deltas)
    std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    stderr = std_delta / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    t_stat, p_value = stats.ttest_rel(derm_values, mobile_values)
    t_critical = stats.t.ppf(0.975, len(deltas) - 1) if len(deltas) > 1 else 0.0
    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "values": deltas,
        "paired_t": float(t_stat),
        "df": len(deltas) - 1,
        "p_value_two_sided": float(p_value),
        "ci95": [mean_delta - t_critical * stderr, mean_delta + t_critical * stderr],
        "exact_sign_test": _exact_sign_test(deltas),
        "exact_sign_flip_permutation": _exact_sign_flip_permutation(deltas),
    }


def _per_class_delta(derm_metrics: dict[str, Any], mobile_metrics: dict[str, Any]) -> dict[str, float]:
    derm_recalls = derm_metrics["summary"]["derm_foundation_linear_probe"]["per_class_recall"]
    mobile_recalls = mobile_metrics["summary"]["per_class_recall"]
    return {
        label: derm_recalls[label]["mean"] - mobile_recalls[label]["mean"]
        for label in sorted(mobile_recalls)
    }


def _compare(seed_count: int, mobile_path: str, derm_path: str) -> dict[str, Any]:
    mobile = _load(mobile_path)
    derm = _load(derm_path)
    derm_summary = derm["summary"]["derm_foundation_linear_probe"]
    return {
        "seeds": mobile["config"]["seeds"],
        "artifacts": {"mobile": mobile_path, "derm_foundation": derm_path},
        "majority_class_dermatitis_baseline": _majority_baseline(mobile),
        "fair_mobile": _summary_block(mobile),
        "derm_foundation": {
            "accuracy": derm_summary["accuracy"],
            "macro_recall": derm_summary["macro_recall"],
        },
        "paired_derm_minus_mobile": {
            "accuracy": _paired(
                derm_summary["accuracy"]["values"],
                mobile["summary"]["accuracy"]["values"],
            ),
            "macro_recall": _paired(
                derm_summary["macro_recall"]["values"],
                mobile["summary"]["macro_recall"]["values"],
            ),
        },
        "per_class_mean_delta": _per_class_delta(derm, mobile),
        "note": (
            f"{seed_count} matched completed seeds. Exact non-parametric tests are included because "
            "the five-seed paired t-test is too assumption-sensitive to carry alone."
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "completed_seed_count_sensitivity",
        "seed_counts": {
            "5": _compare(5, args.mobile_5, args.derm_5),
            "10": _compare(10, args.mobile_10, args.derm_10),
            "12": _compare(12, args.mobile_12, args.derm_12),
        },
        "interpretation": (
            "Increasing from 5 to 10 and 12 matched seeds preserves a large positive paired accuracy "
            "delta for Derm Foundation over fair MobileNet. The five-seed exact tests cannot reach "
            "p<0.05, so the original t-test alone was too fragile. At 10 and 12 seeds, the accuracy "
            "lift is supported by exact sign tests; the macro-recall lift is smaller but also positive "
            "under the expanded matched-seed runs. Class-level gains remain uneven, and the low-support "
            "tail labels are still underpowered."
        ),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.write_text(json.dumps(build_report(args), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
