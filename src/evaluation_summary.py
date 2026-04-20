"""Aggregate segmentation evaluation results across all scenes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


METRICS_DIR = Path("outputs/metrics")
REPORTS_DIR = Path("outputs/reports")


SCENE_FILES = {
    "scene1": {
        "summary": "scene1_metrics.csv",
        "comparison": "scene1_method_comparison.csv",
        "type": "multi-class",
        "primary_score": "mean_iou",
    },
    "scene2": {
        "summary": "scene2_metrics.csv",
        "comparison": "scene2_method_comparison.csv",
        "type": "binary",
        "primary_score": "iou",
    },
    "scene3": {
        "summary": "scene3_metrics.csv",
        "comparison": "scene3_method_comparison.csv",
        "type": "binary",
        "primary_score": "iou",
    },
    "scene4": {
        "summary": "scene4_metrics.csv",
        "comparison": "scene4_method_comparison.csv",
        "type": "binary",
        "primary_score": "iou",
    },
}


def _read_first_row(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty metrics file: {path}")
    return df.iloc[0].to_dict()


def _round_metrics(row: dict[str, object]) -> dict[str, object]:
    rounded = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _markdown_table(df: pd.DataFrame) -> str:
    """Format a DataFrame as Markdown without optional pandas dependencies."""

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("-" if math.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_evaluation_summary() -> dict[str, object]:
    """Generate final evaluation comparison CSV, JSON, and Markdown files."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    final_rows = []
    all_comparisons = []
    missing_files = []

    for scene, config in SCENE_FILES.items():
        summary_path = METRICS_DIR / config["summary"]
        comparison_path = METRICS_DIR / config["comparison"]
        if not summary_path.exists():
            missing_files.append(str(summary_path))
            continue
        if not comparison_path.exists():
            missing_files.append(str(comparison_path))
            continue

        summary = _read_first_row(summary_path)
        summary["scene"] = scene
        summary["evaluation_type"] = config["type"]
        summary["primary_score"] = config["primary_score"]
        final_rows.append(summary)

        comparison = pd.read_csv(comparison_path)
        comparison["scene"] = scene
        all_comparisons.append(comparison)

    if missing_files:
        raise FileNotFoundError("Missing metric files: " + ", ".join(missing_files))

    final_df = pd.DataFrame(final_rows)
    comparison_df = pd.concat(all_comparisons, ignore_index=True, sort=False)

    final_csv = METRICS_DIR / "final_scene_metrics.csv"
    comparison_csv = METRICS_DIR / "final_method_comparison.csv"
    final_json = METRICS_DIR / "final_evaluation_summary.json"
    report_path = REPORTS_DIR / "evaluation_summary.md"

    final_df.to_csv(final_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)

    payload = {
        "scenes": [_round_metrics(row) for row in final_df.to_dict(orient="records")],
        "method_comparison_rows": len(comparison_df),
        "notes": [
            "Scene 1 is multi-class; primary score is mean_iou.",
            "Scenes 2, 3, and 4 are binary; primary score is iou.",
            "GT1, GT2, and GT3 required normalization before evaluation.",
            "All segmentation methods use scene inputs only; Ground Truth is used after prediction for evaluation.",
        ],
    }
    final_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_lines = [
        "# Evaluation Summary",
        "",
        "This report consolidates the final selected metric row for each scene and the method comparison tables produced by each scene pipeline.",
        "",
        "## Final Scene Metrics",
        "",
        _markdown_table(final_df),
        "",
        "## Evaluation Notes",
        "",
        "- Scene 1 is multi-class; `mean_iou` is the primary segmentation score.",
        "- Scenes 2, 3, and 4 are binary; `iou` is the primary segmentation score.",
        "- Binary metrics include TP, TN, FP, FN, accuracy, precision, recall, F1/Dice, and IoU.",
        "- Multi-class metrics include accuracy, precision/recall/F1/IoU per class, and mean IoU.",
        "- Ground Truth masks are loaded only after prediction to avoid data leakage.",
        "- `GT1`, `GT2`, and `GT3` are not clean simple masks and require normalization before evaluation.",
        "",
        "## Output Files",
        "",
        f"- `{final_csv}`",
        f"- `{comparison_csv}`",
        f"- `{final_json}`",
        f"- `{report_path}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "final_csv": str(final_csv),
        "comparison_csv": str(comparison_csv),
        "final_json": str(final_json),
        "report": str(report_path),
        "scene_count": len(final_df),
        "method_comparison_rows": len(comparison_df),
    }


if __name__ == "__main__":
    print(json.dumps(generate_evaluation_summary(), indent=2))
