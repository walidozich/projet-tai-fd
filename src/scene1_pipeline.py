"""Scene 1 segmentation pipeline.

Segmentation uses only `Scene_1.png`. `GT1.png` is loaded after prediction for
evaluation and visual comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .data_loading import load_ground_truth, load_scene_inputs, save_mask
from .evaluation import match_predicted_labels, multiclass_metrics, normalize_mask_by_color_clustering
from .preprocessing import extract_rgb_features, extract_rgb_xy_features, normalize_minmax, resize_for_clustering
from .segmentation_fd import clustering_metrics, labels_to_image, pca_2d, run_agnes, run_kmeans, run_kmedoids
from .segmentation_tai import gaussian_filter


SCENE_ID = "scene1"
N_CLASSES = 4
RANDOM_STATE = 42
PALETTE = np.array(
    [
        [80, 160, 230],
        [230, 150, 70],
        [170, 135, 85],
        [55, 130, 65],
    ],
    dtype=np.uint8,
)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    """Format a list of dictionaries as a Markdown table without tabulate."""

    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32)
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        if label >= 0:
            output[labels == label] = PALETTE[label % len(PALETTE)]
    return output


def _blend_labels(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a colorized label image with the original image."""

    colorized = _colorize_labels(labels).astype(np.float32)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * colorized
    return np.clip(blended, 0, 255).astype(np.uint8)


def _fill_unassigned_labels(labels: np.ndarray, max_iterations: int = 12) -> np.ndarray:
    """Fill `-1` pixels from neighboring valid labels."""

    cleaned = labels.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max_iterations):
        unassigned = cleaned == -1
        if not np.any(unassigned):
            break
        next_labels = cleaned.copy()
        for label in np.unique(cleaned):
            if label == -1:
                continue
            neighbors = cv2.dilate((cleaned == label).astype(np.uint8), kernel, iterations=1) > 0
            next_labels[unassigned & neighbors] = label
        if np.array_equal(next_labels, cleaned):
            break
        cleaned = next_labels
    cleaned[cleaned == -1] = labels[cleaned == -1]
    return cleaned


def _clean_multiclass_labels(label_image: np.ndarray, min_area: int = 120) -> np.ndarray:
    """Remove tiny connected components in a multi-class label image."""

    cleaned = label_image.copy().astype(np.int32)
    for label in np.unique(label_image):
        binary = (label_image == label).astype(np.uint8)
        num_labels, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_id in range(1, num_labels):
            if stats[component_id, cv2.CC_STAT_AREA] < min_area:
                cleaned[components == component_id] = -1
    return _fill_unassigned_labels(cleaned)


def _cluster_stats(image: np.ndarray, label_image: np.ndarray) -> list[dict[str, object]]:
    rows = []
    height, width = label_image.shape
    yy, xx = np.indices((height, width))
    for label in sorted(np.unique(label_image).tolist()):
        mask = label_image == label
        rgb_mean = image[mask].mean(axis=0)
        rows.append(
            {
                "cluster": int(label),
                "pixel_count": int(mask.sum()),
                "area_ratio": float(mask.mean()),
                "mean_r": float(rgb_mean[0]),
                "mean_g": float(rgb_mean[1]),
                "mean_b": float(rgb_mean[2]),
                "mean_x": float(xx[mask].mean() / max(width - 1, 1)),
                "mean_y": float(yy[mask].mean() / max(height - 1, 1)),
            }
        )
    return rows


def _semantic_mapping(stats: list[dict[str, object]]) -> dict[int, str]:
    """Heuristically name clusters from color and position statistics."""

    remaining = {int(row["cluster"]) for row in stats}
    stats_by = {int(row["cluster"]): row for row in stats}
    mapping: dict[int, str] = {}

    sky = max(
        remaining,
        key=lambda c: float(stats_by[c]["mean_b"]) - float(stats_by[c]["mean_r"]) + (1 - float(stats_by[c]["mean_y"])) * 80,
    )
    mapping[sky] = "ciel"
    remaining.remove(sky)

    trees = max(
        remaining,
        key=lambda c: float(stats_by[c]["mean_g"]) - 0.5 * (float(stats_by[c]["mean_r"]) + float(stats_by[c]["mean_b"])),
    )
    mapping[trees] = "arbres"
    remaining.remove(trees)

    soil = max(remaining, key=lambda c: float(stats_by[c]["mean_y"]))
    mapping[soil] = "sol"
    remaining.remove(soil)

    for cluster in remaining:
        mapping[cluster] = "chat"
    return mapping


def run_scene1() -> dict[str, object]:
    out_masks = Path("outputs/masks_predits")
    out_vis = Path("outputs/visualisations")
    out_metrics = Path("outputs/metrics")
    out_reports = Path("outputs/reports")
    for path in (out_masks, out_vis, out_metrics, out_reports):
        path.mkdir(parents=True, exist_ok=True)

    inputs = load_scene_inputs(SCENE_ID)
    image = inputs["images"][0]
    smoothed = gaussian_filter(image, kernel_size=5, sigma=1.0)

    rgb_features = normalize_minmax(extract_rgb_features(smoothed))
    rgb_xy_features = normalize_minmax(extract_rgb_xy_features(smoothed))

    kmeans_rgb = run_kmeans(rgb_features, n_clusters=N_CLASSES, random_state=RANDOM_STATE)
    kmeans_rgb_xy = run_kmeans(rgb_xy_features, n_clusters=N_CLASSES, random_state=RANDOM_STATE)

    pred_rgb = labels_to_image(kmeans_rgb.labels, image.shape[:2])
    pred_rgb_xy = labels_to_image(kmeans_rgb_xy.labels, image.shape[:2])
    pred_rgb_xy_clean = _clean_multiclass_labels(pred_rgb_xy, min_area=120)

    gt = load_ground_truth(SCENE_ID, target_shape=image.shape[:2])["mask"]
    gt_labels = normalize_mask_by_color_clustering(gt, n_classes=N_CLASSES, random_state=RANDOM_STATE)
    matched_pred, pred_to_gt = match_predicted_labels(gt_labels, pred_rgb_xy_clean, n_classes=N_CLASSES)
    metrics = multiclass_metrics(gt_labels, matched_pred, n_classes=N_CLASSES)
    internal_metrics = clustering_metrics(rgb_xy_features, pred_rgb_xy_clean.ravel(), sample_size=5000, random_state=RANDOM_STATE)

    small_image, scale = resize_for_clustering(smoothed, max_size=64)
    small_features = normalize_minmax(extract_rgb_xy_features(small_image))
    small_sample = small_features[::4]
    kmedoids = run_kmedoids(small_sample, n_clusters=N_CLASSES, random_state=RANDOM_STATE)
    agnes = run_agnes(small_sample, n_clusters=N_CLASSES)
    method_comparison = [
        {
            "method": "kmeans_rgb_full",
            **clustering_metrics(rgb_features, pred_rgb.ravel(), sample_size=5000, random_state=RANDOM_STATE),
        },
        {"method": "kmeans_rgb_xy_full", **internal_metrics},
        {"method": "kmedoids_pam_reduced_sample", **clustering_metrics(small_sample, kmedoids.labels, sample_size=None, random_state=RANDOM_STATE)},
        {"method": "agnes_reduced_sample", **clustering_metrics(small_sample, agnes.labels, sample_size=None, random_state=RANDOM_STATE)},
    ]

    stats = _cluster_stats(image, pred_rgb_xy_clean)
    semantics = _semantic_mapping(stats)
    for row in stats:
        row["semantic_label_heuristic"] = semantics[int(row["cluster"])]

    save_mask(pred_rgb_xy_clean.astype(np.uint8), out_masks / "scene1_kmeans_rgb_xy_labels.png")
    Image.fromarray(_colorize_labels(pred_rgb_xy_clean)).save(out_masks / "scene1_kmeans_rgb_xy_color.png")
    Image.fromarray(_colorize_labels(gt_labels)).save(out_masks / "scene1_gt_4labels_color.png")
    Image.fromarray(_colorize_labels(matched_pred)).save(out_masks / "scene1_kmeans_matched_to_gt_color.png")

    metrics_flat = {
        "scene": SCENE_ID,
        "method": "kmeans_rgb_xy_k4",
        "accuracy": metrics["accuracy"],
        "mean_iou": metrics["mean_iou"],
        "mean_f1": metrics["mean_f1"],
        "internal_silhouette": internal_metrics["silhouette"],
        "internal_davies_bouldin": internal_metrics["davies_bouldin"],
        "internal_calinski_harabasz": internal_metrics["calinski_harabasz"],
        "internal_inertia": internal_metrics["inertia"],
    }
    pd.DataFrame([metrics_flat]).to_csv(out_metrics / "scene1_metrics.csv", index=False)
    pd.DataFrame(metrics["per_class"]).to_csv(out_metrics / "scene1_per_class_metrics.csv", index=False)
    pd.DataFrame(method_comparison).to_csv(out_metrics / "scene1_method_comparison.csv", index=False)
    pd.DataFrame(stats).to_csv(out_metrics / "scene1_cluster_stats.csv", index=False)
    with open(out_metrics / "scene1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": metrics_flat,
                "per_class": metrics["per_class"],
                "confusion_matrix": metrics["confusion_matrix"],
                "pred_to_gt_label_mapping": pred_to_gt,
                "cluster_semantic_mapping_heuristic": {str(k): v for k, v in semantics.items()},
                "method_comparison": method_comparison,
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    images_and_titles = [
        (image, "Scene 1 originale"),
        (gt, "GT1 aligne brut"),
        (_colorize_labels(gt_labels), "GT normalise en 4 labels"),
        (_colorize_labels(pred_rgb_xy_clean), "K-Means RGB+XY k=4"),
        (_colorize_labels(matched_pred), "Prediction appariee au GT"),
        (_blend_labels(image, pred_rgb_xy_clean), "Overlay clusters"),
    ]
    for ax, (plot_image, title) in zip(axes, images_and_titles):
        ax.imshow(plot_image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_vis / "scene1_segmentation_comparison.png", dpi=150)
    plt.close(fig)

    projection, _ = pca_2d(rgb_xy_features[::20], random_state=RANDOM_STATE)
    labels_sample = pred_rgb_xy_clean.ravel()[::20]
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels_sample, cmap="tab10", s=4)
    ax.set_title("Scene 1 - PCA 2D des features RGB+XY")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(out_vis / "scene1_pca_clusters.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    comparison_df = pd.DataFrame(method_comparison)
    ax.bar(comparison_df["method"], comparison_df["inertia"])
    ax.set_title("Scene 1 - comparaison inertie/SSE")
    ax.set_ylabel("Inertie/SSE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_vis / "scene1_method_comparison_inertia.png", dpi=150)
    plt.close(fig)

    rgb_matched, _ = match_predicted_labels(gt_labels, pred_rgb, n_classes=N_CLASSES)
    rgb_metrics = multiclass_metrics(gt_labels, rgb_matched, n_classes=N_CLASSES)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    comparison_images = [
        (image, "Image originale"),
        (_colorize_labels(gt_labels), "GT normalise"),
        (_colorize_labels(pred_rgb), f"K-Means RGB brut mIoU {rgb_metrics['mean_iou']:.3f}"),
        (_colorize_labels(rgb_matched), f"K-Means RGB apparie acc {rgb_metrics['accuracy']:.3f}"),
        (_colorize_labels(pred_rgb_xy_clean), f"K-Means RGB+XY brut mIoU {metrics['mean_iou']:.3f}"),
        (_colorize_labels(matched_pred), f"K-Means RGB+XY apparie acc {metrics['accuracy']:.3f}"),
    ]
    for ax, (plot_image, title) in zip(axes, comparison_images):
        ax.imshow(plot_image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_vis / "scene1_rgb_vs_rgbxy_check.png", dpi=150)
    plt.close(fig)

    report = f"""# Scene 1 Report

## Objective

Segmenter l'image en 4 clusters distincts : chat, ciel, sol, arbres.

## Data usage

- Segmentation input: `Scene_1.png` only, via `load_scene_inputs('scene1')`.
- Ground Truth: `GT1.png`, loaded only after prediction for evaluation.
- `GT1.png` is RGBA with many unique colors, so it was compressed into 4 labels using color clustering for metric calculation.

## Main method

- Preprocessing: Gaussian smoothing with kernel `5x5`, sigma `1.0`.
- Features tested: `[R, G, B]` and `[R, G, B, x, y]`.
- Selected prediction: K-Means with `k=4` on normalized `[R, G, B, x, y]`.

## Heuristic semantic cluster association

{_markdown_table(stats)}

This semantic association is heuristic and based on cluster color/position statistics. Numeric evaluation uses label matching against normalized GT labels, not this heuristic mapping.

## Metrics against normalized GT

- Accuracy: `{metrics_flat['accuracy']:.4f}`
- Mean IoU: `{metrics_flat['mean_iou']:.4f}`
- Mean F1: `{metrics_flat['mean_f1']:.4f}`
- Internal silhouette: `{metrics_flat['internal_silhouette']:.4f}`
- Davies-Bouldin: `{metrics_flat['internal_davies_bouldin']:.4f}`
- Calinski-Harabasz: `{metrics_flat['internal_calinski_harabasz']:.4f}`
- Inertia/SSE: `{metrics_flat['internal_inertia']:.4f}`

## Method comparison

{_markdown_table(method_comparison)}

## Outputs

- `outputs/masks_predits/scene1_kmeans_rgb_xy_labels.png`
- `outputs/masks_predits/scene1_kmeans_rgb_xy_color.png`
- `outputs/masks_predits/scene1_gt_4labels_color.png`
- `outputs/masks_predits/scene1_kmeans_matched_to_gt_color.png`
- `outputs/visualisations/scene1_segmentation_comparison.png`
- `outputs/visualisations/scene1_rgb_vs_rgbxy_check.png`
- `outputs/visualisations/scene1_pca_clusters.png`
- `outputs/visualisations/scene1_method_comparison_inertia.png`
- `outputs/metrics/scene1_metrics.csv`
- `outputs/metrics/scene1_per_class_metrics.csv`
- `outputs/metrics/scene1_method_comparison.csv`
- `outputs/metrics/scene1_cluster_stats.csv`
- `outputs/metrics/scene1_metrics.json`

## Interpretation

K-Means on RGB+XY is used as the principal method because spatial coordinates help form coherent regions, while RGB-only clustering can group similar colors even when they are spatially separate. K-Medoids/PAM and AGNES were tested on a reduced sample to keep runtime and memory reasonable. The metrics should be interpreted with caution because `GT1.png` is not a clean 4-label mask and had to be normalized before comparison.
"""
    (out_reports / "scene1_report.md").write_text(report, encoding="utf-8")

    return metrics_flat


if __name__ == "__main__":
    print(json.dumps(run_scene1(), indent=2))
