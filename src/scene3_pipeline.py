"""Scene 3 road segmentation pipeline.

Segmentation uses only `Scene_3.png`. `GT3.png` is loaded after prediction for
evaluation and visual comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loading import load_ground_truth, load_scene_inputs, save_mask
from .evaluation import binary_metrics, normalize_binary_mask
from .preprocessing import normalize_minmax, remove_small_components
from .segmentation_fd import clustering_metrics, detect_elbow_k, elbow_curve, labels_to_image, pca_2d, run_dbscan, run_kmeans
from .segmentation_tai import gaussian_filter, gradient_magnitude_direction, rgb_to_gray, sobel_gradients
from .visualization import overlay_mask


SCENE_ID = "scene3"
RANDOM_STATE = 42


def _road_feature_matrix(image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build `[R,G,B,intensity,gradient,saturation,value,x,y]` road features."""

    gray = rgb_to_gray(image).astype(np.float32)
    grad_x, grad_y = sobel_gradients(image)
    gradient, _ = gradient_magnitude_direction(grad_x, grad_y)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    height, width = gray.shape
    yy, xx = np.indices((height, width))
    features = np.column_stack(
        (
            image.reshape(-1, 3).astype(np.float32),
            gray.ravel(),
            gradient.ravel(),
            saturation.ravel(),
            value.ravel(),
            xx.ravel() / max(width - 1, 1),
            yy.ravel() / max(height - 1, 1),
        )
    )
    return normalize_minmax(features), {
        "gray": gray,
        "gradient": gradient,
        "saturation": saturation,
        "value": value,
    }


def _clean_mask(mask: np.ndarray, min_area: int = 80, open_kernel: int | None = None) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if open_kernel is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return remove_small_components(binary, min_area=min_area)


def _hsv_road_mask(
    image: np.ndarray,
    saturation_max: int = 20,
    value_min: int = 150,
    value_max: int = 255,
    morph: str = "open",
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    mask = ((saturation <= saturation_max) & (value >= value_min) & (value <= value_max)).astype(np.uint8) * 255
    if morph == "open":
        return _clean_mask(mask, min_area=80, open_kernel=3)
    if morph == "close":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return remove_small_components(mask, min_area=80)
    return _clean_mask(mask, min_area=80, open_kernel=None)


def _kmeans_road_mask(
    image: np.ndarray,
    features: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run K-Means and pick the low-saturation/high-value cluster as road."""

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    result = run_kmeans(features, n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = labels_to_image(result.labels, image.shape[:2])

    stats = []
    for label in range(n_clusters):
        cluster = labels == label
        area_ratio = float(cluster.mean())
        mean_s = float(saturation[cluster].mean())
        mean_v = float(value[cluster].mean())
        # Roads are usually bright/desaturated; the area penalty avoids selecting
        # a huge background cluster.
        score = mean_v - 3.0 * mean_s - 80.0 * area_ratio
        stats.append(
            {
                "cluster": label,
                "area_ratio": area_ratio,
                "mean_saturation": mean_s,
                "mean_value": mean_v,
                "road_score": score,
            }
        )

    road_cluster = max(stats, key=lambda row: float(row["road_score"]))["cluster"]
    mask = _clean_mask(((labels == road_cluster) * 255).astype(np.uint8), min_area=80, open_kernel=3)
    return mask, {
        "inertia": result.inertia,
        "selected_cluster": int(road_cluster),
        "cluster_stats": stats,
        "internal_metrics": clustering_metrics(features, labels.ravel(), sample_size=5000, random_state=RANDOM_STATE),
    }


def _dbscan_candidate_mask(image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Run DBSCAN on HSV-selected candidate road pixels."""

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    gray = rgb_to_gray(image)
    candidates = (saturation <= 20) & (value >= 130)
    rows, cols = np.nonzero(candidates)
    height, width = gray.shape
    features = np.column_stack(
        (
            cols / max(width - 1, 1),
            rows / max(height - 1, 1),
            gray[rows, cols] / 255.0,
            saturation[rows, cols] / 255.0,
            value[rows, cols] / 255.0,
        )
    ).astype(np.float32)

    result = run_dbscan(features, eps=0.06, min_samples=10)
    labels = result.labels
    mask = np.zeros((height, width), dtype=np.uint8)
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster = labels == label
        if int(cluster.sum()) >= 50:
            mask[rows[cluster], cols[cluster]] = 255
    return _clean_mask(mask, min_area=80), {
        "candidate_pixels": int(candidates.sum()),
        "noise_count": int(np.sum(labels == -1)),
        "clusters": int(len(set(labels.tolist())) - (1 if -1 in labels else 0)),
    }


def _metric_rows(gt_binary: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for method, mask in masks.items():
        rows.append({"scene": SCENE_ID, "method": method, **binary_metrics(gt_binary, mask > 0)})
    return rows


def run_scene3() -> dict[str, object]:
    out_masks = Path("outputs/masks_predits")
    out_vis = Path("outputs/visualisations")
    out_metrics = Path("outputs/metrics")
    out_reports = Path("outputs/reports")
    for path in (out_masks, out_vis, out_metrics, out_reports):
        path.mkdir(parents=True, exist_ok=True)

    inputs = load_scene_inputs(SCENE_ID)
    image = inputs["images"][0]
    smoothed = gaussian_filter(image, kernel_size=5, sigma=1.0)
    features, channels = _road_feature_matrix(smoothed)
    gray = channels["gray"].astype(np.uint8)
    gradient = channels["gradient"]

    gt = load_ground_truth(SCENE_ID, target_shape=image.shape[:2])["mask"]
    gt_binary = normalize_binary_mask(gt, foreground="bright")

    masks: dict[str, np.ndarray] = {}
    method_details: dict[str, object] = {}
    masks["tai_hsv_s20_v150_open"] = _hsv_road_mask(smoothed, saturation_max=20, value_min=150)
    masks["tai_hsv_s20_v140_open"] = _hsv_road_mask(smoothed, saturation_max=20, value_min=140)
    masks["tai_hsv_recall_s25_v80_200_close"] = _hsv_road_mask(
        smoothed,
        saturation_max=25,
        value_min=80,
        value_max=200,
        morph="close",
    )

    for k in (3, 4, 5):
        method = f"kmeans_road_features_k{k}"
        masks[method], method_details[method] = _kmeans_road_mask(smoothed, features, n_clusters=k)

    masks["dbscan_candidate_pixels"], method_details["dbscan_candidate_pixels"] = _dbscan_candidate_mask(smoothed)

    rows = _metric_rows(gt_binary, masks)
    metrics_df = pd.DataFrame(rows).sort_values(by=["iou", "recall"], ascending=False)
    best_method = str(metrics_df.iloc[0]["method"])
    best_mask = masks[best_method]
    best_metrics = metrics_df.iloc[0].to_dict()

    curve = elbow_curve(features[::8], range(2, 8), random_state=RANDOM_STATE)
    detected_k = detect_elbow_k(curve)

    for method, mask in masks.items():
        save_mask(mask, out_masks / f"scene3_{method}.png")
    save_mask(best_mask, out_masks / "scene3_best_binary_mask.png")
    save_mask((gt_binary * 255).astype(np.uint8), out_masks / "scene3_gt_binary_mask.png")

    metrics_df.to_csv(out_metrics / "scene3_method_comparison.csv", index=False)
    pd.DataFrame([best_metrics]).to_csv(out_metrics / "scene3_metrics.csv", index=False)
    with open(out_metrics / "scene3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_method": best_method,
                "best_metrics": best_metrics,
                "elbow_curve": curve,
                "detected_elbow_k": detected_k,
                "method_details": method_details,
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.ravel()
    plot_items = [
        (image, "Scene 3 originale", None),
        (gray, "Niveaux de gris", "gray"),
        (gradient, "Gradient Sobel", "magma"),
        (gt_binary, "GT routes binaire", "gray"),
        (masks["tai_hsv_s20_v150_open"], f"TAI HSV s20/v150\nIoU={rows[0]['iou']:.3f}", "gray"),
        (masks["tai_hsv_s20_v140_open"], f"TAI HSV s20/v140\nIoU={rows[1]['iou']:.3f}", "gray"),
        (masks["tai_hsv_recall_s25_v80_200_close"], f"HSV recall\nIoU={rows[2]['iou']:.3f}", "gray"),
        (masks["kmeans_road_features_k5"], "K-Means k=5", "gray"),
        (overlay_mask(image, best_mask, alpha=0.45), f"Best overlay\n{best_method}", None),
    ]
    for ax, (plot_image, title, cmap) in zip(axes, plot_items):
        ax.imshow(plot_image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_vis / "scene3_segmentation_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([item["k"] for item in curve], [item["inertia"] for item in curve], marker="o")
    ax.axvline(detected_k, color="red", linestyle="--", label=f"elbow k={detected_k}")
    ax.set_title("Scene 3 - methode du coude K-Means")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertie")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_vis / "scene3_elbow_curve.png", dpi=150)
    plt.close(fig)

    projection, _ = pca_2d(features[::20], random_state=RANDOM_STATE)
    kmeans_labels = run_kmeans(features[::20], n_clusters=5, random_state=RANDOM_STATE).labels
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=kmeans_labels, cmap="tab10", s=4)
    ax.set_title("Scene 3 - PCA 2D des features routes")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(out_vis / "scene3_pca_clusters.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(metrics_df["method"], metrics_df["iou"])
    ax.set_title("Scene 3 - comparaison IoU")
    ax.set_ylabel("IoU")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_vis / "scene3_method_comparison_iou.png", dpi=150)
    plt.close(fig)

    report = f"""# Scene 3 Report

## Objective

Segmenter uniquement les routes a partir d'une vue aerienne.

## Data usage

- Segmentation input: `Scene_3.png` only, via `load_scene_inputs('scene3')`.
- Ground Truth: `GT3.png`, loaded only after prediction for evaluation.
- `GT3.png` is an RGBA red/black road mask and was converted to a binary mask using brightness clustering.

## Image analysis

Roads are generally gray/desaturated and brighter than vegetation or soil. The main difficulty is that roofs and light buildings have similar gray values, so color thresholding alone creates false positives. The road network is also thin, so recall is important but difficult to maximize without increasing false positives.

## Methods tested

- TAI HSV thresholding: low saturation and high value, followed by morphological opening and connected-components cleanup.
- A broader recall-oriented HSV threshold was also tested to recover more road pixels at the cost of false positives.
- K-Means on `[R,G,B,intensity,gradient,saturation,value,x,y]` with `k=3`, `k=4`, and `k=5`.
- DBSCAN on HSV-selected candidate road pixels.
- Elbow method on sampled K-Means features for cluster-count analysis.

## Best method

- Best method by IoU: `{best_method}`
- Detected elbow k: `{detected_k}`

## Metrics

| method | accuracy | precision | recall | F1/Dice | IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for row in rows:
        report += (
            f"| {row['method']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['iou']:.4f} |\n"
        )

    report += f"""
## Interpretation

The best-by-IoU result is still limited because road pixels are visually similar to roofs and other bright gray structures. The stricter HSV threshold improves precision but misses parts of the visible road network, while the broader recall-oriented threshold and DBSCAN recover more road-like pixels at the cost of more false positives. K-Means helps compare cluster structure but does not fully isolate roads because color/gradient/position features overlap with non-road objects. The Ground Truth is a stylized red road mask and appears not perfectly aligned with the visual road centerlines, so small alignment or width differences strongly affect IoU.

## Outputs

- `outputs/masks_predits/scene3_best_binary_mask.png`
- `outputs/masks_predits/scene3_gt_binary_mask.png`
- `outputs/visualisations/scene3_segmentation_comparison.png`
- `outputs/visualisations/scene3_elbow_curve.png`
- `outputs/visualisations/scene3_pca_clusters.png`
- `outputs/visualisations/scene3_method_comparison_iou.png`
- `outputs/metrics/scene3_metrics.csv`
- `outputs/metrics/scene3_method_comparison.csv`
- `outputs/metrics/scene3_metrics.json`
"""
    (out_reports / "scene3_report.md").write_text(report, encoding="utf-8")

    return {
        "scene": SCENE_ID,
        "best_method": best_method,
        "detected_elbow_k": detected_k,
        "best_metrics": best_metrics,
    }


if __name__ == "__main__":
    print(json.dumps(run_scene3(), indent=2))
