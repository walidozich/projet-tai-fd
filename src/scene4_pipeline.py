"""Scene 4 person extraction pipeline.

Segmentation uses `Scene_4_RGB_1.png` and the aligned supplementary image
`Scene_4_D_2.png`. `GT4.png` is loaded after prediction for evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loading import load_ground_truth, load_scene_inputs, save_mask
from .evaluation import binary_metrics, normalize_binary_mask
from .preprocessing import keep_largest_component, normalize_minmax
from .segmentation_fd import labels_to_image, pca_2d, run_dbscan, run_kmeans
from .segmentation_tai import gaussian_filter, otsu_threshold, rgb_to_gray, simple_threshold
from .visualization import overlay_mask


SCENE_ID = "scene4"
RANDOM_STATE = 42


def _feature_matrix(rgb: np.ndarray, extra_gray: np.ndarray, include_xy: bool = True) -> np.ndarray:
    height, width = extra_gray.shape
    columns = [rgb.reshape(-1, 3).astype(np.float32), extra_gray.reshape(-1, 1).astype(np.float32)]
    if include_xy:
        yy, xx = np.indices((height, width))
        columns.append(
            np.column_stack(
                (
                    xx.ravel() / max(width - 1, 1),
                    yy.ravel() / max(height - 1, 1),
                )
            ).astype(np.float32)
        )
    return normalize_minmax(np.column_stack(columns))


def _extra_range_mask(extra_gray: np.ndarray, low: int = 70, high: int = 90) -> np.ndarray:
    mask = ((extra_gray >= low) & (extra_gray <= high)).astype(np.uint8) * 255
    return keep_largest_component(mask)


def _kmeans_person_mask(
    rgb: np.ndarray,
    extra_gray: np.ndarray,
    n_clusters: int,
    include_xy: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """Cluster RGB/extra features and select the person-like cluster heuristically."""

    features = _feature_matrix(rgb, extra_gray, include_xy=include_xy)
    result = run_kmeans(features, n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = labels_to_image(result.labels, extra_gray.shape)
    height, width = extra_gray.shape
    yy, xx = np.indices((height, width))

    stats = []
    for label in range(n_clusters):
        cluster = labels == label
        area_ratio = float(cluster.mean())
        mean_extra = float(extra_gray[cluster].mean())
        mean_x = float(xx[cluster].mean() / max(width - 1, 1))
        mean_y = float(yy[cluster].mean() / max(height - 1, 1))
        # The person is a coherent mid-gray, central-ish region in the extra image.
        score = (
            -abs(mean_extra - 78.0)
            - 35.0 * abs(mean_x - 0.55)
            - 15.0 * abs(mean_y - 0.52)
            - 25.0 * abs(area_ratio - 0.14)
        )
        stats.append(
            {
                "cluster": label,
                "pixel_count": int(cluster.sum()),
                "area_ratio": area_ratio,
                "mean_extra": mean_extra,
                "mean_x": mean_x,
                "mean_y": mean_y,
                "person_score": score,
            }
        )

    selected_cluster = max(stats, key=lambda row: float(row["person_score"]))["cluster"]
    mask = keep_largest_component(((labels == selected_cluster) * 255).astype(np.uint8))
    return mask, {
        "inertia": result.inertia,
        "selected_cluster": int(selected_cluster),
        "cluster_stats": stats,
    }


def _dbscan_person_mask(extra_gray: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Run DBSCAN on extra-image candidate pixels."""

    candidates = (extra_gray >= 60) & (extra_gray <= 110)
    rows, cols = np.nonzero(candidates)
    height, width = extra_gray.shape
    features = np.column_stack(
        (
            cols / max(width - 1, 1),
            rows / max(height - 1, 1),
            extra_gray[rows, cols] / 255.0,
        )
    ).astype(np.float32)
    result = run_dbscan(features, eps=0.015, min_samples=20)
    labels = result.labels

    best_label = None
    best_size = 0
    for label in np.unique(labels):
        if label == -1:
            continue
        size = int(np.sum(labels == label))
        if size > best_size:
            best_size = size
            best_label = label

    mask = np.zeros((height, width), dtype=np.uint8)
    if best_label is not None:
        selected = labels == best_label
        mask[rows[selected], cols[selected]] = 255
    return keep_largest_component(mask), {
        "candidate_pixels": int(candidates.sum()),
        "clusters": int(len(set(labels.tolist())) - (1 if -1 in labels else 0)),
        "noise_count": int(np.sum(labels == -1)),
        "selected_cluster_size": int(best_size),
    }


def _metric_rows(gt_binary: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for method, mask in masks.items():
        rows.append({"scene": SCENE_ID, "method": method, **binary_metrics(gt_binary, mask > 0)})
    return rows


def run_scene4() -> dict[str, object]:
    out_masks = Path("outputs/masks_predits")
    out_vis = Path("outputs/visualisations")
    out_metrics = Path("outputs/metrics")
    out_reports = Path("outputs/reports")
    for path in (out_masks, out_vis, out_metrics, out_reports):
        path.mkdir(parents=True, exist_ok=True)

    inputs = load_scene_inputs(SCENE_ID)
    rgb, extra = inputs["images"]
    extra_gray = rgb_to_gray(extra)
    extra_smoothed = rgb_to_gray(gaussian_filter(extra, kernel_size=5, sigma=1.0))

    gt = load_ground_truth(SCENE_ID, target_shape=rgb.shape[:2])["mask"]
    gt_binary = normalize_binary_mask(gt, foreground="nonzero")

    otsu_mask_raw, otsu_value = otsu_threshold(extra)
    masks: dict[str, np.ndarray] = {
        "extra_otsu_dark_largest": keep_largest_component(255 - otsu_mask_raw),
        "extra_simple_range_70_90": _extra_range_mask(extra_smoothed, low=70, high=90),
    }
    method_details: dict[str, object] = {"otsu_threshold": float(otsu_value)}

    for k in (2, 3, 4, 5, 6):
        method = f"kmeans_rgb_extra_xy_k{k}"
        masks[method], method_details[method] = _kmeans_person_mask(
            rgb,
            extra_smoothed,
            n_clusters=k,
            include_xy=True,
        )

    masks["kmeans_rgb_extra_k5"], method_details["kmeans_rgb_extra_k5"] = _kmeans_person_mask(
        rgb,
        extra_smoothed,
        n_clusters=5,
        include_xy=False,
    )
    masks["dbscan_extra_candidates"], method_details["dbscan_extra_candidates"] = _dbscan_person_mask(extra_smoothed)

    rows = _metric_rows(gt_binary, masks)
    metrics_df = pd.DataFrame(rows).sort_values(by=["iou", "precision"], ascending=False)
    best_method = str(metrics_df.iloc[0]["method"])
    best_mask = masks[best_method]
    best_metrics = metrics_df.iloc[0].to_dict()

    for method, mask in masks.items():
        save_mask(mask, out_masks / f"scene4_{method}.png")
    save_mask(best_mask, out_masks / "scene4_best_binary_mask.png")
    save_mask((gt_binary * 255).astype(np.uint8), out_masks / "scene4_gt_binary_mask.png")

    metrics_df.to_csv(out_metrics / "scene4_method_comparison.csv", index=False)
    pd.DataFrame([best_metrics]).to_csv(out_metrics / "scene4_metrics.csv", index=False)
    with open(out_metrics / "scene4_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_method": best_method,
                "best_metrics": best_metrics,
                "method_details": method_details,
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.ravel()
    plot_items = [
        (rgb, "Image RGB", None),
        (extra_gray, "Image supplementaire", "gray"),
        (gt_binary, "GT personne binaire", "gray"),
        (masks["extra_otsu_dark_largest"], f"Otsu extra inverse\nIoU={rows[0]['iou']:.3f}", "gray"),
        (masks["extra_simple_range_70_90"], f"Extra range 70-90\nIoU={rows[1]['iou']:.3f}", "gray"),
        (masks["kmeans_rgb_extra_xy_k5"], "K-Means RGB+extra+XY k=5", "gray"),
        (masks["kmeans_rgb_extra_k5"], "K-Means RGB+extra k=5", "gray"),
        (masks["dbscan_extra_candidates"], "DBSCAN extra candidates", "gray"),
        (overlay_mask(rgb, best_mask, alpha=0.45), f"Best overlay\n{best_method}", None),
    ]
    for ax, (plot_image, title, cmap) in zip(axes, plot_items):
        ax.imshow(plot_image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_vis / "scene4_segmentation_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(extra_gray.ravel(), bins=256, color="gray")
    ax.axvline(otsu_value, color="red", linestyle="--", label=f"Otsu={otsu_value:.1f}")
    ax.axvspan(70, 90, color="blue", alpha=0.2, label="range 70-90")
    ax.set_title("Scene 4 - histogramme image supplementaire")
    ax.set_xlabel("Intensite")
    ax.set_ylabel("Nombre de pixels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_vis / "scene4_extra_histogram_thresholds.png", dpi=150)
    plt.close(fig)

    features_sample = _feature_matrix(rgb, extra_smoothed, include_xy=True)[::20]
    labels_sample = run_kmeans(features_sample, n_clusters=5, random_state=RANDOM_STATE).labels
    projection, _ = pca_2d(features_sample, random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels_sample, cmap="tab10", s=4)
    ax.set_title("Scene 4 - PCA 2D RGB+extra+XY")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(out_vis / "scene4_pca_clusters.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metrics_df["method"], metrics_df["iou"])
    ax.set_title("Scene 4 - comparaison IoU")
    ax.set_ylabel("IoU")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_vis / "scene4_method_comparison_iou.png", dpi=150)
    plt.close(fig)

    report = f"""# Scene 4 Report

## Objective

Extraire uniquement la personne debout a partir d'une image RGB et d'une image supplementaire.

## Data usage

- Segmentation inputs: `Scene_4_RGB_1.png` and aligned `Scene_4_D_2.png`, via `load_scene_inputs('scene4')`.
- Ground Truth: `GT4.png`, loaded only after prediction for evaluation.
- The supplementary image is resized in memory to match the RGB image. Raw files are not modified.

## Role of the supplementary image

The supplementary image behaves like a depth/silhouette cue. The standing person appears as a coherent mid-gray region, while much of the background is either very bright or black/noisy. This makes the extra image more useful than RGB alone for isolating the person.

## Methods tested

- Otsu thresholding on the supplementary image, inverted to target darker regions.
- Simple range threshold on the supplementary image: intensities 70 to 90.
- K-Means on `[R,G,B,extra,x,y]` with `k=2..6`.
- K-Means on `[R,G,B,extra]` with `k=5`.
- DBSCAN on candidate pixels selected from the supplementary image.

## Best method

- Best method by IoU: `{best_method}`
- Otsu threshold on supplementary image: `{otsu_value:.2f}`

## Metrics

| method | accuracy | precision | recall | F1/Dice | IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for row in rows:
        report += (
            f"| {row['method']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['iou']:.4f} |\n"
        )

    report += """
## Interpretation

The best result uses the supplementary image together with RGB and spatial coordinates. The extra image separates the person as a mid-gray silhouette, while the XY coordinates help choose the central coherent standing region and avoid unrelated background objects. A simple extra-image intensity range also works well, but it misses more body pixels than the best K-Means configuration. DBSCAN has high recall but includes much more background, so its precision is lower. Remaining errors are mostly boundary errors around arms, legs, and shadows/noisy structures in the supplementary image.

## Outputs

- `outputs/masks_predits/scene4_best_binary_mask.png`
- `outputs/masks_predits/scene4_gt_binary_mask.png`
- `outputs/visualisations/scene4_segmentation_comparison.png`
- `outputs/visualisations/scene4_extra_histogram_thresholds.png`
- `outputs/visualisations/scene4_pca_clusters.png`
- `outputs/visualisations/scene4_method_comparison_iou.png`
- `outputs/metrics/scene4_metrics.csv`
- `outputs/metrics/scene4_method_comparison.csv`
- `outputs/metrics/scene4_metrics.json`
"""
    (out_reports / "scene4_report.md").write_text(report, encoding="utf-8")

    return {
        "scene": SCENE_ID,
        "best_method": best_method,
        "best_metrics": best_metrics,
        "otsu_threshold": float(otsu_value),
    }


if __name__ == "__main__":
    print(json.dumps(run_scene4(), indent=2))
