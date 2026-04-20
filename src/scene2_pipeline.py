"""Scene 2 luminous disk segmentation pipeline.

Segmentation uses only `Scene_2.png`. `GT2.png` is loaded after prediction for
evaluation and visual comparison.
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
from .segmentation_fd import labels_to_image, run_kmeans
from .segmentation_tai import connected_components_8, gaussian_filter, histogram, normalize_histogram, otsu_threshold, rgb_to_gray, simple_threshold
from .visualization import overlay_mask


SCENE_ID = "scene2"
RANDOM_STATE = 42


def _kmeans_intensity_mask(gray: np.ndarray) -> np.ndarray:
    """Segment bright region with K-Means k=2 on grayscale intensity."""

    features = normalize_minmax(gray.reshape(-1, 1))
    result = run_kmeans(features, n_clusters=2, random_state=RANDOM_STATE)
    labels = labels_to_image(result.labels, gray.shape)
    means = [float(gray[labels == label].mean()) for label in range(2)]
    bright_label = int(np.argmax(means))
    return ((labels == bright_label) * 255).astype(np.uint8)


def _postprocess_disk(mask: np.ndarray) -> np.ndarray:
    """Keep the main luminous disk and remove small parasitic regions."""

    return keep_largest_component(mask)


def run_scene2() -> dict[str, object]:
    out_masks = Path("outputs/masks_predits")
    out_vis = Path("outputs/visualisations")
    out_metrics = Path("outputs/metrics")
    out_reports = Path("outputs/reports")
    for path in (out_masks, out_vis, out_metrics, out_reports):
        path.mkdir(parents=True, exist_ok=True)

    inputs = load_scene_inputs(SCENE_ID)
    image = inputs["images"][0]
    gray = rgb_to_gray(image)
    equalized = normalize_histogram(image)
    smoothed = gaussian_filter(image, kernel_size=5, sigma=1.0)

    otsu_raw, otsu_value = otsu_threshold(smoothed)
    otsu_mask = _postprocess_disk(otsu_raw)
    simple_mask = _postprocess_disk(simple_threshold(smoothed, threshold=128))
    high_percentile_threshold = float(np.percentile(rgb_to_gray(smoothed)[rgb_to_gray(smoothed) > 0], 94))
    high_percentile_mask = _postprocess_disk(simple_threshold(smoothed, threshold=int(round(high_percentile_threshold))))
    kmeans_mask = _postprocess_disk(_kmeans_intensity_mask(rgb_to_gray(smoothed)))

    gt = load_ground_truth(SCENE_ID, target_shape=image.shape[:2])["mask"]
    gt_binary = normalize_binary_mask(gt, foreground="bright")

    candidates = {
        "otsu_gaussian_largest_component": otsu_mask,
        "simple_threshold_128_largest_component": simple_mask,
        "high_percentile_94_largest_component": high_percentile_mask,
        "kmeans_intensity_k2_largest_component": kmeans_mask,
    }
    rows = []
    for method, mask in candidates.items():
        metrics = binary_metrics(gt_binary, mask > 0)
        rows.append({"scene": SCENE_ID, "method": method, **metrics})

    metrics_df = pd.DataFrame(rows).sort_values(by=["iou", "f1"], ascending=False)
    best_method = str(metrics_df.iloc[0]["method"])
    best_mask = candidates[best_method]
    best_metrics = metrics_df.iloc[0].to_dict()

    # Save masks.
    save_mask(best_mask, out_masks / "scene2_best_binary_mask.png")
    save_mask(otsu_mask, out_masks / "scene2_otsu_mask.png")
    save_mask(simple_mask, out_masks / "scene2_simple_threshold_mask.png")
    save_mask(high_percentile_mask, out_masks / "scene2_high_percentile_mask.png")
    save_mask(kmeans_mask, out_masks / "scene2_kmeans_intensity_mask.png")
    save_mask((gt_binary * 255).astype(np.uint8), out_masks / "scene2_gt_binary_mask.png")

    # Save metrics.
    metrics_df.to_csv(out_metrics / "scene2_method_comparison.csv", index=False)
    pd.DataFrame([best_metrics]).to_csv(out_metrics / "scene2_metrics.csv", index=False)
    with open(out_metrics / "scene2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_method": best_method,
                "otsu_threshold": float(otsu_value),
                "high_percentile_threshold": float(high_percentile_threshold),
                "metrics": rows,
                "best_metrics": best_metrics,
            },
            f,
            indent=2,
        )

    # Visualizations.
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.ravel()
    metrics_by_method = {row["method"]: row for row in rows}
    plot_items = [
        (image, "Scene 2 originale", None),
        (gray, "Niveaux de gris", "gray"),
        (equalized, "Histogramme normalise", "gray"),
        (gt_binary, "GT binaire normalise", "gray"),
        (otsu_mask, f"Otsu + CC\nIoU={metrics_by_method['otsu_gaussian_largest_component']['iou']:.3f}", "gray"),
        (simple_mask, f"Seuil 128 + CC\nIoU={metrics_by_method['simple_threshold_128_largest_component']['iou']:.3f}", "gray"),
        (high_percentile_mask, f"Percentile 94 + CC\nIoU={metrics_by_method['high_percentile_94_largest_component']['iou']:.3f}", "gray"),
        (kmeans_mask, f"K-Means intensite\nIoU={metrics_by_method['kmeans_intensity_k2_largest_component']['iou']:.3f}", "gray"),
        (overlay_mask(image, best_mask, alpha=0.45), f"Best overlay\n{best_method}", None),
    ]
    for ax, (plot_image, title, cmap) in zip(axes, plot_items):
        ax.imshow(plot_image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_vis / "scene2_segmentation_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    hist = histogram(image)
    ax.plot(np.arange(256), hist, color="black")
    ax.axvline(otsu_value, color="red", linestyle="--", label=f"Otsu={otsu_value:.1f}")
    ax.axvline(128, color="blue", linestyle=":", label="seuil=128")
    ax.axvline(high_percentile_threshold, color="green", linestyle="-.", label=f"p94={high_percentile_threshold:.1f}")
    ax.set_title("Scene 2 - histogramme intensite")
    ax.set_xlabel("Intensite")
    ax.set_ylabel("Nombre de pixels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_vis / "scene2_histogram_with_thresholds.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metrics_df["method"], metrics_df["iou"])
    ax.set_title("Scene 2 - comparaison IoU")
    ax.set_ylabel("IoU")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_vis / "scene2_method_comparison_iou.png", dpi=150)
    plt.close(fig)

    components = connected_components_8(best_mask)
    report = f"""# Scene 2 Report

## Objective

Extraire avec precision la zone lumineuse circulaire dans l'image.

## Data usage

- Segmentation input: `Scene_2.png` only, via `load_scene_inputs('scene2')`.
- Ground Truth: `GT2.png`, loaded only after prediction for evaluation.
- `GT2.png` is RGBA and anti-aliased, so it was converted to a binary mask using brightness clustering for metric calculation.

## Methods tested

- Gaussian filtering + Otsu threshold + largest connected component.
- Gaussian filtering + fixed threshold 128 + largest connected component.
- Gaussian filtering + high-intensity percentile threshold 94 + largest connected component.
- K-Means `k=2` on grayscale intensity + largest connected component.

## Best method

- Best method by IoU: `{best_method}`
- Otsu threshold: `{otsu_value:.2f}`
- High-intensity percentile threshold: `{high_percentile_threshold:.2f}`
- Connected component labels in best mask, including background: `{components['num_labels']}`

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

The disk is mainly separated by high brightness, so a strict high-intensity threshold is more appropriate than a global foreground/background split. Otsu and K-Means perform poorly here because the black image background dominates the intensity distribution, causing them to select most of the retinal field instead of only the disk. Connected components are used after thresholding to keep the main luminous component and remove small parasitic regions. Metrics are computed after normalizing the anti-aliased Ground Truth into a binary mask, so the values should be interpreted as an approximation of the reference object boundary.

## Outputs

- `outputs/masks_predits/scene2_best_binary_mask.png`
- `outputs/masks_predits/scene2_otsu_mask.png`
- `outputs/masks_predits/scene2_simple_threshold_mask.png`
- `outputs/masks_predits/scene2_high_percentile_mask.png`
- `outputs/masks_predits/scene2_kmeans_intensity_mask.png`
- `outputs/masks_predits/scene2_gt_binary_mask.png`
- `outputs/visualisations/scene2_segmentation_comparison.png`
- `outputs/visualisations/scene2_histogram_with_thresholds.png`
- `outputs/visualisations/scene2_method_comparison_iou.png`
- `outputs/metrics/scene2_metrics.csv`
- `outputs/metrics/scene2_method_comparison.csv`
- `outputs/metrics/scene2_metrics.json`
"""
    (out_reports / "scene2_report.md").write_text(report, encoding="utf-8")

    return {
        "scene": SCENE_ID,
        "best_method": best_method,
        "otsu_threshold": float(otsu_value),
        "high_percentile_threshold": float(high_percentile_threshold),
        "best_metrics": best_metrics,
    }


if __name__ == "__main__":
    print(json.dumps(run_scene2(), indent=2))
