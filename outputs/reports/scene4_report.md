# Scene 4 Report

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

- Best method by IoU: `kmeans_rgb_extra_xy_k5`
- Otsu threshold on supplementary image: `130.00`

## Metrics

| method | accuracy | precision | recall | F1/Dice | IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| extra_otsu_dark_largest | 0.5761 | 0.2408 | 0.9744 | 0.3862 | 0.2393 |
| extra_simple_range_70_90 | 0.9390 | 0.7965 | 0.7448 | 0.7698 | 0.6257 |
| kmeans_rgb_extra_xy_k2 | 0.8747 | 0.5310 | 0.7222 | 0.6120 | 0.4409 |
| kmeans_rgb_extra_xy_k3 | 0.6293 | 0.0436 | 0.0816 | 0.0568 | 0.0292 |
| kmeans_rgb_extra_xy_k4 | 0.6300 | 0.0437 | 0.0816 | 0.0570 | 0.0293 |
| kmeans_rgb_extra_xy_k5 | 0.9546 | 0.9021 | 0.7500 | 0.8190 | 0.6935 |
| kmeans_rgb_extra_xy_k6 | 0.9457 | 0.8987 | 0.6798 | 0.7741 | 0.6314 |
| kmeans_rgb_extra_k5 | 0.8776 | 0.5645 | 0.4642 | 0.5095 | 0.3418 |
| dbscan_extra_candidates | 0.8332 | 0.4461 | 0.9065 | 0.5980 | 0.4265 |

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
