# Scene 1 Report

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

| cluster | pixel_count | area_ratio | mean_r | mean_g | mean_b | mean_x | mean_y | semantic_label_heuristic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 54646 | 0.2015 | 79.3983 | 82.5685 | 65.8994 | 0.6421 | 0.3008 | chat |
| 1 | 55063 | 0.2030 | 231.7098 | 235.5877 | 240.0029 | 0.4181 | 0.1407 | ciel |
| 2 | 78107 | 0.2880 | 123.0063 | 128.3731 | 90.9032 | 0.7543 | 0.7103 | arbres |
| 3 | 83372 | 0.3074 | 118.8043 | 123.4020 | 87.0415 | 0.2227 | 0.6709 | sol |

This semantic association is heuristic and based on cluster color/position statistics. Numeric evaluation uses label matching against normalized GT labels, not this heuristic mapping.

## Metrics against normalized GT

- Accuracy: `0.6258`
- Mean IoU: `0.5057`
- Mean F1: `0.6429`
- Internal silhouette: `0.3949`
- Davies-Bouldin: `0.9880`
- Calinski-Harabasz: `232645.4419`
- Inertia/SSE: `26379.1294`

## Method comparison

| method | silhouette | davies_bouldin | calinski_harabasz | inertia |
| --- | --- | --- | --- | --- |
| kmeans_rgb_full | 0.5578 | 0.5543 | 1592672.4950 | 2626.2249 |
| kmeans_rgb_xy_full | 0.3949 | 0.9880 | 232645.4419 | 26379.1294 |
| kmedoids_pam_reduced_sample | 0.3909 | 1.0252 | 821.8099 | 87.4875 |
| agnes_reduced_sample | 0.3508 | 1.1263 | 682.8338 | 99.8922 |

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
