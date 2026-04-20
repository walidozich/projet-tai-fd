# Scene 3 Report

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

- Best method by IoU: `tai_hsv_s20_v150_open`
- Detected elbow k: `4`

## Metrics

| method | accuracy | precision | recall | F1/Dice | IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| tai_hsv_s20_v150_open | 0.9802 | 0.6002 | 0.2743 | 0.3765 | 0.2319 |
| tai_hsv_s20_v140_open | 0.9745 | 0.3902 | 0.3009 | 0.3398 | 0.2047 |
| tai_hsv_recall_s25_v80_200_close | 0.8810 | 0.0742 | 0.3890 | 0.1246 | 0.0664 |
| kmeans_road_features_k3 | 0.7984 | 0.0406 | 0.3648 | 0.0731 | 0.0379 |
| kmeans_road_features_k4 | 0.8252 | 0.0469 | 0.3640 | 0.0831 | 0.0434 |
| kmeans_road_features_k5 | 0.8842 | 0.0403 | 0.1895 | 0.0665 | 0.0344 |
| dbscan_candidate_pixels | 0.9599 | 0.2236 | 0.3400 | 0.2698 | 0.1559 |

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
