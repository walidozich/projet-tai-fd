# Evaluation Summary

This report consolidates the final selected metric row for each scene and the method comparison tables produced by each scene pipeline.

## Final Scene Metrics

| scene | method | accuracy | mean_iou | mean_f1 | internal_silhouette | internal_davies_bouldin | internal_calinski_harabasz | internal_inertia | evaluation_type | primary_score | tp | tn | fp | fn | precision | recall | f1 | dice | iou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene1 | kmeans_rgb_xy_k4 | 0.6258 | 0.5057 | 0.6429 | 0.3949 | 0.9880 | 232645.4419 | 26379.1294 | multi-class | mean_iou | - | - | - | - | - | - | - | - | - |
| scene2 | high_percentile_94_largest_component | 0.9588 | - | - | - | - | - | - | binary | iou | 1401.0000 | 121064.0000 | 3521.0000 | 1748.0000 | 0.2846 | 0.4449 | 0.3472 | 0.3472 | 0.2100 |
| scene3 | tai_hsv_s20_v150_open | 0.9802 | - | - | - | - | - | - | binary | iou | 1009.0000 | 164570.0000 | 672.0000 | 2670.0000 | 0.6002 | 0.2743 | 0.3765 | 0.3765 | 0.2319 |
| scene4 | kmeans_rgb_extra_xy_k5 | 0.9546 | - | - | - | - | - | - | binary | iou | 29210.0000 | 242449.0000 | 3169.0000 | 9738.0000 | 0.9021 | 0.7500 | 0.8190 | 0.8190 | 0.6935 |

## Evaluation Notes

- Scene 1 is multi-class; `mean_iou` is the primary segmentation score.
- Scenes 2, 3, and 4 are binary; `iou` is the primary segmentation score.
- Binary metrics include TP, TN, FP, FN, accuracy, precision, recall, F1/Dice, and IoU.
- Multi-class metrics include accuracy, precision/recall/F1/IoU per class, and mean IoU.
- Ground Truth masks are loaded only after prediction to avoid data leakage.
- `GT1`, `GT2`, and `GT3` are not clean simple masks and require normalization before evaluation.

## Output Files

- `outputs/metrics/final_scene_metrics.csv`
- `outputs/metrics/final_method_comparison.csv`
- `outputs/metrics/final_evaluation_summary.json`
- `outputs/reports/evaluation_summary.md`
