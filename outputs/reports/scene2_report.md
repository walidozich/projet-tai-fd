# Scene 2 Report

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

- Best method by IoU: `high_percentile_94_largest_component`
- Otsu threshold: `51.00`
- High-intensity percentile threshold: `168.00`
- Connected component labels in best mask, including background: `2`

## Metrics

| method | accuracy | precision | recall | F1/Dice | IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| otsu_gaussian_largest_component | 0.2968 | 0.0226 | 0.6513 | 0.0437 | 0.0223 |
| simple_threshold_128_largest_component | 0.8457 | 0.0991 | 0.6500 | 0.1720 | 0.0941 |
| high_percentile_94_largest_component | 0.9588 | 0.2846 | 0.4449 | 0.3472 | 0.2100 |
| kmeans_intensity_k2_largest_component | 0.2968 | 0.0226 | 0.6513 | 0.0437 | 0.0223 |

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
