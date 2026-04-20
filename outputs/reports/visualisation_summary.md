# Visualisation Summary

This report verifies that the required visualization outputs were produced.

## Required visualizations covered

- Original images: covered by `scene*_image_mask_overview.png` and scene comparison figures.
- Ground Truth masks: covered by `scene*_image_mask_overview.png` and scene comparison figures.
- Predicted masks: covered by each `scene*_segmentation_comparison.png` figure and `outputs/masks_predits/`.
- Overlay visualizations: covered by each scene segmentation comparison figure.
- Side-by-side comparison: covered by each `scene*_segmentation_comparison.png` figure.
- Histograms: covered by exploration histograms and scene-specific threshold histograms.
- Elbow curve: covered by `scene3_elbow_curve.png`; Scene 1 uses fixed `k=4` from the statement.
- PCA 2D plots: covered by Scene 1, Scene 3, and Scene 4 PCA figures.
- Confusion matrices: generated in this step.

## Confusion matrix figures generated

- `outputs/visualisations/scene1_confusion_matrix.png`
- `outputs/visualisations/scene2_confusion_matrix.png`
- `outputs/visualisations/scene3_confusion_matrix.png`
- `outputs/visualisations/scene4_confusion_matrix.png`
