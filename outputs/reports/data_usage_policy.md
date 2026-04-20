# Data Usage Policy

This project separates input data from Ground Truth data to avoid data leakage.

## Inputs used by segmentation

Segmentation and clustering algorithms may use only the scene input image files:

| Scene | Allowed input files |
|---|---|
| scene1 | `Scene_1.png` |
| scene2 | `Scene_2.png` |
| scene3 | `Scene_3.png` |
| scene4 | `Scene_4_RGB_1.png`, `Scene_4_D_2.png` |

Use `load_scene_inputs(scene_id)` for preprocessing and segmentation code.

## Ground Truth usage

Ground Truth files are not allowed as segmentation inputs:

| Scene | Ground Truth file | Usage |
|---|---|---|
| scene1 | `GT1.png` | evaluation and visual comparison only |
| scene2 | `GT2.png` | evaluation and visual comparison only |
| scene3 | `GT3.png` | evaluation and visual comparison only |
| scene4 | `GT4.png` | evaluation and visual comparison only |

Use `load_ground_truth(scene_id, target_shape=...)` only after a predicted mask has been produced.

## Alignment rule

Raw dataset files are never overwritten.

- Input scene images are converted to RGB.
- Extra input images, such as scene 4 depth/additional image, are aligned to the first image with bilinear resizing if required.
- Ground Truth masks are aligned to the input image shape with nearest-neighbor resizing if required.
- Nearest-neighbor is mandatory for masks because interpolation would create invalid label values.

## Evaluation rule

The evaluation pipeline must follow this order:

```text
load_scene_inputs(scene_id)
-> preprocess input image(s)
-> produce predicted mask
-> load_ground_truth(scene_id, target_shape=predicted_mask.shape)
-> normalize Ground Truth labels if needed
-> calculate metrics
```
