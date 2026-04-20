# Dataset Summary

Dataset files are stored in `dataset/` using the flat structure downloaded for the project.

## Scene mapping

| Scene | Image file(s) | Ground Truth | Objective | Raw dimension check | Loading decision |
|---|---|---|---|---|---|
| scene1 | Scene_1.png | GT1.png | Segmenter chat, ciel, sol, arbres | Mismatch | Align in memory during loading |
| scene2 | Scene_2.png | GT2.png | Extraire le disque lumineux | Mismatch | Align in memory during loading |
| scene3 | Scene_3.png | GT3.png | Isoler les routes | Mismatch | Align in memory during loading |
| scene4 | Scene_4_RGB_1.png, Scene_4_D_2.png | GT4.png | Extraire la personne debout | Mismatch | Align in memory during loading |

## File details

### scene1

Objective: Segmenter chat, ciel, sol, arbres

| File | Role | Raw size W x H | Mode | Shape | Dtype | Min | Max | Mask unique values |
|---|---|---:|---|---|---|---:|---:|---|
| `Scene_1.png` | image | 558 x 486 | RGBA | (486, 558, 4) | uint8 | 0 | 255 | - |
| `GT1.png` | ground truth | 580 x 481 | RGBA | (481, 580, 4) | uint8 | 68 | 255 | 68, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, ... |

### scene2

Objective: Extraire le disque lumineux

| File | Role | Raw size W x H | Mode | Shape | Dtype | Min | Max | Mask unique values |
|---|---|---:|---|---|---|---:|---:|---|
| `Scene_2.png` | image | 366 x 349 | RGBA | (349, 366, 4) | uint8 | 0 | 255 | - |
| `GT2.png` | ground truth | 360 x 334 | RGBA | (334, 360, 4) | uint8 | 0 | 255 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, ... |

### scene3

Objective: Isoler les routes

| File | Role | Raw size W x H | Mode | Shape | Dtype | Min | Max | Mask unique values |
|---|---|---:|---|---|---|---:|---:|---|
| `Scene_3.png` | image | 411 x 411 | RGBA | (411, 411, 4) | uint8 | 11 | 255 | - |
| `GT3.png` | ground truth | 375 x 376 | RGBA | (376, 375, 4) | uint8 | 0 | 255 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, ... |

### scene4

Objective: Extraire la personne debout

| File | Role | Raw size W x H | Mode | Shape | Dtype | Min | Max | Mask unique values |
|---|---|---:|---|---|---|---:|---:|---|
| `Scene_4_RGB_1.png` | image | 541 x 526 | RGBA | (526, 541, 4) | uint8 | 0 | 255 | - |
| `Scene_4_D_2.png` | image | 535 x 531 | RGBA | (531, 535, 4) | uint8 | 0 | 255 | - |
| `GT4.png` | ground truth | 507 x 493 | L | (493, 507) | uint8 | 0 | 255 | 0, 255 |

## Loaded shapes after alignment

| Scene | Loaded image shape(s) | Loaded mask shape | Image resize flags | Mask resize flag |
|---|---|---|---|---|
| scene1 | (486, 558, 3) | (486, 558, 4) | [False] | True |
| scene2 | (349, 366, 3) | (349, 366, 4) | [False] | True |
| scene3 | (411, 411, 3) | (411, 411, 4) | [False] | True |
| scene4 | (526, 541, 3), (526, 541, 3) | (526, 541) | [False, True] | True |

## Compatibility decision

- Missing expected files: none.
- Raw Ground Truth dimensions do not match the scene image dimensions.
- Raw files must not be overwritten.
- During loading, scene images are converted to RGB arrays.
- Extra scene images are aligned to the first scene image with bilinear resizing when needed.
- Ground Truth masks are aligned to the first scene image with nearest-neighbor resizing when needed.
- Nearest-neighbor is used for masks to avoid creating invalid interpolated labels.
