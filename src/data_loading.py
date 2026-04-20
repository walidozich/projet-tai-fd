"""Dataset loading helpers.

Raw files are never modified. When a Ground Truth mask does not match the
scene image size, it is aligned in memory using nearest-neighbor resizing.
Nearest-neighbor is required for masks because bilinear interpolation would
create invalid intermediate label values.

Important project rule:
- Segmentation algorithms must use only `load_scene_inputs`.
- Ground Truth masks must be loaded only for evaluation/visual comparison.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .data_config import DATASET_DIR, SCENES


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8 array."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_image(path: str | Path, rgb: bool = True) -> np.ndarray:
    """Load an image as a NumPy array.

    If `rgb` is true, the image is converted to RGB. Otherwise, its original
    mode is preserved.
    """

    with Image.open(path) as image:
        if rgb:
            image = image.convert("RGB")
        return np.asarray(image.copy())


def load_mask_raw(path: str | Path) -> np.ndarray:
    """Load a mask without changing its color mode."""

    with Image.open(path) as mask:
        return np.asarray(mask.copy())


def load_mask(path: str | Path) -> np.ndarray:
    """Load a segmentation mask without changing labels or channels."""

    return load_mask_raw(path)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save a mask array to disk, creating parent directories if needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def resize_mask_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D or 3D mask to `(height, width)` using nearest-neighbor."""

    target_height, target_width = target_shape
    pil_mask = Image.fromarray(mask)
    resized = pil_mask.resize((target_width, target_height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized)


def resize_image_bilinear(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize an RGB image to `(height, width)` using bilinear interpolation."""

    target_height, target_width = target_shape
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized)


def _get_scene_config(scene_id: str):
    """Return scene config or raise a clear error."""

    if scene_id not in SCENES:
        raise KeyError(f"Unknown scene id: {scene_id}")
    return SCENES[scene_id]


def load_scene_inputs(scene_id: str) -> dict[str, object]:
    """Load only the input image(s) for a scene.

    This is the correct loader for preprocessing and segmentation. It does not
    return the Ground Truth mask, which prevents accidental data leakage.

    Returns a dictionary containing:
    - `config`: scene metadata
    - `images`: list of RGB arrays
    - `images_were_resized`: resize flags for images after alignment to the first image
    - `target_shape`: shape `(height, width)` used by predictions and evaluation
    """

    config = _get_scene_config(scene_id)
    images = [load_rgb_image(DATASET_DIR / image_file) for image_file in config.image_files]

    target_shape = images[0].shape[:2]
    aligned_images = [images[0]]
    images_were_resized = [False]
    for image in images[1:]:
        image_was_resized = image.shape[:2] != target_shape
        images_were_resized.append(image_was_resized)
        aligned_images.append(resize_image_bilinear(image, target_shape) if image_was_resized else image)

    return {
        "config": config,
        "images": aligned_images,
        "images_were_resized": images_were_resized,
        "target_shape": target_shape,
    }


def load_ground_truth(
    scene_id: str,
    target_shape: tuple[int, int] | None = None,
    align: bool = True,
) -> dict[str, object]:
    """Load the Ground Truth mask for evaluation or visual comparison only."""

    config = _get_scene_config(scene_id)
    mask = load_mask_raw(DATASET_DIR / config.mask_file)

    mask_was_resized = target_shape is not None and mask.shape[:2] != target_shape
    if align and target_shape is not None and mask_was_resized:
        mask = resize_mask_nearest(mask, target_shape)

    return {
        "config": config,
        "mask": mask,
        "mask_was_resized": mask_was_resized,
    }


def load_scene(scene_id: str, align_mask: bool = True) -> dict[str, object]:
    """Load scene input(s) and Ground Truth together for exploration only.

    Do not use this function inside segmentation algorithms. Use
    `load_scene_inputs` for segmentation and `load_ground_truth` for metrics.
    """

    inputs = load_scene_inputs(scene_id)
    ground_truth = load_ground_truth(
        scene_id,
        target_shape=inputs["target_shape"],
        align=align_mask,
    )

    return {
        "config": inputs["config"],
        "images": inputs["images"],
        "mask": ground_truth["mask"],
        "images_were_resized": inputs["images_were_resized"],
        "mask_was_resized": ground_truth["mask_was_resized"],
        "target_shape": inputs["target_shape"],
    }


def validate_dataset_files(dataset_dir: str | Path = DATASET_DIR) -> list[Path]:
    """Return missing dataset files, if any."""

    dataset_dir = Path(dataset_dir)
    expected_files: list[Path] = []
    for scene in SCENES.values():
        expected_files.extend(dataset_dir / image_file for image_file in scene.image_files)
        expected_files.append(dataset_dir / scene.mask_file)
    return [path for path in expected_files if not path.exists()]
