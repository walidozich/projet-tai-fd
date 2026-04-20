"""Dataset loading helpers.

Raw files are never modified. When a Ground Truth mask does not match the
scene image size, it is aligned in memory using nearest-neighbor resizing.
Nearest-neighbor is required for masks because bilinear interpolation would
create invalid intermediate label values.
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


def load_scene(scene_id: str, align_mask: bool = True) -> dict[str, object]:
    """Load scene image(s) and Ground Truth mask.

    Returns a dictionary containing:
    - `config`: scene metadata
    - `images`: list of RGB arrays
    - `mask`: raw or aligned Ground Truth mask
    - `images_were_resized`: resize flags for images after alignment to the first image
    - `mask_was_resized`: whether mask resizing was applied
    """

    if scene_id not in SCENES:
        raise KeyError(f"Unknown scene id: {scene_id}")

    config = SCENES[scene_id]
    images = [load_rgb_image(DATASET_DIR / image_file) for image_file in config.image_files]
    mask = load_mask_raw(DATASET_DIR / config.mask_file)

    target_shape = images[0].shape[:2]
    aligned_images = [images[0]]
    images_were_resized = [False]
    for image in images[1:]:
        image_was_resized = image.shape[:2] != target_shape
        images_were_resized.append(image_was_resized)
        aligned_images.append(resize_image_bilinear(image, target_shape) if image_was_resized else image)

    mask_was_resized = mask.shape[:2] != target_shape
    if align_mask and mask_was_resized:
        mask = resize_mask_nearest(mask, target_shape)

    return {
        "config": config,
        "images": aligned_images,
        "mask": mask,
        "images_were_resized": images_were_resized,
        "mask_was_resized": mask_was_resized,
    }


def validate_dataset_files(dataset_dir: str | Path = DATASET_DIR) -> list[Path]:
    """Return missing dataset files, if any."""

    dataset_dir = Path(dataset_dir)
    expected_files: list[Path] = []
    for scene in SCENES.values():
        expected_files.extend(dataset_dir / image_file for image_file in scene.image_files)
        expected_files.append(dataset_dir / scene.mask_file)
    return [path for path in expected_files if not path.exists()]
