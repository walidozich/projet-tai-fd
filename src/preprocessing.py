"""Common image preprocessing and feature extraction helpers."""

import cv2
import numpy as np
from PIL import Image


def normalize_minmax(features: np.ndarray) -> np.ndarray:
    """Scale each feature column to the [0, 1] range."""

    values = features.astype(np.float32, copy=False)
    min_values = values.min(axis=0)
    max_values = values.max(axis=0)
    ranges = max_values - min_values
    ranges[ranges == 0] = 1.0
    return (values - min_values) / ranges


def standardize_zscore(features: np.ndarray) -> np.ndarray:
    """Standardize each feature column with z-score normalization."""

    values = features.astype(np.float32, copy=False)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    return (values - means) / stds


def resize_for_clustering(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    """Resize large images while preserving aspect ratio.

    Returns the resized image and the scale factor. If no resize is needed, the
    original image and scale `1.0` are returned.
    """

    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_size:
        return image, 1.0

    scale = max_size / largest_side
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.asarray(resized), scale


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to uint8 grayscale."""

    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return gray.astype(np.uint8)


def extract_rgb_features(image: np.ndarray) -> np.ndarray:
    """Return one `[R, G, B]` feature row per pixel."""

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3)")
    return image[..., :3].reshape(-1, 3).astype(np.float32)


def extract_rgb_xy_features(image: np.ndarray) -> np.ndarray:
    """Return `[R, G, B, x, y]` features for each pixel.

    Coordinates are normalized to [0, 1] so they can be combined with scaled
    color features.
    """

    height, width = image.shape[:2]
    rgb = extract_rgb_features(image)
    yy, xx = np.indices((height, width))
    xy = np.column_stack((xx.ravel() / max(width - 1, 1), yy.ravel() / max(height - 1, 1)))
    return np.column_stack((rgb, xy.astype(np.float32)))


def extract_gray_gradient_xy_features(image: np.ndarray) -> np.ndarray:
    """Return `[gray, gradient, x, y]` features for each pixel."""

    gray = rgb_to_gray(image)
    gray_float = gray.astype(np.float32)
    sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    height, width = gray.shape
    yy, xx = np.indices((height, width))
    features = np.column_stack(
        (
            gray_float.ravel(),
            gradient.ravel(),
            xx.ravel() / max(width - 1, 1),
            yy.ravel() / max(height - 1, 1),
        )
    )
    return features.astype(np.float32)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a mask-like array to boolean foreground/background."""

    if mask.ndim == 3:
        return np.any(mask[..., :3] > 0, axis=2)
    return mask > 0


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected foreground components smaller than `min_area`."""

    binary = _binary_mask(mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return (cleaned * 255).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component."""

    binary = _binary_mask(mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ((labels == largest_label) * 255).astype(np.uint8)
