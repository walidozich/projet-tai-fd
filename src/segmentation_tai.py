"""TAI image processing methods used before and during segmentation."""

from collections import deque

import cv2
import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image loaded with OpenCV to RGB."""

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected a BGR image with shape (height, width, 3)")
    return image[..., :3][..., ::-1].copy()


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to uint8 grayscale."""

    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return gray.astype(np.uint8)


def histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute a grayscale histogram."""

    gray = rgb_to_gray(image)
    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    return hist.astype(np.int64)


def cumulative_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute the cumulative grayscale histogram."""

    return np.cumsum(histogram(image, bins=bins))


def normalize_histogram(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to improve grayscale contrast."""

    gray = rgb_to_gray(image)
    return cv2.equalizeHist(gray)


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a mean/box filter."""

    return cv2.blur(image, (kernel_size, kernel_size))


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing."""

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filtering, useful for impulse noise."""

    return cv2.medianBlur(image, kernel_size)


def sobel_gradients(image: np.ndarray, kernel_size: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sobel horizontal and vertical gradients."""

    gray = rgb_to_gray(image).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel_size)
    return grad_x, grad_y


def gradient_magnitude_direction(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient magnitude and direction from Sobel gradients."""

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    return magnitude, direction


def simple_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Segment foreground with a fixed grayscale threshold."""

    gray = rgb_to_gray(image)
    return ((gray >= threshold) * 255).astype(np.uint8)


def otsu_threshold(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Segment foreground using Otsu's automatic threshold."""

    gray = rgb_to_gray(image)
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8), float(threshold)


def connected_components_8(mask: np.ndarray) -> dict[str, np.ndarray | int]:
    """Compute 8-connected components for a binary mask."""

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return {
        "num_labels": num_labels,
        "labels": labels,
        "stats": stats,
        "centroids": centroids,
    }


def flood_fill_region(
    image: np.ndarray,
    seed: tuple[int, int],
    tolerance: int = 10,
) -> np.ndarray:
    """Grow a region from `seed=(row, col)` using intensity tolerance."""

    gray = rgb_to_gray(image)
    height, width = gray.shape
    seed_row, seed_col = seed
    if not (0 <= seed_row < height and 0 <= seed_col < width):
        raise ValueError("Seed is outside the image bounds")

    seed_value = int(gray[seed_row, seed_col])
    visited = np.zeros((height, width), dtype=bool)
    region = np.zeros((height, width), dtype=np.uint8)
    queue: deque[tuple[int, int]] = deque([(seed_row, seed_col)])
    visited[seed_row, seed_col] = True

    while queue:
        row, col = queue.popleft()
        if abs(int(gray[row, col]) - seed_value) > tolerance:
            continue

        region[row, col] = 255
        for next_row in (row - 1, row, row + 1):
            for next_col in (col - 1, col, col + 1):
                if next_row == row and next_col == col:
                    continue
                if 0 <= next_row < height and 0 <= next_col < width and not visited[next_row, next_col]:
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

    return region
