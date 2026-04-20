"""Visualization helpers for images, masks, overlays, plots, and confusion matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary or label mask on an RGB image."""

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3)")

    image_rgb = image[..., :3].astype(np.float32)
    if mask.ndim == 3:
        foreground = np.any(mask[..., :3] > 0, axis=2)
    else:
        foreground = mask > 0

    result = image_rgb.copy()
    overlay_color = np.array(color, dtype=np.float32)
    result[foreground] = (1 - alpha) * result[foreground] + alpha * overlay_color
    return np.clip(result, 0, 255).astype(np.uint8)


def show_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Show an image next to a mask and optionally save the figure."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray" if mask.ndim == 2 else None)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig, axes
