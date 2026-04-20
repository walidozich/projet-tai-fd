"""Segmentation evaluation metrics against Ground Truth masks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


def normalize_mask_by_color_clustering(
    mask: np.ndarray,
    n_classes: int,
    random_state: int = 42,
) -> np.ndarray:
    """Convert a multi-channel colored mask into `n_classes` labels.

    Some provided Ground Truth masks are RGBA images with many unique colors
    caused by anti-aliasing/resizing. For evaluation, this compresses those
    colors into a small label image. This must be used only on Ground Truth,
    never as an input to segmentation.
    """

    if mask.ndim == 2:
        values = mask.reshape(-1, 1).astype(np.float32)
    else:
        values = mask[..., :3].reshape(-1, 3).astype(np.float32)

    model = KMeans(n_clusters=n_classes, random_state=random_state, n_init=10)
    labels = model.fit_predict(values)
    return labels.reshape(mask.shape[:2]).astype(np.int32)


def multiclass_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Return an `n_classes x n_classes` confusion matrix."""

    return confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=list(range(n_classes)))


def match_predicted_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Match predicted labels to true labels by maximizing overlap."""

    matrix = multiclass_confusion_matrix(y_true, y_pred, n_classes)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    pred_to_true = {int(pred): int(true) for true, pred in zip(row_ind, col_ind)}
    matched = np.zeros_like(y_pred, dtype=np.int32)
    for pred_label in range(n_classes):
        matched[y_pred == pred_label] = pred_to_true.get(pred_label, pred_label)
    return matched, pred_to_true


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[str, object]:
    """Compute accuracy, per-class precision/recall/F1/IoU, and mean IoU."""

    matrix = multiclass_confusion_matrix(y_true, y_pred, n_classes)
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    per_class = []
    ious = []
    f1_scores = []
    for label in range(n_classes):
        tp = float(matrix[label, label])
        fp = float(matrix[:, label].sum() - tp)
        fn = float(matrix[label, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
        per_class.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            }
        )
        ious.append(iou)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
