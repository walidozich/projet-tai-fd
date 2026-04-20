"""FD segmentation methods: clustering, elbow selection, PCA, and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn_extra.cluster import KMedoids


@dataclass(frozen=True)
class ClusteringResult:
    """Common clustering result container."""

    labels: np.ndarray
    method: str
    inertia: float | None = None
    extra: dict[str, object] | None = None


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
) -> ClusteringResult:
    """Cluster features with K-Means."""

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(features)
    return ClusteringResult(
        labels=labels,
        method="kmeans",
        inertia=float(model.inertia_),
        extra={"cluster_centers": model.cluster_centers_},
    )


def run_kmedoids(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    method: str = "pam",
) -> ClusteringResult:
    """Cluster features with K-Medoids/PAM."""

    model = KMedoids(n_clusters=n_clusters, random_state=random_state, method=method)
    labels = model.fit_predict(features)
    return ClusteringResult(
        labels=labels,
        method="kmedoids",
        inertia=float(model.inertia_),
        extra={"medoid_indices": model.medoid_indices_, "cluster_centers": model.cluster_centers_},
    )


def run_agnes(
    features: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
) -> ClusteringResult:
    """Cluster features with AGNES/agglomerative hierarchical clustering."""

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(features)
    return ClusteringResult(labels=labels, method="agnes", inertia=compute_inertia(features, labels))


def run_diana(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> ClusteringResult:
    """Approximate DIANA divisive clustering by recursively splitting high-SSE clusters.

    DIANA was studied conceptually in TP. Scikit-learn does not provide a direct
    DIANA implementation, so this function uses a transparent divisive strategy:
    repeatedly split the current cluster with highest SSE using K-Means with
    `k=2` until the requested number of clusters is reached.
    """

    n_samples = features.shape[0]
    if not 1 <= n_clusters <= n_samples:
        raise ValueError("n_clusters must be between 1 and number of samples")
    if n_clusters == 1:
        return ClusteringResult(
            labels=np.zeros(n_samples, dtype=np.int32),
            method="diana_approx",
            inertia=compute_inertia(features, np.zeros(n_samples, dtype=np.int32)),
        )

    labels = np.zeros(n_samples, dtype=np.int32)
    next_label = 1
    rng_state = random_state

    while len(np.unique(labels)) < n_clusters:
        unique_labels = np.unique(labels)
        splittable = [label for label in unique_labels if np.sum(labels == label) >= 2]
        if not splittable:
            break

        label_to_split = max(
            splittable,
            key=lambda label: compute_inertia(features[labels == label], np.zeros(np.sum(labels == label))),
        )
        indices = np.flatnonzero(labels == label_to_split)
        split_model = KMeans(n_clusters=2, random_state=rng_state, n_init=10)
        split_labels = split_model.fit_predict(features[indices])
        labels[indices[split_labels == 1]] = next_label
        next_label += 1
        rng_state += 1

    labels = relabel_consecutive(labels)
    return ClusteringResult(labels=labels, method="diana_approx", inertia=compute_inertia(features, labels))


def run_dbscan(
    features: np.ndarray,
    eps: float,
    min_samples: int = 5,
) -> ClusteringResult:
    """Cluster features with DBSCAN. Noise points are labeled `-1`."""

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)
    return ClusteringResult(
        labels=labels,
        method="dbscan",
        inertia=None,
        extra={"eps": eps, "min_samples": min_samples, "noise_count": int(np.sum(labels == -1))},
    )


def compute_pairwise_distances(
    features: np.ndarray,
    metric: str = "euclidean",
    max_samples: int | None = None,
) -> np.ndarray:
    """Compute pairwise distances, optionally on the first `max_samples` rows."""

    sample = features[:max_samples] if max_samples is not None else features
    return pairwise_distances(sample, metric=metric)


def compute_inertia(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute intra-cluster SSE/inertia for arbitrary labels.

    Noise label `-1` is ignored because DBSCAN does not assign noise to a
    regular cluster.
    """

    total = 0.0
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster = features[labels == label]
        if cluster.size == 0:
            continue
        center = cluster.mean(axis=0)
        total += float(np.sum((cluster - center) ** 2))
    return total


def elbow_curve(
    features: np.ndarray,
    k_values: list[int] | range,
    random_state: int = 42,
) -> list[dict[str, float | int]]:
    """Compute K-Means inertia values for candidate cluster counts."""

    results = []
    for k in k_values:
        result = run_kmeans(features, n_clusters=int(k), random_state=random_state)
        results.append({"k": int(k), "inertia": float(result.inertia)})
    return results


def detect_elbow_k(curve: list[dict[str, float | int]]) -> int:
    """Detect elbow by maximum distance from the line joining first/last points."""

    if len(curve) < 3:
        return int(curve[0]["k"])

    points = np.array([[item["k"], item["inertia"]] for item in curve], dtype=np.float64)
    first = points[0]
    last = points[-1]
    line = last - first
    norm = np.linalg.norm(line)
    if norm == 0:
        return int(points[0, 0])

    distances = np.abs(np.cross(line, first - points) / norm)
    return int(points[int(np.argmax(distances)), 0])


def pca_2d(features: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, PCA]:
    """Project features to 2D with PCA for visualization."""

    model = PCA(n_components=2, random_state=random_state)
    projection = model.fit_transform(features)
    return projection, model


def clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 5000,
    random_state: int = 42,
) -> dict[str, float | None]:
    """Compute internal clustering metrics when label structure allows it."""

    valid = labels != -1
    valid_features = features[valid]
    valid_labels = labels[valid]
    unique = np.unique(valid_labels)
    if valid_features.shape[0] < 2 or unique.size < 2 or unique.size >= valid_features.shape[0]:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "inertia": compute_inertia(features, labels),
        }

    metric_sample_size = None
    if sample_size is not None and valid_features.shape[0] > sample_size:
        metric_sample_size = sample_size

    return {
        "silhouette": float(
            silhouette_score(
                valid_features,
                valid_labels,
                sample_size=metric_sample_size,
                random_state=random_state,
            )
        ),
        "davies_bouldin": float(davies_bouldin_score(valid_features, valid_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(valid_features, valid_labels)),
        "inertia": compute_inertia(features, labels),
    }


def labels_to_image(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reshape flat pixel labels back to an image mask."""

    return labels.reshape(shape).astype(np.int32)


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """Relabel non-noise labels to consecutive integers while preserving `-1`."""

    output = np.full_like(labels, fill_value=-1)
    next_label = 0
    for label in np.unique(labels):
        if label == -1:
            continue
        output[labels == label] = next_label
        next_label += 1
    return output
