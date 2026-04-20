# FD Clustering Summary

Implemented and smoke-tested FD clustering utilities on a resized `scene1` input.

## Functions verified

- `run_kmeans(features, n_clusters)`
- `run_kmedoids(features, n_clusters)`
- `run_agnes(features, n_clusters)`
- `run_diana(features, n_clusters)` as a documented DIANA approximation
- `run_dbscan(features, eps, min_samples)`
- `elbow_curve(features, k_values)`
- `detect_elbow_k(curve)`
- `compute_pairwise_distances(features)`
- `compute_inertia(features, labels)`
- `pca_2d(features)`
- `clustering_metrics(features, labels)`
- `labels_to_image(labels, shape)`

## Smoke-test details

- Test scene: `scene1`
- Resized image shape: `(56, 64, 3)`
- Scale factor: `0.11469534050179211`
- Full feature matrix shape: `(3584, 5)`
- Sample feature matrix shape: `(896, 5)`
- K-Means inertia: `353.5774`
- K-Medoids inertia: `258.1982`
- AGNES inertia: `96.8982`
- DIANA approximation inertia: `88.9006`
- DBSCAN noise points: `177`
- Elbow candidates: `[{'k': 2, 'inertia': 157.85580444335938}, {'k': 3, 'inertia': 109.9737548828125}, {'k': 4, 'inertia': 86.32343292236328}, {'k': 5, 'inertia': 69.28736114501953}]`
- Detected elbow k: `3`
- K-Medoids metrics: `{'silhouette': 0.3909454941749573, 'davies_bouldin': 1.0255734871537499, 'calinski_harabasz': 821.8172728612657, 'inertia': 87.52821731567383}`
- PCA explained variance ratio: `[0.6311047673225403, 0.23324938118457794]`
