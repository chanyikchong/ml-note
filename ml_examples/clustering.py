"""
K-Means Clustering Demo
=======================

Demonstrates:
1. K-Means algorithm from scratch
2. Initialization methods (random, k-means++)
3. Elbow method for choosing k
4. Clustering evaluation metrics
"""

import numpy as np
from typing import Tuple, List


class KMeansFromScratch:
    """
    K-Means Clustering implementation from scratch.

    Algorithm:
    1. Initialize k centroids
    2. Assign each point to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = "kmeans++",
        random_state: int = None,
    ):
        """
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            tol: Convergence tolerance
            init: Initialization method ('random' or 'kmeans++')
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None  # Within-cluster sum of squares
        self.n_iter_ = 0

    def _init_centroids_random(self, X: np.ndarray) -> np.ndarray:
        """Random initialization: pick k random points as centroids."""
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices].copy()

    def _init_centroids_kmeans_pp(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means++ initialization.

        1. Choose first centroid uniformly at random
        2. For each subsequent centroid:
           - Compute D(x) = distance to nearest existing centroid
           - Choose new centroid with probability proportional to D(x)^2
        """
        centroids = []

        # First centroid: random
        idx = np.random.randint(len(X))
        centroids.append(X[idx])

        for _ in range(1, self.n_clusters):
            # Compute squared distances to nearest centroid
            distances = np.array(
                [min(np.sum((x - c) ** 2) for c in centroids) for x in X]
            )

            # Choose new centroid with probability proportional to D^2
            probs = distances / distances.sum()
            idx = np.random.choice(len(X), p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def fit(self, X: np.ndarray) -> "KMeansFromScratch":
        """
        Fit K-Means to data.

        Args:
            X: Data matrix (n_samples, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize centroids
        if self.init == "kmeans++":
            self.centroids_ = self._init_centroids_kmeans_pp(X)
        else:
            self.centroids_ = self._init_centroids_random(X)

        for i in range(self.max_iter):
            # Step 1: Assign points to nearest centroid
            distances = self._compute_distances(X)
            self.labels_ = np.argmin(distances, axis=1)

            # Step 2: Update centroids
            new_centroids = np.array(
                [X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)]
            )

            # Check convergence
            if np.allclose(self.centroids_, new_centroids, atol=self.tol):
                break

            self.centroids_ = new_centroids
            self.n_iter_ = i + 1

        # Compute final inertia
        self.inertia_ = self._compute_inertia(X)

        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute distances from each point to each centroid."""
        # Shape: (n_samples, n_clusters)
        distances = np.zeros((len(X), self.n_clusters))
        for k, centroid in enumerate(self.centroids_):
            distances[:, k] = np.sum((X - centroid) ** 2, axis=1)
        return distances

    def _compute_inertia(self, X: np.ndarray) -> float:
        """Compute within-cluster sum of squares."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            inertia += np.sum((cluster_points - self.centroids_[k]) ** 2)
        return inertia

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score (simplified version).

    For each point:
    - a = mean distance to points in same cluster
    - b = mean distance to points in nearest other cluster
    - silhouette = (b - a) / max(a, b)

    Returns mean silhouette over all points.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    silhouettes = []

    for i in range(n_samples):
        # a: mean distance to same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a = np.mean([np.linalg.norm(X[i] - x) for x in same_cluster if not np.array_equal(x, X[i])])
        else:
            a = 0

        # b: mean distance to nearest other cluster
        b = float("inf")
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = X[labels == label]
                mean_dist = np.mean([np.linalg.norm(X[i] - x) for x in other_cluster])
                b = min(b, mean_dist)

        # Silhouette coefficient
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouettes.append(s)

    return np.mean(silhouettes)


def run_demo():
    """Run the K-Means clustering demonstration."""
    np.random.seed(42)

    print("1. Generating synthetic clustered data...")
    # Generate 3 clusters
    centers = np.array([[0, 0], [5, 5], [10, 0]])
    n_per_cluster = 100

    X = np.vstack([
        np.random.randn(n_per_cluster, 2) + centers[0],
        np.random.randn(n_per_cluster, 2) + centers[1],
        np.random.randn(n_per_cluster, 2) + centers[2],
    ])
    true_labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)

    print(f"   Data shape: {X.shape}")
    print(f"   True cluster centers:\n{centers}")

    print("\n2. K-Means with random initialization...")
    kmeans_random = KMeansFromScratch(n_clusters=3, init="random", random_state=42)
    labels_random = kmeans_random.fit_predict(X)
    print(f"   Iterations: {kmeans_random.n_iter_}")
    print(f"   Inertia: {kmeans_random.inertia_:.2f}")
    print(f"   Learned centers:\n{kmeans_random.centroids_}")

    print("\n3. K-Means with K-Means++ initialization...")
    kmeans_pp = KMeansFromScratch(n_clusters=3, init="kmeans++", random_state=42)
    labels_pp = kmeans_pp.fit_predict(X)
    print(f"   Iterations: {kmeans_pp.n_iter_}")
    print(f"   Inertia: {kmeans_pp.inertia_:.2f}")
    print(f"   Learned centers:\n{kmeans_pp.centroids_}")

    print("\n4. Elbow method (finding optimal k)...")
    inertias = []
    k_values = range(1, 8)

    for k in k_values:
        km = KMeansFromScratch(n_clusters=k, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)

    print("   k | Inertia")
    print("   --|--------")
    for k, inertia in zip(k_values, inertias):
        marker = " <-- elbow" if k == 3 else ""
        print(f"   {k} | {inertia:.2f}{marker}")

    print("\n5. Silhouette score evaluation...")
    for k in [2, 3, 4, 5]:
        km = KMeansFromScratch(n_clusters=k, random_state=42)
        km.fit(X)
        score = silhouette_score(X, km.labels_)
        marker = " <-- best" if k == 3 else ""
        print(f"   k={k}: silhouette = {score:.4f}{marker}")

    print("\n6. Cluster sizes...")
    unique, counts = np.unique(labels_pp, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   Cluster {label}: {count} points")


if __name__ == "__main__":
    run_demo()
