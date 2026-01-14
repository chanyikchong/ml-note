"""
Principal Component Analysis (PCA) Demo
=======================================

Demonstrates:
1. PCA from scratch using eigendecomposition
2. PCA from scratch using SVD
3. Dimensionality reduction
4. Variance explained
"""

import numpy as np
from typing import Tuple


class PCAFromScratch:
    """
    Principal Component Analysis implementation from scratch.

    PCA finds orthogonal directions of maximum variance.

    Algorithm:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Compute eigendecomposition
    4. Sort by eigenvalue (variance explained)
    5. Project onto top k components
    """

    def __init__(self, n_components: int = None):
        """
        Args:
            n_components: Number of components to keep. None = keep all.
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # Principal components (eigenvectors)
        self.explained_variance_ = None  # Eigenvalues
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        """
        Fit PCA on data X.

        Args:
            X: Data matrix (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute covariance matrix
        # Cov = (1/(n-1)) * X.T @ X
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort by eigenvalue (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Store results
        n_components = self.n_components or n_features
        self.components_ = eigenvectors[:, :n_components].T  # Shape: (n_components, n_features)
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = eigenvalues[:n_components] / np.sum(eigenvalues)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components.

        Args:
            X: Data matrix (n_samples, n_features)

        Returns:
            Projected data (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal components.

        Args:
            X_transformed: Projected data (n_samples, n_components)

        Returns:
            Reconstructed data (n_samples, n_features)
        """
        return X_transformed @ self.components_ + self.mean_


def pca_svd(X: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA using Singular Value Decomposition.

    More numerically stable than eigendecomposition.

    X = U @ S @ V.T
    Principal components = V
    Scores = U @ S

    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of components

    Returns:
        Tuple of (transformed_data, components, explained_variance_ratio)
    """
    n_samples, n_features = X.shape
    n_components = n_components or min(n_samples, n_features)

    # Center
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Variance explained
    variance = (S**2) / (n_samples - 1)
    variance_ratio = variance / np.sum(variance)

    # Keep top components
    components = Vt[:n_components]
    transformed = X_centered @ components.T

    return transformed, components, variance_ratio[:n_components]


def run_demo():
    """Run the PCA demonstration."""
    np.random.seed(42)

    print("1. Generating synthetic data with correlated features...")
    n_samples = 200
    n_features = 5

    # Create data with specific structure
    # First 2 dimensions have most variance
    latent = np.random.randn(n_samples, 2)
    mixing = np.random.randn(2, n_features)
    noise = np.random.randn(n_samples, n_features) * 0.1
    X = latent @ mixing + noise

    print(f"   Data shape: {X.shape}")
    print(f"   Data variance per feature: {np.var(X, axis=0)}")

    print("\n2. PCA from scratch (eigendecomposition)...")
    pca = PCAFromScratch(n_components=3)
    X_transformed = pca.fit_transform(X)

    print(f"   Transformed shape: {X_transformed.shape}")
    print(f"   Explained variance: {pca.explained_variance_}")
    print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"   Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")

    print("\n3. PCA using SVD...")
    X_svd, components_svd, var_ratio_svd = pca_svd(X, n_components=3)
    print(f"   Explained variance ratio: {var_ratio_svd}")
    print(f"   SVD matches eigendecomposition: {np.allclose(np.abs(X_transformed), np.abs(X_svd))}")

    print("\n4. Reconstruction error...")
    # Full reconstruction (all components)
    pca_full = PCAFromScratch(n_components=5)
    X_full = pca_full.fit_transform(X)
    X_reconstructed_full = pca_full.inverse_transform(X_full)
    error_full = np.mean((X - X_reconstructed_full) ** 2)

    # Reduced reconstruction (3 components)
    X_reconstructed_3 = pca.inverse_transform(X_transformed)
    error_3 = np.mean((X - X_reconstructed_3) ** 2)

    # Reduced reconstruction (2 components)
    pca_2 = PCAFromScratch(n_components=2)
    X_2 = pca_2.fit_transform(X)
    X_reconstructed_2 = pca_2.inverse_transform(X_2)
    error_2 = np.mean((X - X_reconstructed_2) ** 2)

    print(f"   5 components MSE: {error_full:.6f}")
    print(f"   3 components MSE: {error_3:.6f}")
    print(f"   2 components MSE: {error_2:.6f}")

    print("\n5. Determining number of components...")
    pca_full = PCAFromScratch()
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    print("   Components | Cumulative Variance")
    print("   -----------|--------------------")
    for i, cv in enumerate(cumvar):
        marker = " <-- 95%" if cv >= 0.95 and (i == 0 or cumvar[i - 1] < 0.95) else ""
        print(f"   {i+1:10d} | {cv:.4f}{marker}")

    print("\n6. Principal components (loadings)...")
    print("   Top 2 principal components:")
    print(f"   PC1: {pca.components_[0]}")
    print(f"   PC2: {pca.components_[1]}")

    # Check orthogonality
    print(f"\n   Orthogonality check (PC1 · PC2): {np.dot(pca.components_[0], pca.components_[1]):.6f}")

    print("\n7. Whitening (standardizing PC scores)...")
    # Whitened data has unit variance in each PC direction
    X_whitened = X_transformed / np.sqrt(pca.explained_variance_)
    print(f"   Variance of whitened data: {np.var(X_whitened, axis=0)}")


if __name__ == "__main__":
    run_demo()
