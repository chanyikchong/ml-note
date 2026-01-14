"""
Support Vector Machine Demo
===========================

Demonstrates:
1. Linear SVM concept (maximum margin)
2. Soft margin intuition
3. Kernel trick intuition
4. Using sklearn for practical SVM
"""

import numpy as np
from typing import Tuple


class SimplifiedLinearSVM:
    """
    Simplified Linear SVM using gradient descent on hinge loss.

    Objective: min (1/2)||w||² + C * Σ max(0, 1 - y_i * (w·x_i + b))

    This is a simplified version for educational purposes.
    For production, use sklearn.svm.SVC or libsvm.
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        random_state: int = None,
    ):
        """
        Args:
            C: Regularization parameter (inverse of regularization strength)
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimplifiedLinearSVM":
        """
        Fit SVM using sub-gradient descent.

        Args:
            X: Features (n_samples, n_features)
            y: Labels in {-1, +1}
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Convert labels to {-1, +1}
        y = np.where(y == 0, -1, y)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            # Compute margins
            margins = y * (X @ self.w + self.b)

            # Sub-gradient of hinge loss
            grad_w = self.w.copy()  # From regularization term
            grad_b = 0

            for i in range(n_samples):
                if margins[i] < 1:  # Hinge loss is active
                    grad_w -= self.C * y[i] * X[i]
                    grad_b -= self.C * y[i]

            grad_w /= n_samples
            grad_b /= n_samples

            # Update
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function (signed distance to hyperplane)."""
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.sign(self.decision_function(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        y = np.where(y == 0, -1, y)
        return np.mean(self.predict(X) == y)


def compute_margin(w: np.ndarray, b: float) -> float:
    """
    Compute geometric margin of hyperplane.

    Margin = 2 / ||w||
    """
    return 2 / np.linalg.norm(w)


def demonstrate_kernel_idea():
    """
    Demonstrate the kernel trick concept.

    The kernel trick allows computing dot products in high-dimensional
    feature space without explicitly computing the transformation.
    """
    print("Kernel Trick Intuition:")
    print("-" * 40)

    # Example: Polynomial kernel
    def polynomial_features(x: np.ndarray, degree: int = 2) -> np.ndarray:
        """Explicit polynomial feature expansion for 2D."""
        # For degree=2: [1, x1, x2, x1², x1*x2, x2²]
        x1, x2 = x[0], x[1]
        return np.array([1, x1, x2, x1**2, x1 * x2, x2**2])

    def polynomial_kernel(x: np.ndarray, z: np.ndarray, degree: int = 2) -> float:
        """Polynomial kernel: (x·z + 1)^d"""
        return (np.dot(x, z) + 1) ** degree

    x = np.array([1.0, 2.0])
    z = np.array([3.0, 4.0])

    # Method 1: Explicit feature mapping + dot product
    phi_x = polynomial_features(x)
    phi_z = polynomial_features(z)
    explicit = np.dot(phi_x, phi_z)

    # Method 2: Kernel function (much cheaper)
    kernel = polynomial_kernel(x, z)

    print(f"   Point x: {x}")
    print(f"   Point z: {z}")
    print(f"\n   Explicit φ(x): {phi_x}")
    print(f"   Explicit φ(z): {phi_z}")
    print(f"\n   Explicit <φ(x), φ(z)>: {explicit}")
    print(f"   Kernel K(x, z) = (x·z + 1)²: {kernel}")
    print(f"\n   Match: {np.isclose(explicit, kernel)}")


def run_demo():
    """Run the SVM demonstration."""
    np.random.seed(42)

    print("1. Generating linearly separable data...")
    n_samples = 200

    # Two separated clusters
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X = np.vstack([X_class0, X_class1])
    y = np.array([-1] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    print(f"   Data shape: {X.shape}")

    print("\n2. Training simplified linear SVM...")
    svm = SimplifiedLinearSVM(C=1.0, learning_rate=0.001, max_iter=1000, random_state=42)
    svm.fit(X, y)

    print(f"   Learned weights: {svm.w}")
    print(f"   Learned bias: {svm.b:.4f}")
    print(f"   Margin: {compute_margin(svm.w, svm.b):.4f}")
    print(f"   Training accuracy: {svm.score(X, y):.4f}")

    print("\n3. Comparing different C values...")
    for C in [0.01, 0.1, 1.0, 10.0]:
        svm_c = SimplifiedLinearSVM(C=C, learning_rate=0.001, max_iter=1000, random_state=42)
        svm_c.fit(X, y)
        margin = compute_margin(svm_c.w, svm_c.b)
        acc = svm_c.score(X, y)
        print(f"   C={C:5.2f}: Margin={margin:.4f}, Accuracy={acc:.4f}")

    print("\n4. Support vectors intuition...")
    # Points close to decision boundary
    decision_values = np.abs(svm.decision_function(X))
    support_mask = decision_values < 1.5
    n_support = np.sum(support_mask)
    print(f"   Points near margin (|f(x)| < 1.5): {n_support}/{n_samples}")

    print("\n5. Kernel trick demonstration...")
    demonstrate_kernel_idea()

    print("\n6. Using sklearn SVM (recommended for production)...")
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        # Non-linear data (circles)
        from sklearn.datasets import make_circles

        X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

        # Linear SVM (will fail)
        svm_linear = SVC(kernel="linear")
        svm_linear.fit(X_circles, y_circles)
        acc_linear = svm_linear.score(X_circles, y_circles)

        # RBF kernel SVM (will work)
        svm_rbf = SVC(kernel="rbf", gamma="auto")
        svm_rbf.fit(X_circles, y_circles)
        acc_rbf = svm_rbf.score(X_circles, y_circles)

        print(f"   Circles dataset (non-linear):")
        print(f"   Linear SVM accuracy: {acc_linear:.4f}")
        print(f"   RBF SVM accuracy: {acc_rbf:.4f}")
        print(f"   RBF support vectors: {len(svm_rbf.support_)}/{len(X_circles)}")

    except ImportError:
        print("   sklearn not available, skipping this section")


if __name__ == "__main__":
    run_demo()
