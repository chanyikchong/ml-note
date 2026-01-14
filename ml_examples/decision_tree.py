"""
Decision Tree Demo
==================

Demonstrates:
1. Simple decision tree from scratch
2. Splitting criteria (Gini, Entropy)
3. Tree structure and predictions
4. Using sklearn for practical trees
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import Counter


class DecisionTreeNode:
    """Node in a decision tree."""

    def __init__(
        self,
        feature_idx: int = None,
        threshold: float = None,
        left: "DecisionTreeNode" = None,
        right: "DecisionTreeNode" = None,
        value: int = None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf prediction

    def is_leaf(self) -> bool:
        return self.value is not None


class SimpleDecisionTree:
    """
    Simple Decision Tree Classifier.

    Uses recursive binary splitting with Gini impurity or entropy.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
    ):
        """
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            criterion: 'gini' or 'entropy'
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes = None

    def _gini(self, y: np.ndarray) -> float:
        """
        Gini impurity.

        Gini = 1 - Σ(p_k)²
        """
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def _entropy(self, y: np.ndarray) -> float:
        """
        Entropy.

        H = -Σ p_k * log(p_k)
        """
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))

    def _impurity(self, y: np.ndarray) -> float:
        """Compute impurity based on criterion."""
        if self.criterion == "gini":
            return self._gini(y)
        else:
            return self._entropy(y)

    def _information_gain(
        self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """
        Information gain from a split.

        IG = H(parent) - weighted_avg(H(children))
        """
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0

        parent_impurity = self._impurity(y)
        child_impurity = (n_left / n) * self._impurity(y_left) + (
            n_right / n
        ) * self._impurity(y_right)

        return parent_impurity - child_impurity

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split for a node.

        Returns:
            (best_feature, best_threshold, best_gain)
        """
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # Get unique thresholds (midpoints between sorted values)
            values = np.unique(X[:, feature_idx])
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Compute information gain
                gain = self._information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        """Recursively build the tree."""
        n_samples = len(y)

        # Stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            # Create leaf node with majority class
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)

        # Find best split
        feature_idx, threshold, gain = self._best_split(X, y)

        if gain == 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDecisionTree":
        """Fit the decision tree."""
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_sample(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Predict class for a single sample."""
        if node.is_leaf():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for all samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)

    def print_tree(self, node: DecisionTreeNode = None, indent: str = ""):
        """Print tree structure."""
        if node is None:
            node = self.root

        if node.is_leaf():
            print(f"{indent}Leaf: class {node.value}")
        else:
            print(f"{indent}Feature {node.feature_idx} <= {node.threshold:.4f}")
            print(f"{indent}  Left:")
            self.print_tree(node.left, indent + "    ")
            print(f"{indent}  Right:")
            self.print_tree(node.right, indent + "    ")


def run_demo():
    """Run the decision tree demonstration."""
    np.random.seed(42)

    print("1. Generating classification data...")
    n_samples = 200
    n_features = 4

    # Generate data with clear decision boundaries
    X = np.random.randn(n_samples, n_features)
    # Class 1 if (x0 > 0 and x1 > 0) or (x0 < 0 and x1 < 0)
    y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(int)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Features: {n_features}")

    print("\n2. Impurity measures comparison...")

    def show_impurity(probs, name):
        y_sample = np.repeat(np.arange(len(probs)), (np.array(probs) * 100).astype(int))
        tree = SimpleDecisionTree()
        tree.n_classes = len(probs)
        gini = tree._gini(y_sample)
        tree.criterion = "entropy"
        entropy = tree._entropy(y_sample)
        print(f"   {name}: Gini={gini:.4f}, Entropy={entropy:.4f}")

    show_impurity([0.5, 0.5], "p=[0.5, 0.5]")
    show_impurity([0.9, 0.1], "p=[0.9, 0.1]")
    show_impurity([1.0, 0.0], "p=[1.0, 0.0]")

    print("\n3. Training decision tree with Gini...")
    tree_gini = SimpleDecisionTree(max_depth=4, criterion="gini")
    tree_gini.fit(X_train, y_train)

    train_acc = tree_gini.score(X_train, y_train)
    test_acc = tree_gini.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")

    print("\n4. Training decision tree with Entropy...")
    tree_entropy = SimpleDecisionTree(max_depth=4, criterion="entropy")
    tree_entropy.fit(X_train, y_train)

    train_acc_e = tree_entropy.score(X_train, y_train)
    test_acc_e = tree_entropy.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_e:.4f}")
    print(f"   Test accuracy: {test_acc_e:.4f}")

    print("\n5. Effect of max_depth...")
    for depth in [1, 2, 3, 4, 5, 10]:
        tree = SimpleDecisionTree(max_depth=depth)
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train, y_train)
        test_acc = tree.score(X_test, y_test)
        print(f"   Depth {depth:2d}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    print("\n6. Tree structure (depth=2)...")
    tree_small = SimpleDecisionTree(max_depth=2)
    tree_small.fit(X_train, y_train)
    tree_small.print_tree()

    print("\n7. Using sklearn DecisionTreeClassifier...")
    try:
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(X_train, y_train)

        print(f"   sklearn Train accuracy: {clf.score(X_train, y_train):.4f}")
        print(f"   sklearn Test accuracy: {clf.score(X_test, y_test):.4f}")
        print(f"   Feature importances: {clf.feature_importances_}")

    except ImportError:
        print("   sklearn not available")


if __name__ == "__main__":
    run_demo()
