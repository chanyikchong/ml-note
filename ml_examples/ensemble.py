"""
Ensemble Methods Demo
=====================

Demonstrates:
1. Random Forest concept (bagging + random features)
2. Gradient Boosting concept
3. Using sklearn for practical ensemble methods
"""

import numpy as np
from typing import List


def bootstrap_sample(X: np.ndarray, y: np.ndarray) -> tuple:
    """Create a bootstrap sample (sampling with replacement)."""
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X[indices], y[indices]


def run_random_forest_demo():
    """Demonstrate Random Forest concepts."""
    np.random.seed(42)

    print("=" * 60)
    print("RANDOM FOREST DEMO")
    print("=" * 60)

    print("\n1. Generating classification data...")
    n_samples = 500
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Non-linear decision boundary
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 2).astype(int)

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n2. Bootstrap sampling demonstration...")
    X_boot, y_boot = bootstrap_sample(X_train[:20], y_train[:20])
    unique_samples = len(np.unique(np.arange(20)))
    print(f"   Original 20 samples -> Bootstrap has ~{len(np.unique(X_boot, axis=0))} unique")
    print(f"   (Expected ~63.2% unique due to sampling with replacement)")

    print("\n3. Using sklearn RandomForestClassifier...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        # Single decision tree
        single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        single_tree.fit(X_train, y_train)
        single_acc = single_tree.score(X_test, y_test)

        # Random forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            max_features="sqrt",  # Random feature subspace
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_acc = rf.score(X_test, y_test)

        print(f"   Single Tree accuracy: {single_acc:.4f}")
        print(f"   Random Forest accuracy: {rf_acc:.4f}")
        print(f"   Improvement: {rf_acc - single_acc:.4f}")

        print("\n4. Effect of number of trees...")
        for n_trees in [1, 5, 10, 50, 100, 200]:
            rf_n = RandomForestClassifier(n_estimators=n_trees, max_depth=5, random_state=42)
            rf_n.fit(X_train, y_train)
            acc = rf_n.score(X_test, y_test)
            print(f"   {n_trees:3d} trees: {acc:.4f}")

        print("\n5. Feature importance...")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("   Feature ranking:")
        for i, idx in enumerate(indices[:5]):
            print(f"     {i+1}. Feature {idx}: {importances[idx]:.4f}")

        print("\n6. Out-of-Bag (OOB) score...")
        rf_oob = RandomForestClassifier(
            n_estimators=100, max_depth=5, oob_score=True, random_state=42
        )
        rf_oob.fit(X_train, y_train)
        print(f"   OOB Score: {rf_oob.oob_score_:.4f}")
        print(f"   Test Score: {rf_oob.score(X_test, y_test):.4f}")
        print("   (OOB score approximates test performance without holdout)")

    except ImportError:
        print("   sklearn not available")


def run_gradient_boosting_demo():
    """Demonstrate Gradient Boosting concepts."""
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING DEMO")
    print("=" * 60)

    print("\n1. Gradient Boosting concept...")
    print("""
   Key idea: Sequentially fit weak learners to residuals

   F_0(x) = initial prediction (e.g., mean for regression)
   For m = 1 to M:
       r_m = y - F_{m-1}(x)           # Compute residuals
       h_m = fit weak learner to r_m   # Fit to residuals
       F_m(x) = F_{m-1}(x) + η * h_m(x)  # Update prediction

   Final: F(x) = Σ η * h_m(x)
    """)

    print("2. Simple gradient boosting regression example...")
    # Generate regression data
    n = 100
    X = np.random.uniform(0, 10, (n, 1))
    y = np.sin(X.squeeze()) + np.random.normal(0, 0.2, n)

    # Manual gradient boosting with stumps
    n_estimators = 5
    learning_rate = 0.5

    predictions = np.zeros(n)
    print(f"   Initial MSE: {np.mean((y - predictions)**2):.4f}")

    for i in range(n_estimators):
        residuals = y - predictions

        # Fit a simple stump (split at median)
        median_x = np.median(X)
        left_mask = X.squeeze() <= median_x
        left_pred = np.mean(residuals[left_mask])
        right_pred = np.mean(residuals[~left_mask])

        # Update predictions
        stump_pred = np.where(X.squeeze() <= median_x, left_pred, right_pred)
        predictions += learning_rate * stump_pred

        mse = np.mean((y - predictions) ** 2)
        print(f"   After {i+1} stumps, MSE: {mse:.4f}")

    print("\n3. Using sklearn GradientBoostingClassifier...")
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.datasets import make_classification

        X_clf, y_clf = make_classification(
            n_samples=500, n_features=10, n_informative=5, random_state=42
        )
        split = int(0.8 * len(X_clf))
        X_train, X_test = X_clf[:split], X_clf[split:]
        y_train, y_test = y_clf[:split], y_clf[split:]

        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        gb.fit(X_train, y_train)

        print(f"   Train accuracy: {gb.score(X_train, y_train):.4f}")
        print(f"   Test accuracy: {gb.score(X_test, y_test):.4f}")

        print("\n4. Effect of learning rate...")
        for lr in [0.01, 0.05, 0.1, 0.5, 1.0]:
            gb_lr = GradientBoostingClassifier(
                n_estimators=100, learning_rate=lr, max_depth=3, random_state=42
            )
            gb_lr.fit(X_train, y_train)
            acc = gb_lr.score(X_test, y_test)
            print(f"   lr={lr:.2f}: {acc:.4f}")

        print("\n5. Staged prediction (training dynamics)...")
        train_scores = []
        test_scores = []
        for pred_train, pred_test in zip(
            gb.staged_predict(X_train), gb.staged_predict(X_test)
        ):
            train_scores.append(np.mean(pred_train == y_train))
            test_scores.append(np.mean(pred_test == y_test))

        for i in [0, 9, 49, 99]:
            print(f"   After {i+1:3d} trees: Train={train_scores[i]:.4f}, Test={test_scores[i]:.4f}")

        print("\n6. XGBoost-style concepts (if available)...")
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            hgb = HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.1, max_depth=3, random_state=42
            )
            hgb.fit(X_train, y_train)
            print(f"   HistGradientBoosting accuracy: {hgb.score(X_test, y_test):.4f}")
            print("   (Uses histogram-based splitting like XGBoost/LightGBM)")
        except ImportError:
            pass

    except ImportError:
        print("   sklearn not available")


def run_demo():
    """Run both ensemble demos."""
    run_random_forest_demo()
    run_gradient_boosting_demo()


if __name__ == "__main__":
    run_demo()
