"""
Logistic Regression Demo
========================

Demonstrates:
1. Sigmoid function and its properties
2. Gradient descent for logistic regression
3. Binary cross-entropy loss
4. Evaluation metrics
"""

import numpy as np
from typing import Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    σ(z) = 1 / (1 + exp(-z))
    """
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy loss.

    L = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def logistic_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 1000,
    reg: float = 0.0,
) -> Tuple[np.ndarray, float, list]:
    """
    Logistic Regression via Gradient Descent.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (n_samples,)
        lr: Learning rate
        epochs: Number of iterations
        reg: L2 regularization strength

    Returns:
        Tuple of (weights, bias, loss_history)
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []

    for _ in range(epochs):
        # Forward pass
        z = X @ w + b
        p_hat = sigmoid(z)

        # Compute loss
        loss = binary_cross_entropy(y, p_hat)
        if reg > 0:
            loss += (reg / 2) * np.sum(w**2)
        losses.append(loss)

        # Compute gradients
        error = p_hat - y
        grad_w = (1 / n_samples) * X.T @ error + reg * w
        grad_b = (1 / n_samples) * np.sum(error)

        # Update parameters
        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b, losses


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict probabilities."""
    return sigmoid(X @ w + b)


def predict(X: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> np.ndarray:
    """Predict binary class labels."""
    return (predict_proba(X, w, b) >= threshold).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def run_demo():
    """Run the logistic regression demonstration."""
    np.random.seed(42)

    print("1. Generating synthetic classification data...")
    n_samples = 500
    n_features = 5

    # Generate two clusters
    X_class0 = np.random.randn(n_samples // 2, n_features) - 1
    X_class1 = np.random.randn(n_samples // 2, n_features) + 1
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    print("\n2. Sigmoid function properties...")
    z_test = np.array([-5, -2, 0, 2, 5])
    print(f"   z values:       {z_test}")
    print(f"   sigmoid(z):     {sigmoid(z_test)}")
    print(f"   sigmoid(0) = 0.5: {sigmoid(0):.4f}")
    print(f"   sigmoid(-z) = 1 - sigmoid(z): {sigmoid(-2):.4f} = {1 - sigmoid(2):.4f}")

    print("\n3. Training logistic regression with GD...")
    w, b, losses = logistic_regression_gd(X_train, y_train, lr=0.1, epochs=500, reg=0.01)
    print(f"   Learned weights: {w}")
    print(f"   Learned bias: {b:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")

    print("\n4. Evaluating on training set...")
    y_train_pred = predict(X_train, w, b)
    train_metrics = compute_metrics(y_train, y_train_pred)
    print(f"   Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"   Precision: {train_metrics['precision']:.4f}")
    print(f"   Recall:    {train_metrics['recall']:.4f}")
    print(f"   F1 Score:  {train_metrics['f1']:.4f}")

    print("\n5. Evaluating on test set...")
    y_test_pred = predict(X_test, w, b)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")

    cm = test_metrics["confusion_matrix"]
    print("\n   Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"   Actual 0    {cm['tn']:4d}   {cm['fp']:4d}")
    print(f"   Actual 1    {cm['fn']:4d}   {cm['tp']:4d}")

    print("\n6. Probability calibration check...")
    y_test_proba = predict_proba(X_test, w, b)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print("   Bin       | Count | Actual Rate")
    print("   ----------|-------|------------")
    for i in range(len(bins) - 1):
        mask = (y_test_proba >= bins[i]) & (y_test_proba < bins[i + 1])
        if np.sum(mask) > 0:
            actual_rate = np.mean(y_test[mask])
            print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}    | {np.sum(mask):5d} | {actual_rate:.4f}")


if __name__ == "__main__":
    run_demo()
