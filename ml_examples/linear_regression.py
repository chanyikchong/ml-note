"""
Linear Regression Demo
======================

Demonstrates:
1. Closed-form OLS solution
2. Gradient descent implementation
3. Ridge and Lasso regularization comparison
"""

import numpy as np
from typing import Tuple


def ols_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares closed-form solution.

    β = (X'X)^(-1) X'y

    Args:
        X: Design matrix (n_samples, n_features)
        y: Target vector (n_samples,)

    Returns:
        Coefficient vector (n_features,)
    """
    return np.linalg.solve(X.T @ X, X.T @ y)


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Ridge Regression closed-form solution.

    β = (X'X + λI)^(-1) X'y

    Args:
        X: Design matrix
        y: Target vector
        lambda_: Regularization strength

    Returns:
        Coefficient vector
    """
    n_features = X.shape[1]
    return np.linalg.solve(X.T @ X + lambda_ * np.eye(n_features), X.T @ y)


def linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1000,
    reg_type: str = "none",
    lambda_: float = 0.0,
) -> Tuple[np.ndarray, list]:
    """
    Linear Regression via Gradient Descent.

    Args:
        X: Design matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        lr: Learning rate
        epochs: Number of iterations
        reg_type: 'none', 'l2', or 'l1'
        lambda_: Regularization strength

    Returns:
        Tuple of (coefficients, loss_history)
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []

    for _ in range(epochs):
        # Predictions
        y_pred = X @ w

        # Loss
        mse = np.mean((y - y_pred) ** 2)
        if reg_type == "l2":
            loss = mse + lambda_ * np.sum(w**2)
        elif reg_type == "l1":
            loss = mse + lambda_ * np.sum(np.abs(w))
        else:
            loss = mse
        losses.append(loss)

        # Gradient
        grad = (-2 / n_samples) * X.T @ (y - y_pred)
        if reg_type == "l2":
            grad += 2 * lambda_ * w
        elif reg_type == "l1":
            grad += lambda_ * np.sign(w)

        # Update
        w = w - lr * grad

    return w, losses


def run_demo():
    """Run the linear regression demonstration."""
    np.random.seed(42)

    print("1. Generating synthetic data...")
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([3.0, -2.0, 1.5, 0.0, 0.0])  # Sparse coefficients
    y = X @ true_w + np.random.randn(n_samples) * 0.5

    # Add intercept column
    X_with_intercept = np.c_[np.ones(n_samples), X]
    true_w_with_intercept = np.r_[0, true_w]

    print(f"   True coefficients: {true_w}")
    print(f"   Data shape: X={X.shape}, y={y.shape}")

    print("\n2. OLS Closed-form Solution...")
    w_ols = ols_closed_form(X_with_intercept, y)
    y_pred_ols = X_with_intercept @ w_ols
    mse_ols = np.mean((y - y_pred_ols) ** 2)
    print(f"   Coefficients: {w_ols[1:]}")  # Exclude intercept
    print(f"   MSE: {mse_ols:.4f}")

    print("\n3. Gradient Descent Solution...")
    w_gd, losses = linear_regression_gd(X_with_intercept, y, lr=0.01, epochs=1000)
    y_pred_gd = X_with_intercept @ w_gd
    mse_gd = np.mean((y - y_pred_gd) ** 2)
    print(f"   Coefficients: {w_gd[1:]}")
    print(f"   MSE: {mse_gd:.4f}")
    print(f"   Final loss after {len(losses)} epochs: {losses[-1]:.4f}")

    print("\n4. Ridge Regression (L2)...")
    lambda_ridge = 1.0
    w_ridge = ridge_closed_form(X_with_intercept, y, lambda_ridge)
    y_pred_ridge = X_with_intercept @ w_ridge
    mse_ridge = np.mean((y - y_pred_ridge) ** 2)
    print(f"   Lambda: {lambda_ridge}")
    print(f"   Coefficients: {w_ridge[1:]}")
    print(f"   MSE: {mse_ridge:.4f}")
    print(f"   L2 norm of coefficients: {np.linalg.norm(w_ridge[1:]):.4f}")

    print("\n5. Lasso-like (L1) via GD...")
    lambda_l1 = 0.1
    w_l1, _ = linear_regression_gd(
        X_with_intercept, y, lr=0.01, epochs=2000, reg_type="l1", lambda_=lambda_l1
    )
    y_pred_l1 = X_with_intercept @ w_l1
    mse_l1 = np.mean((y - y_pred_l1) ** 2)
    sparsity = np.sum(np.abs(w_l1[1:]) < 0.1)
    print(f"   Lambda: {lambda_l1}")
    print(f"   Coefficients: {w_l1[1:]}")
    print(f"   MSE: {mse_l1:.4f}")
    print(f"   Near-zero coefficients: {sparsity}/{n_features}")

    print("\n6. Comparison Summary:")
    print("   Method          | MSE    | L2 Norm")
    print("   ----------------|--------|--------")
    print(f"   OLS             | {mse_ols:.4f} | {np.linalg.norm(w_ols[1:]):.4f}")
    print(f"   Gradient Desc   | {mse_gd:.4f} | {np.linalg.norm(w_gd[1:]):.4f}")
    print(f"   Ridge (λ=1.0)   | {mse_ridge:.4f} | {np.linalg.norm(w_ridge[1:]):.4f}")
    print(f"   L1 (λ=0.1)      | {mse_l1:.4f} | {np.linalg.norm(w_l1[1:]):.4f}")


if __name__ == "__main__":
    run_demo()
