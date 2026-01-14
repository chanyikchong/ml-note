"""
Model Calibration and Thresholding Demo
=======================================

Demonstrates:
1. What calibration means
2. Reliability diagrams
3. Calibration methods (Platt scaling, isotonic)
4. Threshold optimization for different objectives
"""

import numpy as np
from typing import Tuple, List


def compute_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, List[dict]]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ (n_b / n) * |acc(b) - conf(b)|

    where:
    - acc(b) = actual accuracy in bin b
    - conf(b) = average predicted confidence in bin b
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include 1.0 in last bin
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        n_samples = np.sum(mask)
        if n_samples > 0:
            accuracy = np.mean(y_true[mask])
            confidence = np.mean(y_prob[mask])
            bin_ece = n_samples * np.abs(accuracy - confidence)
            ece += bin_ece

            bin_data.append({
                "bin": (bin_boundaries[i], bin_boundaries[i + 1]),
                "n_samples": n_samples,
                "accuracy": accuracy,
                "confidence": confidence,
            })

    ece /= len(y_true)
    return ece, bin_data


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier Score - measures calibration and refinement.

    BS = (1/n) * Σ (p_i - y_i)²

    Lower is better. Range [0, 1].
    """
    return np.mean((y_prob - y_true) ** 2)


class PlattScaling:
    """
    Platt Scaling for calibration.

    Fits a sigmoid: P(y=1|f) = 1 / (1 + exp(A*f + B))
    where f is the model's output (e.g., logits or raw probabilities).
    """

    def __init__(self):
        self.A = None
        self.B = None

    def fit(self, y_true: np.ndarray, scores: np.ndarray, max_iter: int = 1000, lr: float = 0.1):
        """Fit Platt scaling parameters."""
        # Convert to logits if probabilities
        if np.all((scores >= 0) & (scores <= 1)):
            scores = np.log(scores / (1 - scores + 1e-15) + 1e-15)

        # Initialize
        A, B = 0.0, 0.0

        for _ in range(max_iter):
            z = A * scores + B
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-15, 1 - 1e-15)

            # Gradient
            error = p - y_true
            grad_A = np.mean(error * scores)
            grad_B = np.mean(error)

            A -= lr * grad_A
            B -= lr * grad_B

        self.A = A
        self.B = B
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if np.all((scores >= 0) & (scores <= 1)):
            scores = np.log(scores / (1 - scores + 1e-15) + 1e-15)
        z = self.A * scores + self.B
        return 1 / (1 + np.exp(-z))


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """
    Find optimal classification threshold for a given metric.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: 'f1', 'accuracy', or 'balanced_accuracy'

    Returns:
        (optimal_threshold, optimal_metric_value)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        if metric == "accuracy":
            score = np.mean(y_pred == y_true)
        elif metric == "f1":
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif metric == "balanced_accuracy":
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (tpr + tnr) / 2

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def run_demo():
    """Run the calibration demonstration."""
    np.random.seed(42)

    print("1. Generating predictions from an overconfident model...")
    n_samples = 1000

    # True labels
    y_true = np.random.binomial(1, 0.5, n_samples)

    # Simulated overconfident model (probabilities pushed toward 0 or 1)
    base_probs = np.random.beta(2, 2, n_samples)
    y_prob_overconfident = np.where(
        y_true == 1,
        0.5 + 0.4 * base_probs,  # Push toward 1
        0.1 + 0.4 * (1 - base_probs),  # Push toward 0
    )

    # Add some noise and clip
    y_prob_overconfident += np.random.normal(0, 0.1, n_samples)
    y_prob_overconfident = np.clip(y_prob_overconfident, 0.01, 0.99)

    print(f"   Samples: {n_samples}")
    print(f"   Class balance: {np.mean(y_true):.2f}")

    print("\n2. Evaluating calibration metrics...")
    ece, bin_data = compute_calibration_error(y_true, y_prob_overconfident)
    bs = brier_score(y_true, y_prob_overconfident)
    print(f"   Expected Calibration Error (ECE): {ece:.4f}")
    print(f"   Brier Score: {bs:.4f}")

    print("\n3. Reliability diagram (before calibration)...")
    print("   Bin          | Samples | Accuracy | Confidence | Gap")
    print("   -------------|---------|----------|------------|-----")
    for bin_info in bin_data:
        gap = bin_info["accuracy"] - bin_info["confidence"]
        marker = "↑" if gap > 0.05 else ("↓" if gap < -0.05 else "")
        print(
            f"   {bin_info['bin'][0]:.1f}-{bin_info['bin'][1]:.1f}      | "
            f"{bin_info['n_samples']:7d} | "
            f"{bin_info['accuracy']:.4f}   | "
            f"{bin_info['confidence']:.4f}     | "
            f"{gap:+.2f} {marker}"
        )

    print("\n4. Applying Platt scaling...")
    # Split for calibration
    split = int(0.5 * n_samples)
    y_cal, y_test = y_true[:split], y_true[split:]
    prob_cal, prob_test = y_prob_overconfident[:split], y_prob_overconfident[split:]

    platt = PlattScaling()
    platt.fit(y_cal, prob_cal)
    prob_calibrated = platt.transform(prob_test)

    ece_before, _ = compute_calibration_error(y_test, prob_test)
    ece_after, _ = compute_calibration_error(y_test, prob_calibrated)
    bs_before = brier_score(y_test, prob_test)
    bs_after = brier_score(y_test, prob_calibrated)

    print(f"   Before calibration:")
    print(f"     ECE: {ece_before:.4f}, Brier: {bs_before:.4f}")
    print(f"   After Platt scaling:")
    print(f"     ECE: {ece_after:.4f}, Brier: {bs_after:.4f}")
    print(f"   Improvement: ECE reduced by {ece_before - ece_after:.4f}")

    print("\n5. Threshold optimization...")
    for metric in ["accuracy", "f1", "balanced_accuracy"]:
        thresh, score = find_optimal_threshold(y_test, prob_calibrated, metric)
        print(f"   Optimal for {metric:18s}: threshold={thresh:.2f}, score={score:.4f}")

    print("\n6. Precision-Recall tradeoff at different thresholds...")
    print("   Threshold | Precision | Recall | F1")
    print("   ----------|-----------|--------|-----")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (prob_calibrated >= thresh).astype(int)
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"   {thresh:.2f}       | {precision:.4f}    | {recall:.4f} | {f1:.4f}")

    print("\n7. Using sklearn calibration (if available)...")
    try:
        from sklearn.calibration import calibration_curve, CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, random_state=42)
        split = int(0.6 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Uncalibrated model
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        prob_uncal = clf.predict_proba(X_test)[:, 1]

        # Calibrated model
        clf_cal = CalibratedClassifierCV(LogisticRegression(), cv=5, method="sigmoid")
        clf_cal.fit(X_train, y_train)
        prob_cal = clf_cal.predict_proba(X_test)[:, 1]

        ece_uncal, _ = compute_calibration_error(y_test, prob_uncal)
        ece_cal, _ = compute_calibration_error(y_test, prob_cal)

        print(f"   sklearn LogisticRegression ECE: {ece_uncal:.4f}")
        print(f"   sklearn CalibratedClassifierCV ECE: {ece_cal:.4f}")

    except ImportError:
        print("   sklearn not available")


if __name__ == "__main__":
    run_demo()
