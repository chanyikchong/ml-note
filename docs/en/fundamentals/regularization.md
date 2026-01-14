# Regularization

## 1. Interview Summary

**Key Points to Remember:**
- **L2 (Ridge)**: Penalizes large weights, encourages small weights
- **L1 (Lasso)**: Encourages sparsity, feature selection
- **Elastic Net**: Combines L1 and L2
- **Early Stopping**: Stop training when validation error increases
- Regularization reduces variance (overfitting) at cost of bias

**Common Interview Questions:**
- "What's the difference between L1 and L2 regularization?"
- "Why does L1 produce sparse solutions?"
- "How does early stopping act as regularization?"

---

## 2. Core Definitions

### L2 Regularization (Ridge / Weight Decay)
Add squared magnitude of weights to loss:

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda \sum_i w_i^2 = \mathcal{L}_{data} + \lambda \|w\|_2^2$$

**Effects:**
- Shrinks all weights toward zero
- Keeps all features, none exactly zero
- Equivalent to Gaussian prior on weights

### L1 Regularization (Lasso)
Add absolute magnitude of weights to loss:

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda \sum_i |w_i| = \mathcal{L}_{data} + \lambda \|w\|_1$$

**Effects:**
- Produces sparse solutions (many weights exactly zero)
- Built-in feature selection
- Equivalent to Laplace prior on weights

### Elastic Net
Combines L1 and L2:

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$$

### Early Stopping
Stop training when validation error stops improving.

---

## 3. Math and Derivations

### Why L1 Produces Sparsity

Consider minimizing: $\min_w (w - c)^2 + \lambda|w|$

**Solution:**

$$w^* = \begin{cases}
c - \lambda/2 & \text{if } c > \lambda/2 \\
0 & \text{if } |c| \leq \lambda/2 \\
c + \lambda/2 & \text{if } c < -\lambda/2
\end{cases}$$

The solution is **exactly zero** when $|c| \leq \lambda/2$ (soft-thresholding).

For L2: $w^* = c/(1 + \lambda)$ — never exactly zero!

### Bayesian Interpretation

**L2 Regularization:**
Prior: $p(w) \propto \exp(-\lambda\|w\|_2^2)$ — Gaussian with variance $1/(2\lambda)$

**L1 Regularization:**
Prior: $p(w) \propto \exp(-\lambda\|w\|_1)$ — Laplace distribution

MAP estimate with regularization = MLE with prior.

### Ridge Regression Closed Form

For linear regression with L2:

$$\hat{w}_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$$

Compare to OLS: $\hat{w}_{OLS} = (X^TX)^{-1}X^Ty$

The $\lambda I$ term makes inversion numerically stable even when $X^TX$ is singular.

---

## 4. Algorithm Sketch

### Choosing Regularization Strength

```
1. Define grid of λ values (log scale): [1e-4, 1e-3, ..., 1e1]
2. For each λ:
   a. Train model with regularization
   b. Evaluate on validation set
3. Select λ with best validation performance
4. (Optional) Retrain on train+val with selected λ
5. Evaluate on test set
```

### Early Stopping

```
Initialize: best_val_loss = ∞, patience_counter = 0
For each epoch:
    Train on training data
    Compute validation loss

    If val_loss < best_val_loss:
        best_val_loss = val_loss
        Save model checkpoint
        patience_counter = 0
    Else:
        patience_counter += 1

    If patience_counter >= patience:
        Stop training
        Load best checkpoint
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Regularizing bias terms | Bias doesn't contribute to overfitting | Exclude bias from penalty |
| Same λ for all features | Features may have different scales | Standardize features first |
| Too strong regularization | Underfitting | Use validation to tune λ |
| Not using early stopping | Training too long wastes time | Monitor validation loss |
| L1 with correlated features | Arbitrarily selects one | Use Elastic Net instead |

### L1 vs L2 Comparison

| Property | L1 (Lasso) | L2 (Ridge) |
|----------|------------|------------|
| Sparsity | Yes (exact zeros) | No |
| Feature selection | Built-in | No |
| Correlated features | Picks one arbitrarily | Shrinks all equally |
| Solution uniqueness | May not be unique | Always unique |
| Optimization | Subgradient methods | Closed form for linear |

---

## 6. Mini Example

### Python Example: L1 vs L2 Regularization

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Generate data with some irrelevant features
np.random.seed(42)
X, y, true_coef = make_regression(
    n_samples=100, n_features=20, n_informative=5,
    noise=10, coef=True, random_state=42
)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

ridge.fit(X_scaled, y)
lasso.fit(X_scaled, y)
elastic.fit(X_scaled, y)

# Count non-zero coefficients
print(f"Ridge non-zero coefs: {np.sum(ridge.coef_ != 0)}")  # All 20
print(f"Lasso non-zero coefs: {np.sum(np.abs(lasso.coef_) > 1e-6)}")  # ~5-7
print(f"Elastic non-zero coefs: {np.sum(np.abs(elastic.coef_) > 1e-6)}")  # ~7-10

print(f"\nTrue informative features: 5")
print(f"Ridge shrinks all, Lasso selects subset")

# Output:
# Ridge non-zero coefs: 20
# Lasso non-zero coefs: 6
# Elastic non-zero coefs: 8
```

### Early Stopping Example

```python
import numpy as np

def train_with_early_stopping(X_train, y_train, X_val, y_val,
                              patience=5, max_epochs=100):
    """Simulate training with early stopping."""
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    # Simulate training (losses would come from actual training)
    train_losses = np.exp(-np.arange(max_epochs) / 20) + 0.1
    val_losses = np.exp(-np.arange(max_epochs) / 30) + 0.15
    val_losses[40:] += np.arange(max_epochs - 40) * 0.005  # Overfitting

    for epoch in range(max_epochs):
        val_loss = val_losses[epoch]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint here
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            print(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
            return best_epoch

    return best_epoch

# Simulate
best = train_with_early_stopping(None, None, None, None, patience=10)
# Output: Early stopping at epoch ~50, Best epoch: ~40
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Why does L1 regularization produce sparse solutions?</strong></summary>

L1's penalty $|w|$ creates a non-differentiable point at zero with a constant subgradient. The optimization solution involves soft-thresholding:

$$w^* = \text{sign}(c) \cdot \max(|c| - \lambda/2, 0)$$

When the unregularized solution $c$ is smaller than $\lambda/2$, the optimal $w^*$ is exactly zero. L2's smooth penalty $w^2$ only shrinks weights toward zero but never reaches it.
</details>

<details>
<summary><strong>Q2: What is the Bayesian interpretation of regularization?</strong></summary>

Regularization corresponds to placing a prior on the weights:
- **L2** = Gaussian prior: $p(w) \propto \exp(-\lambda\|w\|^2)$
- **L1** = Laplace prior: $p(w) \propto \exp(-\lambda\|w\|_1)$

The regularized loss is the negative log posterior:

$$\mathcal{L}_{reg} = -\log p(y|X,w) - \log p(w) = \mathcal{L}_{data} + \text{regularization}$$

Minimizing regularized loss = finding MAP estimate.
</details>

<details>
<summary><strong>Q3: When would you choose Elastic Net over Lasso?</strong></summary>

Use Elastic Net when:
- Features are **correlated**: Lasso arbitrarily picks one, Elastic Net keeps groups
- **More features than samples**: Lasso selects at most n features, Elastic Net can select more
- You want **some sparsity** but also stability of L2
- There are **groups of correlated features** you want to select together
</details>

<details>
<summary><strong>Q4: How does early stopping act as regularization?</strong></summary>

Early stopping limits the effective model capacity by:
- Restricting how far parameters can move from initialization
- Preventing the model from fitting noise (which takes longer)
- Acting similarly to L2 regularization in linear models

The number of training steps acts like an inverse regularization strength: fewer steps = stronger regularization.
</details>

<details>
<summary><strong>Q5: Should you regularize bias terms?</strong></summary>

Generally **no**. Reasons:
- Bias terms shift predictions without affecting model complexity
- Regularizing bias can hurt when true mean is non-zero
- Bias doesn't contribute to overfitting in the same way weights do
- Standard practice in sklearn, PyTorch, etc. is to exclude bias
</details>

<details>
<summary><strong>Q6: How do you choose the regularization strength λ?</strong></summary>

1. **Cross-validation**: Try multiple λ values, select best on validation
2. **Grid search**: Log-scale grid (e.g., [1e-4, 1e-3, ..., 10])
3. **Regularization path**: Efficient algorithms that compute solutions for all λ
4. **Information criteria**: AIC, BIC for model selection
5. **Bayesian methods**: Treat λ as hyperparameter with prior

Common approach: 5-fold CV over log-spaced λ values.
</details>

---

## 8. References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapters 3, 7.
2. Zou, H., & Hastie, T. (2005). "Regularization and Variable Selection via the Elastic Net." JRSS-B.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 3.3.
4. Prechelt, L. (1998). "Early Stopping - But When?" Neural Networks: Tricks of the Trade.
5. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Chapter 11.
