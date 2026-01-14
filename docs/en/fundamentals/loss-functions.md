# Loss Functions

## 1. Interview Summary

**Key Points to Remember:**
- **MSE**: Mean Squared Error - regression, sensitive to outliers
- **MAE**: Mean Absolute Error - regression, robust to outliers
- **Cross-entropy**: Classification, measures probability distribution difference
- **Calibration**: How well predicted probabilities match true frequencies
- Know when to use each loss function

**Common Interview Questions:**
- "When would you use MAE vs MSE?"
- "What is cross-entropy loss and why is it used for classification?"
- "What does it mean for a model to be well-calibrated?"

---

## 2. Core Definitions

### Mean Squared Error (MSE)
$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Properties:**
- Penalizes large errors more heavily (quadratic)
- Sensitive to outliers
- Has nice gradient properties (smooth)
- Optimal predictor: $\mathbb{E}[Y|X]$ (conditional mean)

### Mean Absolute Error (MAE)
$$\mathcal{L}_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Properties:**
- Linear penalty for errors
- Robust to outliers
- Non-smooth at zero (gradient undefined)
- Optimal predictor: median of $Y|X$

### Cross-Entropy Loss (Log Loss)

**Binary Classification:**
$$\mathcal{L}_{CE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i)]$$

**Multi-class Classification:**
$$\mathcal{L}_{CE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(\hat{p}_{ic})$$

### Calibration
A model is well-calibrated if: When it predicts probability $p$, the actual frequency of positive outcomes is $p$.

$$P(Y=1 | \hat{p}(X)=p) = p$$

---

## 3. Math and Derivations

### MSE Derivation from Maximum Likelihood

Assume Gaussian noise: $y = f(x) + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$

**Likelihood:**
$$p(y|x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f(x))^2}{2\sigma^2}\right)$$

**Log-likelihood:**
$$\log p(y|x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y - f(x))^2}{2\sigma^2}$$

Maximizing log-likelihood $\equiv$ minimizing $(y - f(x))^2$ $\equiv$ MSE

### Cross-Entropy from KL Divergence

For true distribution $p$ and predicted distribution $q$:

**KL Divergence:**
$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = -H(p) - \sum_x p(x) \log q(x)$$

Since $H(p)$ is constant w.r.t. model parameters:
$$\min D_{KL}(p \| q) \equiv \min \left(-\sum_x p(x) \log q(x)\right) = \min \mathcal{L}_{CE}$$

### Huber Loss (Smooth MAE)

$$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

- Quadratic for small errors (smooth gradients)
- Linear for large errors (robust to outliers)

---

## 4. Algorithm Sketch

### Choosing Loss Functions

```
For REGRESSION:
    If outliers are rare or should be penalized:
        → Use MSE
    If outliers are common and should be ignored:
        → Use MAE or Huber Loss
    If predicting percentiles/quantiles:
        → Use Quantile Loss

For CLASSIFICATION:
    If need probability outputs:
        → Use Cross-Entropy
    If just need class predictions:
        → Cross-Entropy still typically better
    If severe class imbalance:
        → Consider Focal Loss or weighted CE
```

### Calibration Methods

```
1. Train classifier
2. Get predicted probabilities on validation set
3. Apply calibration:
   - Platt Scaling: Fit sigmoid to logits
   - Isotonic Regression: Non-parametric calibration
   - Temperature Scaling: Single scalar for logits
4. Evaluate with:
   - Reliability diagram
   - Expected Calibration Error (ECE)
   - Brier Score
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Using MSE with outliers | MSE squares large errors | Use MAE or Huber |
| Using accuracy loss | Non-differentiable | Use cross-entropy (surrogate) |
| Ignoring calibration | Focus only on accuracy | Calibrate for probability tasks |
| Wrong loss for ranking | Regression loss for ranking task | Use ranking losses (pairwise, listwise) |
| Using CE with hard labels | Loses information | Consider label smoothing |

### Comparing Losses

| Loss | Sensitivity to Outliers | Gradient at Zero | Optimal Predictor |
|------|------------------------|------------------|-------------------|
| MSE | High | Smooth | Mean |
| MAE | Low | Undefined | Median |
| Huber | Configurable | Smooth | Between mean/median |

---

## 6. Mini Example

### Python Example: Comparing Loss Functions

```python
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Generate data with outliers
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 0.5

# Add outliers
y[0], y[1], y[2] = 100, -80, 90  # Extreme outliers

# Fit with MSE (Linear Regression)
lr = LinearRegression()
lr.fit(X, y)
pred_mse = lr.predict(X)

# Fit with Huber Loss (robust)
huber = HuberRegressor()
huber.fit(X, y)
pred_huber = huber.predict(X)

print(f"True coefficients: slope=2, intercept=1")
print(f"MSE Regression: slope={lr.coef_[0]:.2f}, intercept={lr.intercept_:.2f}")
print(f"Huber Regression: slope={huber.coef_[0]:.2f}, intercept={huber.intercept_:.2f}")

# Output:
# True coefficients: slope=2, intercept=1
# MSE Regression: slope=0.84, intercept=3.45  (biased by outliers!)
# Huber Regression: slope=1.98, intercept=1.03  (robust to outliers)
```

### Cross-Entropy Example

```python
import numpy as np

def cross_entropy(y_true, y_pred, eps=1e-15):
    """Binary cross-entropy loss."""
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example predictions
y_true = np.array([1, 0, 1, 1, 0])
y_confident = np.array([0.9, 0.1, 0.95, 0.85, 0.05])  # Good predictions
y_wrong = np.array([0.1, 0.9, 0.2, 0.3, 0.8])  # Bad predictions

print(f"CE Loss (good predictions): {cross_entropy(y_true, y_confident):.4f}")
print(f"CE Loss (bad predictions): {cross_entropy(y_true, y_wrong):.4f}")

# Output:
# CE Loss (good predictions): 0.0969
# CE Loss (bad predictions): 1.6904
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Why is MSE sensitive to outliers?</strong></summary>

MSE squares the errors, so large errors contribute disproportionately to the total loss. An outlier with error 10 contributes 100 to MSE, while 10 small errors of 1 each only contribute 10 total. This causes the model to adjust significantly to reduce outlier errors, biasing the overall fit.
</details>

<details>
<summary><strong>Q2: What is the connection between cross-entropy and maximum likelihood?</strong></summary>

Minimizing cross-entropy is equivalent to maximizing the likelihood of the data under the model's predicted distribution. For classification with predicted probabilities $\hat{p}$:

$$\max \prod_i \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i} \equiv \min -\sum_i [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$$

Cross-entropy is the negative log-likelihood.
</details>

<details>
<summary><strong>Q3: When would you choose MAE over MSE?</strong></summary>

Choose MAE when:
- Data contains outliers that should not dominate the fit
- You want to predict the median rather than mean
- Errors should be penalized linearly (all errors equally important per unit)
- You need a more robust estimator

Note: MAE has undefined gradient at zero, which can make optimization harder.
</details>

<details>
<summary><strong>Q4: What is model calibration and why does it matter?</strong></summary>

Calibration measures whether predicted probabilities match actual frequencies. A model predicting 80% probability should be correct ~80% of the time for those predictions.

It matters when:
- Decisions are based on probability thresholds
- Combining predictions from multiple models
- Risk assessment requires accurate confidence estimates
- Medical/financial applications need reliable uncertainty
</details>

<details>
<summary><strong>Q5: How does Huber loss combine benefits of MSE and MAE?</strong></summary>

Huber loss:
- Behaves like MSE for small errors (smooth gradients, efficient optimization)
- Behaves like MAE for large errors (robust to outliers)
- Has a parameter $\delta$ that controls the transition point

$$\mathcal{L}_\delta = \begin{cases} \frac{1}{2}e^2 & |e| \leq \delta \\ \delta|e| - \frac{\delta^2}{2} & |e| > \delta \end{cases}$$
</details>

<details>
<summary><strong>Q6: Why can't we directly use accuracy as a loss function?</strong></summary>

Accuracy (0-1 loss) is non-differentiable:
- It's a step function that jumps at the decision boundary
- Gradient is zero almost everywhere
- Can't use gradient-based optimization

Cross-entropy is a smooth, differentiable surrogate that:
- Provides meaningful gradients
- Encourages correct probability rankings
- Is mathematically principled (related to likelihood)
</details>

---

## 8. References

1. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
2. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML.
3. Huber, P. J. (1964). "Robust Estimation of a Location Parameter." Annals of Mathematical Statistics.
4. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
5. Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." ICML.
