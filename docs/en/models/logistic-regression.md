# Logistic Regression

## 1. Interview Summary

**Key Points to Remember:**
- **Binary classifier** using sigmoid function
- **Decision boundary** is linear (hyperplane)
- Outputs **probabilities**, not just class labels
- Uses **cross-entropy loss** (log loss)
- No closed-form solution; requires **iterative optimization**
- Regularization (L1/L2) prevents overfitting

**Common Interview Questions:**
- "Why use logistic regression instead of linear regression for classification?"
- "Derive the gradient for logistic regression"
- "What is the decision boundary?"

---

## 2. Core Definitions

### Model

$$P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

### Decision Boundary
Predict class 1 if $P(y=1|x) > 0.5$, equivalently if $w^Tx + b > 0$.

The decision boundary is the hyperplane: $w^Tx + b = 0$

### Log-Odds (Logit)

$$\log\frac{P(y=1|x)}{P(y=0|x)} = w^Tx + b$$

Linear in features, hence "logistic regression".

### Loss Function (Binary Cross-Entropy)

$$\mathcal{L}(w) = -\frac{1}{n}\sum_{i=1}^n [y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

---

## 3. Math and Derivations

### Sigmoid Properties

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Key properties:**
- Range: $(0, 1)$
- $\sigma(0) = 0.5$
- $\sigma(-z) = 1 - \sigma(z)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### Gradient Derivation

For a single sample with features $x$ and label $y$:

$$\mathcal{L} = -[y\log(\sigma(w^Tx)) + (1-y)\log(1-\sigma(w^Tx))]$$

Let $z = w^Tx$ and $\hat{p} = \sigma(z)$.

$$\frac{\partial \mathcal{L}}{\partial z} = -\frac{y}{\hat{p}}\cdot\hat{p}(1-\hat{p}) + \frac{1-y}{1-\hat{p}}\cdot\hat{p}(1-\hat{p})$$

$$= -y(1-\hat{p}) + (1-y)\hat{p} = \hat{p} - y$$

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w} = (\hat{p} - y)x$$

**Gradient over all samples:**

$$\nabla_w \mathcal{L} = \frac{1}{n}\sum_{i=1}^n (\hat{p}_i - y_i)x_i = \frac{1}{n}X^T(\hat{p} - y)$$

### Maximum Likelihood Interpretation

Likelihood: $\prod_i \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}$

Log-likelihood: $\sum_i [y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$

Minimizing cross-entropy = Maximizing log-likelihood.

### Multi-class: Softmax Regression

For $K$ classes:

$$P(y=k|x) = \frac{e^{w_k^Tx}}{\sum_{j=1}^K e^{w_j^Tx}}$$

Loss: Cross-entropy over $K$ classes.

---

## 4. Algorithm Sketch

### Gradient Descent for Logistic Regression
```
Input: X (n×p), y (n×1), learning_rate η, iterations T
Initialize: w = 0, b = 0

For t = 1 to T:
    # Forward pass
    z = X @ w + b
    p_hat = sigmoid(z)

    # Compute gradient
    error = p_hat - y
    grad_w = (1/n) * X.T @ error
    grad_b = (1/n) * sum(error)

    # Update
    w = w - η * grad_w
    b = b - η * grad_b

Return w, b
```

### Newton-Raphson (Faster Convergence)
```
Input: X, y, iterations T
Initialize: w = 0

For t = 1 to T:
    p_hat = sigmoid(X @ w)
    gradient = X.T @ (p_hat - y)

    # Hessian: H = X.T @ diag(p*(1-p)) @ X
    S = diag(p_hat * (1 - p_hat))
    Hessian = X.T @ S @ X

    # Newton update
    w = w - inv(Hessian) @ gradient

Return w
```

### Regularized Logistic Regression
```
# L2 Regularization
Loss = cross_entropy + (λ/2) * ||w||²
grad_w = (1/n) * X.T @ (p_hat - y) + λ * w

# L1 Regularization (requires special solver)
Loss = cross_entropy + λ * ||w||₁
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Perfect separation | Weights diverge to infinity | Use regularization |
| Using accuracy as loss | Not differentiable | Use cross-entropy |
| Forgetting to standardize | Slow convergence, unfair regularization | Standardize features |
| Threshold always 0.5 | May not be optimal | Tune based on precision-recall tradeoff |
| Ignoring class imbalance | Biased toward majority class | Weighted loss, resampling |

### Why Not Linear Regression for Classification?

| Issue | Linear Regression | Logistic Regression |
|-------|-------------------|---------------------|
| Output range | $(-\infty, +\infty)$ | $(0, 1)$ (probabilities) |
| Outliers | Sensitive, shifts boundary | Robust |
| Loss function | MSE (not for classification) | Cross-entropy (proper) |
| Interpretation | No probabilistic meaning | Probability of class |

---

## 6. Mini Example

### Python Example: Logistic Regression from Scratch

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression_gd(X, y, lr=0.1, epochs=1000, reg=0.01):
    """Gradient descent with L2 regularization."""
    n, p = X.shape
    w = np.zeros(p)
    b = 0

    for _ in range(epochs):
        z = X @ w + b
        p_hat = sigmoid(z)

        # Gradients
        dw = (1/n) * X.T @ (p_hat - y) + reg * w
        db = (1/n) * np.sum(p_hat - y)

        # Update
        w -= lr * dw
        b -= lr * db

    return w, b

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)

# Generate data
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train
w, b = logistic_regression_gd(X_train_s, y_train)

# Evaluate
train_acc = np.mean(predict(X_train_s, w, b) == y_train)
test_acc = np.mean(predict(X_test_s, w, b) == y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

# Output:
# Train accuracy: 0.872
# Test accuracy: 0.850
```

### Decision Boundary Visualization

```python
import matplotlib.pyplot as plt

# 2D example for visualization
X_2d, y_2d = make_classification(n_samples=200, n_features=2,
                                  n_redundant=0, random_state=42)
scaler_2d = StandardScaler()
X_2d_s = scaler_2d.fit_transform(X_2d)

w_2d, b_2d = logistic_regression_gd(X_2d_s, y_2d)

# Plot decision boundary: w[0]*x1 + w[1]*x2 + b = 0
# => x2 = -(w[0]*x1 + b) / w[1]
x1_range = np.linspace(X_2d_s[:, 0].min()-1, X_2d_s[:, 0].max()+1, 100)
x2_boundary = -(w_2d[0] * x1_range + b_2d) / w_2d[1]

plt.scatter(X_2d_s[y_2d==0, 0], X_2d_s[y_2d==0, 1], label='Class 0')
plt.scatter(X_2d_s[y_2d==1, 0], X_2d_s[y_2d==1, 1], label='Class 1')
plt.plot(x1_range, x2_boundary, 'k--', label='Decision Boundary')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Why can't we use linear regression for classification?</strong></summary>

Linear regression has issues for classification:
1. **Output range**: Predictions can be < 0 or > 1, not valid probabilities
2. **Outlier sensitivity**: Extreme values shift the decision boundary
3. **Loss function**: MSE penalizes confident correct predictions
4. **No probabilistic interpretation**: Can't interpret as class probability

Logistic regression uses sigmoid to bound outputs to (0,1) and cross-entropy loss which is proper for classification.
</details>

<details>
<summary><strong>Q2: Derive the gradient of the log loss with respect to weights.</strong></summary>

Loss for sample $i$: $\mathcal{L}_i = -[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$

Where $\hat{p}_i = \sigma(w^Tx_i)$.

Using chain rule with $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$\frac{\partial \mathcal{L}_i}{\partial w} = (\hat{p}_i - y_i)x_i$$

Over all samples: $\nabla_w\mathcal{L} = \frac{1}{n}X^T(\hat{p} - y)$

Elegant result: gradient = predictions - labels, scaled by features.
</details>

<details>
<summary><strong>Q3: What is the decision boundary of logistic regression?</strong></summary>

The decision boundary is where $P(y=1|x) = 0.5$, which occurs when:

$$w^Tx + b = 0$$

This is a **linear** hyperplane in feature space. In 2D: a line. In 3D: a plane.

Points on one side: $w^Tx + b > 0$ → predict class 1
Points on other side: $w^Tx + b < 0$ → predict class 0
</details>

<details>
<summary><strong>Q4: What happens with perfect separation?</strong></summary>

When classes are perfectly separable:
- Any boundary that separates them achieves zero training loss
- Weights grow toward infinity to maximize confidence
- Model becomes overconfident, poor calibration
- Gradient descent never converges

**Solutions**:
- L2 regularization (constrains weight magnitude)
- Early stopping
- Bayesian logistic regression
</details>

<details>
<summary><strong>Q5: What is the relationship between logistic regression and log-odds?</strong></summary>

Logistic regression models the **log-odds** (logit) as linear in features:

$$\log\frac{P(y=1|x)}{P(y=0|x)} = w^Tx + b$$

This means:
- Each unit increase in $x_j$ changes log-odds by $w_j$
- Equivalently, multiplies odds by $e^{w_j}$
- Provides interpretable coefficients for feature importance
</details>

<details>
<summary><strong>Q6: How does multi-class logistic regression (softmax) work?</strong></summary>

For $K$ classes, use softmax:

$$P(y=k|x) = \frac{e^{w_k^Tx}}{\sum_{j=1}^K e^{w_j^Tx}}$$

Properties:
- Outputs sum to 1 (valid probability distribution)
- Reduces to sigmoid for K=2 (one-vs-rest)
- Loss: categorical cross-entropy
- Gradient: similar form, $(\hat{p} - y)$ per class
</details>

---

## 8. References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 4.3.
2. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
3. Ng, A. (2012). "Machine Learning." Coursera. Logistic Regression lectures.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Section 4.4.
5. Cox, D. R. (1958). "The Regression Analysis of Binary Sequences." JRSS-B.
