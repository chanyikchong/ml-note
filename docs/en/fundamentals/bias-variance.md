# Bias-Variance Tradeoff

## 1. Interview Summary

**Key Points to Remember:**
- **Bias**: Error from incorrect assumptions; underfitting
- **Variance**: Error from sensitivity to training data; overfitting
- **Tradeoff**: Reducing one often increases the other
- **Model complexity** controls the tradeoff
- **Regularization** reduces variance at cost of bias

**Common Interview Questions:**
- "Explain the bias-variance tradeoff"
- "How do you know if your model is underfitting vs overfitting?"
- "How does model complexity affect bias and variance?"

---

## 2. Core Definitions

### Bias
The error introduced by approximating a complex real-world problem with a simplified model.

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$$

**High Bias (Underfitting):**
- Model too simple
- Can't capture underlying pattern
- Both training and test error are high

### Variance
The error from sensitivity to fluctuations in the training set.

$$\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$

**High Variance (Overfitting):**
- Model too complex
- Fits noise in training data
- Low training error, high test error

### Irreducible Error
Noise inherent in the data that cannot be eliminated.

$$\sigma^2 = \text{Var}[\epsilon]$$

---

## 3. Math and Derivations

### Bias-Variance Decomposition

For regression with squared loss, the expected prediction error can be decomposed:

**Setup:**
- True function: $y = f(x) + \epsilon$, where $\mathbb{E}[\epsilon] = 0$, $\text{Var}[\epsilon] = \sigma^2$
- Learned function: $\hat{f}(x)$ trained on dataset $\mathcal{D}$

**Derivation:**

$$\begin{aligned}
\mathbb{E}_\mathcal{D}[(y - \hat{f}(x))^2] &= \mathbb{E}_\mathcal{D}[(f(x) + \epsilon - \hat{f}(x))^2] \\
&= \mathbb{E}_\mathcal{D}[(f(x) - \hat{f}(x))^2] + \mathbb{E}[\epsilon^2] + 2\mathbb{E}_\mathcal{D}[(f(x) - \hat{f}(x))\epsilon] \\
&= \mathbb{E}_\mathcal{D}[(f(x) - \hat{f}(x))^2] + \sigma^2
\end{aligned}$$

The first term decomposes further:

$$\begin{aligned}
\mathbb{E}_\mathcal{D}[(f(x) - \hat{f}(x))^2] &= (f(x) - \mathbb{E}_\mathcal{D}[\hat{f}(x)])^2 + \mathbb{E}_\mathcal{D}[(\hat{f}(x) - \mathbb{E}_\mathcal{D}[\hat{f}(x)])^2] \\
&= \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)]
\end{aligned}$$

**Final Result:**

$$\boxed{\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}}$$

### Model Complexity and Tradeoff

| Complexity | Bias | Variance | Training Error | Test Error |
|------------|------|----------|----------------|------------|
| Low | High | Low | High | High |
| Optimal | Medium | Medium | Low | Low |
| High | Low | High | Very Low | High |

---

## 4. Algorithm Sketch

### Diagnosing Bias vs Variance

```
1. Train model on training set
2. Compute training error E_train
3. Compute validation/test error E_test

If E_train is high and E_test is high:
    → HIGH BIAS (underfitting)
    → Solution: More complex model, more features

If E_train is low and E_test is high:
    → HIGH VARIANCE (overfitting)
    → Solution: Regularization, more data, simpler model

If E_train is low and E_test is low:
    → Good fit! (may still be improvable)
```

### Learning Curves Analysis

```
1. Train model with increasing data sizes [n_1, n_2, ..., n_k]
2. For each size n_i:
   - Train on n_i samples
   - Record training and validation errors
3. Plot both curves vs sample size

High Bias pattern:
   - Both curves converge to HIGH error
   - Adding data doesn't help much

High Variance pattern:
   - Training error << validation error
   - Gap decreases with more data
   - More data helps!
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Confusing bias/variance with train/test error | Related but not identical | Bias-variance is theoretical decomposition |
| Using only test error to diagnose | Can't distinguish bias from variance | Compare train vs test error |
| Adding complexity without regularization | Variance explodes | Always consider regularization with complex models |
| Over-regularizing | Too much penalization | Tune regularization strength via validation |
| Ignoring irreducible error | Expecting perfect predictions | Understand data noise limits achievable accuracy |

### Visual Diagnosis

```
Training Error vs Test Error:

High Bias:          High Variance:       Good Fit:
Error               Error                Error
  |  ----test       |    ----test        |    ----test
  |  ----train      |                    |    ----train
  |                 |    ----train       |
  +--------→        +--------→           +--------→
    Complexity         Complexity           Complexity
```

---

## 6. Mini Example

### Python Example: Observing Bias-Variance Tradeoff

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Generate noisy data from true function
np.random.seed(42)
n_samples = 100
X = np.sort(np.random.uniform(0, 1, n_samples))
y_true = np.sin(2 * np.pi * X)
y = y_true + np.random.normal(0, 0.3, n_samples)

X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Test different polynomial degrees
degrees = [1, 3, 10, 20]
train_errors, test_errors = [], []

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = np.mean((y_train - train_pred)**2)
    test_mse = np.mean((y_test - test_pred)**2)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

    print(f"Degree {degree:2d}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")

# Output:
# Degree  1: Train MSE = 0.4521, Test MSE = 0.5307  (High Bias)
# Degree  3: Train MSE = 0.0892, Test MSE = 0.1124  (Good Fit)
# Degree 10: Train MSE = 0.0731, Test MSE = 0.1456  (Starting to Overfit)
# Degree 20: Train MSE = 0.0412, Test MSE = 0.9823  (High Variance)
```

**Interpretation:**
- Degree 1: Underfitting (high bias) - both errors high
- Degree 3: Good balance - both errors low
- Degree 20: Overfitting (high variance) - train low, test very high

---

## 7. Quiz

<details>
<summary><strong>Q1: What is the bias-variance tradeoff?</strong></summary>

The bias-variance tradeoff describes the tension between two sources of error:
- **Bias**: Error from overly simplistic models that can't capture the true pattern
- **Variance**: Error from overly complex models that fit noise in training data

As model complexity increases, bias typically decreases but variance increases. The optimal model balances both to minimize total error.
</details>

<details>
<summary><strong>Q2: How can you tell if a model is underfitting vs overfitting?</strong></summary>

**Underfitting (High Bias):**
- High training error
- High test error (similar to training)
- Model too simple for the data

**Overfitting (High Variance):**
- Low training error
- High test error (much higher than training)
- Model too complex, fits noise
</details>

<details>
<summary><strong>Q3: Write the bias-variance decomposition formula.</strong></summary>

For squared loss:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

Where:
- $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$
- $\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$
- $\sigma^2$ is irreducible noise
</details>

<details>
<summary><strong>Q4: How do you reduce high variance?</strong></summary>

Strategies to reduce variance:
1. **Regularization** (L1, L2, dropout)
2. **More training data**
3. **Simpler model** (fewer parameters)
4. **Ensemble methods** (bagging averages out variance)
5. **Early stopping**
6. **Feature selection** (reduce dimensionality)
</details>

<details>
<summary><strong>Q5: How do you reduce high bias?</strong></summary>

Strategies to reduce bias:
1. **More complex model** (higher capacity)
2. **Add more features** (polynomial features, interactions)
3. **Reduce regularization**
4. **Use a different model family** (more expressive)
5. **Boosting** (iteratively corrects bias)
</details>

<details>
<summary><strong>Q6: Why can't we reduce bias and variance simultaneously?</strong></summary>

Reducing bias requires more model flexibility to capture complex patterns, but more flexibility means the model can also fit noise (increasing variance). Conversely, constraining the model to reduce variance limits its ability to fit the true function (increasing bias). The only way to reduce both is to get more high-quality training data, which reduces variance without requiring simpler models.
</details>

---

## 8. References

1. Geman, S., Bienenstock, E., & Doursat, R. (1992). "Neural Networks and the Bias/Variance Dilemma." Neural Computation.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 7.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 3.2.
4. James, G., et al. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 2.
5. Domingos, P. (2000). "A Unified Bias-Variance Decomposition." ICML.
