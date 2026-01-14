# Generalization & Capacity

## 1. Interview Summary

**Key Points to Remember:**
- **Generalization**: Performance on unseen data, not just training data
- **Capacity**: Model's ability to fit a wide variety of functions
- **VC Dimension**: Theoretical measure of capacity
- More capacity → can fit more complex patterns, but risk overfitting
- Generalization gap = test error - training error

**Common Interview Questions:**
- "What is generalization and why does it matter?"
- "Explain VC dimension intuitively"
- "How does model capacity relate to overfitting?"

---

## 2. Core Definitions

### Generalization
The ability of a model to perform well on new, unseen data from the same distribution as the training data.

$$\text{Generalization Error} = \mathbb{E}_{(x,y)\sim P}[\mathcal{L}(f(x), y)]$$

### Capacity
The richness of the function class a model can represent.

**Factors affecting capacity:**
- Number of parameters
- Model architecture
- Regularization strength

### VC Dimension
The maximum number of points that can be shattered (perfectly classified for all possible labelings) by a hypothesis class.

**Examples:**
- Linear classifier in $\mathbb{R}^d$: VC dimension = $d + 1$
- Finite hypothesis class $|\mathcal{H}|$: VC dimension ≤ $\log_2|\mathcal{H}|$

### Generalization Gap

$$\text{Gap} = \mathcal{L}_{test} - \mathcal{L}_{train}$$

---

## 3. Math and Derivations

### VC Dimension Bound

For a hypothesis class with VC dimension $d_{VC}$, with probability at least $1 - \delta$:

$$R(h) \leq \hat{R}(h) + \sqrt{\frac{d_{VC}(\ln(2n/d_{VC}) + 1) + \ln(4/\delta)}{n}}$$

where:
- $R(h)$: True risk (generalization error)
- $\hat{R}(h)$: Empirical risk (training error)
- $n$: Number of training samples

**Key Insight**: Generalization gap scales as $O(\sqrt{d_{VC}/n})$

### Rademacher Complexity

A data-dependent measure of capacity:

$$\mathcal{R}_n(\mathcal{H}) = \mathbb{E}_{\sigma}\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^{n} \sigma_i h(x_i)\right]$$

where $\sigma_i$ are Rademacher random variables ($\pm 1$ with equal probability).

**Generalization Bound:**

$$R(h) \leq \hat{R}(h) + 2\mathcal{R}_n(\mathcal{H}) + O\left(\sqrt{\frac{\ln(1/\delta)}{n}}\right)$$

### PAC Learning Framework

A concept class $\mathcal{C}$ is PAC-learnable if there exists an algorithm that, for any:
- $\epsilon > 0$ (accuracy parameter)
- $\delta > 0$ (confidence parameter)

Outputs a hypothesis $h$ such that with probability $\geq 1-\delta$:

$$P(h(x) \neq c(x)) \leq \epsilon$$

using $m = \text{poly}(1/\epsilon, 1/\delta, n, \text{size}(c))$ samples.

---

## 4. Algorithm Sketch

### Assessing Generalization

```
1. Split data: train / validation / test
2. Train model on training set
3. Monitor:
   - Training loss (should decrease)
   - Validation loss (should decrease, then may increase)
4. Compute generalization gap:
   gap = validation_loss - training_loss
5. If gap is large:
   → Model is overfitting
   → Reduce capacity or add regularization
6. If training loss is high:
   → Model is underfitting
   → Increase capacity
```

### Model Selection with Capacity Control

```
1. Define model family with varying capacity
   (e.g., polynomial degree, number of layers, hidden units)
2. For each capacity level:
   a. Train model
   b. Evaluate on validation set
3. Plot train/val error vs capacity
4. Select model at the "elbow" where:
   - Validation error is minimized
   - Generalization gap is acceptable
5. Final evaluation on test set
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Confusing VC dim with # params | VC dim can be < or > # params | Understand VC dim measures expressiveness |
| Ignoring double descent | Modern deep learning can break classical curves | Be aware of interpolation regime |
| Over-relying on theory | VC bounds often very loose | Use validation-based model selection |
| Testing too often | Overfitting to test set | Use test set only once |
| Assuming i.i.d. always holds | Real data may have distribution shift | Validate on realistic held-out data |

### Classical vs Modern Generalization

**Classical View (VC Theory):**
```
Error
  |     ____
  |    /    \_____ test error
  |   /
  |  /______ training error
  +----------------→ Capacity
        ↑
    Sweet spot
```

**Modern View (Double Descent):**
```
Error
  |   __
  |  /  \
  | /    \______ test error
  |/
  |______ training error
  +------------------→ Capacity
       ↑         ↑
   Classical   Over-parameterized
   regime      (interpolation)
```

---

## 6. Mini Example

### Python Example: Generalization vs Capacity

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
n = 100
X = np.random.uniform(-3, 3, n).reshape(-1, 1)
y = np.sin(X.squeeze()) + np.random.normal(0, 0.3, n)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Test different capacities (polynomial degrees)
degrees = range(1, 20)
train_errors = []
test_errors = []

for d in degrees:
    poly = PolynomialFeatures(d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)

    train_errors.append(np.mean((y_train - train_pred)**2))
    test_errors.append(np.mean((y_test - test_pred)**2))

# Find optimal capacity
best_degree = degrees[np.argmin(test_errors)]
best_gap = test_errors[np.argmin(test_errors)] - train_errors[np.argmin(test_errors)]

print(f"Optimal degree: {best_degree}")
print(f"Train MSE at optimal: {train_errors[best_degree-1]:.4f}")
print(f"Test MSE at optimal: {test_errors[best_degree-1]:.4f}")
print(f"Generalization gap: {best_gap:.4f}")

# Output:
# Optimal degree: 5
# Train MSE at optimal: 0.0721
# Test MSE at optimal: 0.1205
# Generalization gap: 0.0484
```

### VC Dimension Example

```python
# VC Dimension of linear classifiers in R^d
def vc_dimension_linear(d):
    """VC dimension of linear classifier in d dimensions."""
    return d + 1

# Example: 2D linear classifier can shatter 3 points
print(f"VC dim of linear classifier in R^2: {vc_dimension_linear(2)}")
# Output: 3

# This means we can find a linear classifier for ANY labeling of 3 points
# (in general position), but there exists some 4 points we cannot shatter.
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is generalization and why is it the goal of ML?</strong></summary>

Generalization is a model's ability to perform well on new, unseen data from the same distribution as training data. It's the goal because:

- We care about predictions on future data, not memorizing training data
- A model that only works on training data is useless in practice
- Good generalization means the model learned true patterns, not noise
- It's what distinguishes learning from memorization
</details>

<details>
<summary><strong>Q2: Explain VC dimension intuitively.</strong></summary>

VC dimension measures a model's **capacity** or **flexibility**:

- It's the maximum number of points that can be "shattered" (classified correctly for ANY labeling)
- Higher VC dim = more expressive model = can fit more complex patterns
- But higher VC dim also means more risk of overfitting with limited data

Example: A line in 2D can shatter 3 points (any labeling of 3 general-position points can be separated), but not 4. So VC dim = 3.
</details>

<details>
<summary><strong>Q3: How does generalization error relate to model capacity and training set size?</strong></summary>

From VC theory:

$$\text{Generalization Error} \leq \text{Training Error} + O\left(\sqrt{\frac{d_{VC}}{n}}\right)$$

This tells us:
- **More capacity ($d_{VC}$)** → larger potential gap → worse generalization
- **More data ($n$)** → smaller gap → better generalization
- Trade-off: need enough capacity to fit the pattern, but not so much that we overfit
</details>

<details>
<summary><strong>Q4: What is the "double descent" phenomenon?</strong></summary>

Double descent challenges the classical bias-variance tradeoff:

1. **Classical regime**: Test error decreases then increases with capacity
2. **Interpolation threshold**: Peak test error when model just fits training data
3. **Over-parameterized regime**: Test error decreases again!

Modern deep networks operate in the over-parameterized regime where more parameters can actually improve generalization, contradicting classical VC bounds.
</details>

<details>
<summary><strong>Q5: Why are VC bounds often not useful in practice?</strong></summary>

VC bounds are often very loose because:
- They're worst-case over all distributions
- They don't account for algorithm-specific properties (e.g., SGD implicit regularization)
- The bounds scale poorly with model size
- They don't explain why over-parameterized models generalize

In practice, we use:
- Validation-based model selection
- Cross-validation for hyperparameter tuning
- Empirical observations about what works
</details>

<details>
<summary><strong>Q6: What factors affect a model's generalization ability?</strong></summary>

Key factors:
1. **Model capacity**: Higher capacity = more risk of overfitting
2. **Training set size**: More data = better generalization
3. **Regularization**: Reduces effective capacity
4. **Data quality**: Noisy labels hurt generalization
5. **Training algorithm**: SGD has implicit regularization
6. **Architecture**: Inductive biases (e.g., CNNs for images)
7. **Early stopping**: Limits effective capacity
8. **Data augmentation**: Increases effective training set size
</details>

---

## 8. References

1. Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.
2. Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.
3. Belkin, M., et al. (2019). "Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-off." PNAS.
4. Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). *Foundations of Machine Learning*. MIT Press.
5. Zhang, C., et al. (2017). "Understanding Deep Learning Requires Rethinking Generalization." ICLR.
