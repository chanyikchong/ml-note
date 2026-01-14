# Support Vector Machines

## 1. Interview Summary

**Key Points to Remember:**
- **Maximum margin classifier**: Find hyperplane with largest margin
- **Support vectors**: Points closest to decision boundary
- **Soft margin (C parameter)**: Trade off between margin and misclassification
- **Kernel trick**: Compute dot products in high-dimensional space without explicit mapping
- **Common kernels**: Linear, polynomial, RBF (Gaussian)

**Common Interview Questions:**
- "What is the intuition behind SVM?"
- "Explain the kernel trick"
- "What do support vectors represent?"

---

## 2. Core Definitions

### Hard Margin SVM
For linearly separable data, find hyperplane $w^Tx + b = 0$ that:
- Correctly classifies all points: $y_i(w^Tx_i + b) \geq 1$
- Maximizes margin: $\frac{2}{\|w\|}$

### Soft Margin SVM
Allow some misclassification with slack variables $\xi_i$:

$$y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### Support Vectors
Training points that lie on or within the margin boundaries:
- Points with $y_i(w^Tx_i + b) = 1$ (on margin)
- Points with $0 < \alpha_i < C$ in dual formulation

### Kernel Function
$K(x, z) = \langle\phi(x), \phi(z)\rangle$ computes inner product in feature space.

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | $x^Tz$ | Linearly separable |
| Polynomial | $(x^Tz + c)^d$ | Polynomial boundaries |
| RBF | $\exp(-\gamma\|x-z\|^2)$ | Complex boundaries |

---

## 3. Math and Derivations

### Primal Formulation (Hard Margin)

$$\min_{w,b} \frac{1}{2}\|w\|^2$$

$$\text{s.t. } y_i(w^Tx_i + b) \geq 1, \quad \forall i$$

### Primal Formulation (Soft Margin)

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$$

$$\text{s.t. } y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### Dual Formulation

Using Lagrangian with multipliers $\alpha_i$:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

$$\text{s.t. } 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

**Key insight**: Only dot products $x_i^Tx_j$ appear → can use kernels!

### Kernel Trick

Replace $x_i^Tx_j$ with $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

**Decision function**:

$$f(x) = \text{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)$$

### RBF Kernel Properties

$$K(x, z) = \exp(-\gamma\|x-z\|^2)$$

- Corresponds to infinite-dimensional feature space
- $\gamma$ controls decision boundary smoothness
- High $\gamma$: Complex boundary (risk of overfitting)
- Low $\gamma$: Smooth boundary (risk of underfitting)

---

## 4. Algorithm Sketch

### Training SVM (SMO Algorithm Intuition)

```
Sequential Minimal Optimization:
1. Initialize all α = 0
2. Repeat until convergence:
   a. Select two alphas (α_i, α_j) that violate KKT conditions
   b. Optimize these two analytically (2D problem)
   c. Update b
3. Support vectors: points with α > 0
```

### Prediction

```
Input: new point x
Output: class label

# Compute decision value
decision = b
For each support vector (x_i, y_i, α_i):
    decision += α_i * y_i * K(x_i, x)

Return sign(decision)
```

### Choosing Parameters

```
For kernel choice:
    Start with RBF (most flexible)
    Try linear if high-dimensional/sparse

For C (regularization):
    High C: Fit training data closely (risk overfit)
    Low C: Larger margin, more misclassification allowed

For γ (RBF):
    High γ: Each point has local influence
    Low γ: Points have broader influence

Use cross-validation to tune C and γ together
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Not scaling features | SVM sensitive to scale | Always standardize |
| Wrong kernel choice | Default may not fit data | Try multiple kernels |
| Ignoring C parameter | Default may overfit/underfit | Grid search C |
| RBF γ too high | Overfitting, training only | Cross-validate γ |
| Too many features for RBF | Curse of dimensionality | Consider linear kernel |

### When to Use SVM

| Scenario | Recommendation |
|----------|---------------|
| High-dimensional, sparse | Linear SVM |
| Small-medium dataset, complex boundary | RBF SVM |
| Large dataset (>100k samples) | Consider other methods (slower training) |
| Need probabilities | SVM + Platt scaling, or use other method |

---

## 6. Mini Example

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification, make_circles

# Generate non-linear data
X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Linear SVM (will fail on circles)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_s, y_train)
print(f"Linear SVM accuracy: {svm_linear.score(X_test_s, y_test):.3f}")

# RBF SVM (should work)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='auto')
svm_rbf.fit(X_train_s, y_train)
print(f"RBF SVM accuracy: {svm_rbf.score(X_test_s, y_test):.3f}")

# Grid search for optimal parameters
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train_s, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
print(f"Test accuracy: {grid.score(X_test_s, y_test):.3f}")

# Support vectors
print(f"Number of support vectors: {len(svm_rbf.support_)}")
print(f"Support vectors per class: {svm_rbf.n_support_}")
```

**Output:**
```
Linear SVM accuracy: 0.500
RBF SVM accuracy: 0.990
Best params: {'C': 10, 'gamma': 1}
Best CV score: 0.985
Test accuracy: 0.990
Number of support vectors: 62
Support vectors per class: [31 31]
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is the intuition behind the maximum margin principle?</strong></summary>

SVM finds the hyperplane that maximizes the distance to the nearest training points (margin). Larger margin means:
- More confident classification
- Better generalization to unseen data
- More robust to noise

The margin is $\frac{2}{\|w\|}$, so maximizing margin = minimizing $\|w\|^2$.
</details>

<details>
<summary><strong>Q2: What are support vectors and why are they important?</strong></summary>

Support vectors are training points that:
- Lie exactly on the margin boundaries
- Have non-zero Lagrange multipliers ($\alpha_i > 0$)
- Completely determine the decision boundary

Importance:
- Removing non-support vectors doesn't change the model
- The decision function only depends on support vectors
- Sparse representation (often few support vectors)
</details>

<details>
<summary><strong>Q3: Explain the kernel trick.</strong></summary>

The kernel trick allows computing dot products in high-dimensional feature space without explicitly computing the transformation.

Instead of: $\phi(x)^T\phi(z)$ (expensive in high dimensions)
Use: $K(x, z)$ (computed in original space)

This works because the SVM optimization and prediction only involve dot products, which can be replaced with kernel evaluations.

Example: RBF kernel corresponds to infinite-dimensional space but computes in O(d) time.
</details>

<details>
<summary><strong>Q4: What does the C parameter control?</strong></summary>

C controls the trade-off between:
- **Large margin** (small C): Prioritize wider margin, allow more misclassifications
- **Correct classification** (large C): Prioritize fitting training data, smaller margin

Equivalently, C is the penalty for misclassification:
- C → 0: Ignores training errors, maximum margin
- C → ∞: Hard margin (no errors allowed)
</details>

<details>
<summary><strong>Q5: When would you use a linear kernel vs RBF kernel?</strong></summary>

**Linear kernel**:
- High-dimensional data (text, genomics)
- Sparse features
- Many samples (faster training)
- When data is linearly separable

**RBF kernel**:
- Low-to-medium dimensional data
- Complex, non-linear decision boundaries
- When linear doesn't work well
- Smaller datasets (RBF training is slower)
</details>

<details>
<summary><strong>Q6: How does the γ parameter in RBF affect the model?</strong></summary>

γ controls the "reach" of each training example:
- **High γ**: Each point influences only nearby points → complex, wiggly boundary → overfitting
- **Low γ**: Each point influences far points → smooth boundary → underfitting

Rule of thumb: Start with γ = 1/(n_features) and tune via cross-validation.
</details>

---

## 8. References

1. Cortes, C., & Vapnik, V. (1995). "Support-Vector Networks." Machine Learning.
2. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
3. Platt, J. (1998). "Sequential Minimal Optimization." Microsoft Research.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 12.
5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 7.
