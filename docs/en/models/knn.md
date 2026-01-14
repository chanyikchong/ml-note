# k-Nearest Neighbors

## 1. Interview Summary

**Key Points to Remember:**
- **Instance-based learning**: No explicit training, stores all data
- **Distance metric**: Euclidean, Manhattan, or custom distance
- **k parameter**: Number of neighbors to consider
- **Lazy learner**: All computation at prediction time
- **Curse of dimensionality**: Performance degrades in high dimensions

**Common Interview Questions:**
- "What is the time complexity of kNN prediction?"
- "How do you choose k?"
- "Why does kNN struggle with high-dimensional data?"

---

## 2. Core Definitions

### Algorithm
For a new point $x$:
1. Compute distance to all training points
2. Find $k$ nearest neighbors
3. For classification: majority vote
4. For regression: average of neighbor values

### Distance Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | $\sqrt{\sum_i (x_i - y_i)^2}$ | Continuous, scaled features |
| Manhattan | $\sum_i |x_i - y_i|$ | Grid-like data, robust to outliers |
| Minkowski | $(\sum_i |x_i - y_i|^p)^{1/p}$ | Generalization (p=1: Manhattan, p=2: Euclidean) |
| Cosine | $1 - \frac{x \cdot y}{\|x\|\|y\|}$ | Text, high-dimensional sparse |

### Weighted kNN
Weight neighbors by inverse distance:
$$\hat{y} = \frac{\sum_{i \in N_k} w_i y_i}{\sum_{i \in N_k} w_i}, \quad w_i = \frac{1}{d(x, x_i)}$$

---

## 3. Math and Derivations

### Bias-Variance Tradeoff with k

**Small k (e.g., k=1):**
- Low bias: Captures local patterns
- High variance: Sensitive to noise
- Decision boundary is complex

**Large k:**
- High bias: Over-smooths, misses local patterns
- Low variance: More stable predictions
- Decision boundary is smoother

### Optimal k Selection
- Use cross-validation
- Rule of thumb: $k \approx \sqrt{n}$
- Odd k for binary classification (avoid ties)

### Curse of Dimensionality

In high dimensions:
- All points become equidistant
- Nearest neighbor is not much closer than farthest

For uniform distribution in $[0,1]^d$:
$$\frac{\text{dist}_{max} - \text{dist}_{min}}{\text{dist}_{min}} \to 0 \text{ as } d \to \infty$$

Volume of hypersphere relative to hypercube vanishes exponentially.

---

## 4. Algorithm Sketch

### Basic kNN

```
Input: Training data (X, y), query point x_q, k
Output: Predicted label/value

1. Compute distances:
   For each training point x_i:
       d_i = distance(x_q, x_i)

2. Find k nearest:
   indices = argsort(distances)[:k]

3. Aggregate:
   Classification: return mode(y[indices])
   Regression: return mean(y[indices])
```

### Efficient kNN with KD-Tree

```
Build Phase:
1. Choose axis with highest variance
2. Split at median
3. Recursively build left/right subtrees

Query Phase:
1. Traverse tree to leaf containing query
2. Backtrack, pruning branches farther than k-th nearest
3. Average case: O(log n) per query
```

### Ball Tree (for high dimensions)

```
Build Phase:
1. Find centroid and radius containing all points
2. Split into two balls
3. Recursively partition

Query Phase:
- Better than KD-tree when d > 20
- O(d * log n) average case
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Not scaling features | Features with large range dominate | Standardize all features |
| Wrong k value | Default k=5 may not be optimal | Cross-validate k |
| Slow prediction | O(nd) for each query | Use KD-tree/Ball tree |
| High dimensionality | Distance becomes meaningless | Reduce dimensions first |
| Imbalanced classes | Majority class dominates | Use weighted voting |

### When to Use kNN

| Scenario | Recommendation |
|----------|---------------|
| Small dataset, low dimensions | Good fit |
| Need interpretable predictions | Good (show neighbors) |
| Large dataset | Consider approximate NN |
| High dimensions (>50) | Use with dimensionality reduction |
| Need fast prediction | Avoid (or use tree structures) |

---

## 6. Mini Example

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        # Compute distances
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # Get k nearest indices
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        # Majority vote
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# Example
np.random.seed(42)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

X_test = np.array([[0.5, 0.5], [-0.5, -0.5], [1.0, -1.0]])

knn = KNN(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(f"Test points: {X_test}")
print(f"Predictions: {predictions}")

# Effect of k
for k in [1, 3, 5, 10, 20]:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    train_acc = np.mean(knn.predict(X_train) == y_train)
    print(f"k={k:2d}: Training accuracy = {train_acc:.3f}")
```

**Output:**
```
Test points: [[ 0.5  0.5] [-0.5 -0.5] [ 1.  -1. ]]
Predictions: [1 0 0]
k= 1: Training accuracy = 1.000
k= 3: Training accuracy = 0.970
k= 5: Training accuracy = 0.960
k=10: Training accuracy = 0.950
k=20: Training accuracy = 0.940
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is the time complexity of kNN prediction?</strong></summary>

Without optimization:
- **Training**: O(1) - just stores data
- **Prediction**: O(nd) per query - compute distance to all n points in d dimensions

With KD-tree/Ball tree:
- **Build**: O(n log n)
- **Prediction**: O(log n) average case (degrades to O(n) in high dimensions)
</details>

<details>
<summary><strong>Q2: How does the choice of k affect bias and variance?</strong></summary>

- **k=1**: Zero training error, high variance, low bias. Very sensitive to noise.
- **Large k**: Smoother boundaries, high bias, low variance. May underfit.
- **k=n**: Predicts majority class for everything (maximum bias).

Optimal k balances bias-variance tradeoff. Use cross-validation to find it.
</details>

<details>
<summary><strong>Q3: Why is feature scaling important for kNN?</strong></summary>

kNN uses distance metrics. Features with larger scales dominate the distance calculation.

Example: If feature A is in [0, 1000] and feature B is in [0, 1]:
- Distance is dominated by feature A
- Feature B effectively ignored

Solution: Standardize (z-score) or normalize (min-max) all features.
</details>

<details>
<summary><strong>Q4: What is the curse of dimensionality and how does it affect kNN?</strong></summary>

In high dimensions:
1. **Distance concentration**: All pairwise distances become similar
2. **Sparse data**: Data points spread out, nearest neighbors are far
3. **Volume**: Most volume is near the surface of hypercube

Effects on kNN:
- "Nearest" neighbor is not meaningfully close
- Need exponentially more data as d increases
- Performance degrades significantly

Solutions: Use dimensionality reduction (PCA, t-SNE) before kNN.
</details>

<details>
<summary><strong>Q5: How do you handle ties in kNN classification?</strong></summary>

Methods to break ties:
1. **Use odd k** for binary classification
2. **Distance weighting**: Closer neighbors get more weight
3. **Random selection**: Among tied classes
4. **Reduce k**: Until tie is broken

Best practice: Use weighted kNN with inverse distance weights.
</details>

<details>
<summary><strong>Q6: When would you use Manhattan distance over Euclidean?</strong></summary>

**Manhattan (L1)**:
- Grid-like movement (taxi distance)
- High-dimensional sparse data
- More robust to outliers
- When features are not correlated

**Euclidean (L2)**:
- Continuous, dense features
- When actual geometric distance matters
- Features are scaled similarly

Manhattan is often better for high-dimensional data as it doesn't amplify outliers through squaring.
</details>

---

## 8. References

1. Cover, T., & Hart, P. (1967). "Nearest Neighbor Pattern Classification." IEEE Transactions on Information Theory.
2. Friedman, J., Bentley, J., & Finkel, R. (1977). "An Algorithm for Finding Best Matches in Logarithmic Expected Time." ACM TOMS.
3. Beyer, K., et al. (1999). "When Is 'Nearest Neighbor' Meaningful?" ICDT.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 13.
5. scikit-learn documentation: Nearest Neighbors.
