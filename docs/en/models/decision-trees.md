# Decision Trees

## 1. Interview Summary

**Key Points to Remember:**
- **Recursive partitioning**: Split feature space into rectangular regions
- **Splitting criteria**: Gini impurity, entropy (information gain), MSE for regression
- **Greedy algorithm**: Choose locally optimal split at each node
- **Interpretable**: Easy to visualize and explain decisions
- **Prone to overfitting**: Deep trees memorize training data

**Common Interview Questions:**
- "What is information gain and how is it calculated?"
- "Compare Gini impurity vs entropy"
- "How do you prevent overfitting in decision trees?"

---

## 2. Core Definitions

### Tree Structure
- **Root node**: Top node, contains all data
- **Internal nodes**: Decision points with splitting rules
- **Leaf nodes**: Terminal nodes with predictions
- **Depth**: Maximum path length from root to any leaf

### Splitting Criteria (Classification)

**Gini Impurity:**

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

**Entropy:**

$$H = -\sum_{k=1}^{K} p_k \log_2 p_k$$

**Information Gain:**

$$IG = H(\text{parent}) - \sum_{\text{child}} \frac{n_{\text{child}}}{n_{\text{parent}}} H(\text{child})$$

### Splitting Criteria (Regression)

**Mean Squared Error:**

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

Split to minimize weighted MSE of children.

---

## 3. Math and Derivations

### Gini Impurity Derivation

Gini measures probability of misclassifying a randomly chosen element:

$$G = \sum_{k=1}^{K} p_k (1 - p_k) = 1 - \sum_{k=1}^{K} p_k^2$$

For binary classification with $p$ positive:

$$G = 2p(1-p)$$

- $G = 0$: Pure node (all same class)
- $G = 0.5$: Maximum impurity (50-50 split)

### Entropy vs Gini

| Property | Entropy | Gini |
|----------|---------|------|
| Range | [0, log₂K] | [0, 1-1/K] |
| Pure node | 0 | 0 |
| Maximum | log₂K (uniform) | 1-1/K |
| Computation | Slower (log) | Faster |
| Sensitivity | More sensitive to class distribution | Less sensitive |

In practice, they often produce similar trees.

### Optimal Split Search

For a feature $x_j$ with values $\{v_1, ..., v_m\}$:
1. Sort unique values
2. Try splits at midpoints: $(v_i + v_{i+1})/2$
3. Choose split maximizing information gain

**Complexity per split:** $O(n \log n)$ for sorting + $O(n)$ for evaluation

### Why Greedy is Suboptimal

Greedy algorithm finds locally optimal split, not globally optimal tree.

Example: XOR problem
- No single feature split helps
- But combination of splits solves it

Finding optimal tree is NP-complete.

---

## 4. Algorithm Sketch

### CART Algorithm (Classification and Regression Trees)

```
BuildTree(data, depth):
    # Base cases
    if depth >= max_depth or |data| < min_samples:
        return LeafNode(majority_class or mean_value)

    if all labels same:
        return LeafNode(label)

    # Find best split
    best_gain = 0
    for each feature j:
        for each threshold t:
            gain = compute_gain(data, j, t)
            if gain > best_gain:
                best_gain = gain
                best_split = (j, t)

    if best_gain == 0:
        return LeafNode(prediction)

    # Split data
    left_data = data where x[j] <= t
    right_data = data where x[j] > t

    # Recurse
    left_child = BuildTree(left_data, depth + 1)
    right_child = BuildTree(right_data, depth + 1)

    return InternalNode(best_split, left_child, right_child)
```

### Prediction

```
Predict(node, x):
    if node is LeafNode:
        return node.prediction

    if x[node.feature] <= node.threshold:
        return Predict(node.left_child, x)
    else:
        return Predict(node.right_child, x)
```

### Pruning (Post-pruning)

```
Prune(node, validation_data):
    if node is LeafNode:
        return node

    # Recursively prune children
    node.left = Prune(node.left, validation_data)
    node.right = Prune(node.right, validation_data)

    # Try replacing subtree with leaf
    original_error = evaluate(node, validation_data)
    pruned_node = LeafNode(majority_prediction(node))
    pruned_error = evaluate(pruned_node, validation_data)

    if pruned_error <= original_error + alpha:
        return pruned_node
    return node
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Overfitting | Trees grow too deep | Prune, set max_depth, min_samples |
| Axis-aligned splits | Can't capture diagonal boundaries | Use oblique trees or ensemble |
| Unstable | Small data changes → different tree | Use ensemble (Random Forest) |
| Biased toward many-valued features | More split points to try | Use information gain ratio |
| Imbalanced classes | Majority class dominates | Class weights, balanced sampling |

### Regularization Parameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| max_depth | Limit tree depth | 3-20 |
| min_samples_split | Min samples to split | 2-20 |
| min_samples_leaf | Min samples in leaf | 1-10 |
| max_features | Features to consider per split | sqrt(n), log2(n), n |
| min_impurity_decrease | Min gain required | 0.0-0.1 |

---

## 6. Mini Example

```python
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def _gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain = 0
        best_feature, best_threshold = None, None
        parent_gini = self._gini(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if sum(left_mask) < self.min_samples or sum(right_mask) < self.min_samples:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])

                n_left, n_right = sum(left_mask), sum(right_mask)
                weighted_gini = (n_left * left_gini + n_right * right_gini) / len(y)
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        # Base cases
        if depth >= self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1:
            return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}

        feature, threshold, gain = self._best_split(X, y)

        if feature is None:
            return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}

        left_mask = X[:, feature] <= threshold
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# Example
np.random.seed(42)
X = np.random.randn(200, 2)
y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)  # First quadrant vs rest

tree = DecisionTree(max_depth=3)
tree.fit(X, y)
predictions = tree.predict(X)
accuracy = np.mean(predictions == y)
print(f"Training accuracy: {accuracy:.3f}")

# Test new points
X_test = np.array([[1, 1], [-1, -1], [1, -1]])
print(f"Test predictions: {tree.predict(X_test)}")
```

**Output:**
```
Training accuracy: 0.985
Test predictions: [1 0 0]
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is the difference between Gini impurity and entropy?</strong></summary>

Both measure node impurity:

**Gini**: $G = 1 - \sum p_k^2$
- Probability of misclassifying random element
- Faster to compute (no logarithm)
- Range: [0, 1-1/K]

**Entropy**: $H = -\sum p_k \log_2 p_k$
- Information theoretic measure
- More sensitive to class distribution changes
- Range: [0, log₂K]

In practice, they produce similar trees. Gini is often preferred for speed.
</details>

<details>
<summary><strong>Q2: Why are decision trees prone to overfitting?</strong></summary>

Decision trees can:
1. **Grow arbitrarily deep**: Memorize training data exactly
2. **Create complex boundaries**: Each leaf can be a single point
3. **Fit noise**: Every noisy point can get its own path

A tree with max_depth=∞ achieves 100% training accuracy but poor generalization.

**Solutions**: Limit max_depth, require min_samples_split, prune after building, use ensembles.
</details>

<details>
<summary><strong>Q3: What is information gain and how is it used?</strong></summary>

Information gain measures impurity reduction from a split:

$$IG = H(\text{parent}) - \sum_{\text{children}} \frac{n_{\text{child}}}{n_{\text{parent}}} H(\text{child})$$

Higher IG = better split (more impurity reduction)

**Usage**: At each node, try all features and thresholds, choose the one with maximum information gain.

**Note**: IG is biased toward features with many values. Information gain ratio (C4.5) addresses this.
</details>

<details>
<summary><strong>Q4: How does a decision tree handle continuous vs categorical features?</strong></summary>

**Continuous features**:
- Try all possible threshold splits: $x_j \leq t$ vs $x_j > t$
- Usually test midpoints between sorted unique values
- Binary split only

**Categorical features**:
- Can do binary split (subset vs complement)
- Or multi-way split (one branch per category)
- CART uses binary only; ID3/C4.5 allow multi-way

Multi-way splits on high-cardinality categorical features can lead to overfitting.
</details>

<details>
<summary><strong>Q5: What is pruning and why is it important?</strong></summary>

Pruning removes subtrees that don't improve generalization:

**Pre-pruning** (early stopping):
- Stop growing when gain below threshold
- Stop when node has too few samples
- Limit max_depth

**Post-pruning** (reduced error pruning):
- Grow full tree first
- Remove subtrees that don't improve validation error
- More accurate but slower

Pruning prevents overfitting while keeping useful structure.
</details>

<details>
<summary><strong>Q6: Why can't decision trees easily capture diagonal decision boundaries?</strong></summary>

Standard trees use **axis-aligned splits**: $x_j \leq t$

To approximate diagonal boundary $x_1 + x_2 = 0$:
- Requires many horizontal and vertical splits
- Creates staircase pattern
- Needs deep tree

**Solutions**:
- **Oblique trees**: Allow splits like $w_1 x_1 + w_2 x_2 \leq t$
- **Random Forests**: Ensemble smooths boundaries
- **Feature engineering**: Create $x_{new} = x_1 + x_2$
</details>

---

## 8. References

1. Breiman, L., et al. (1984). *Classification and Regression Trees*. Wadsworth.
2. Quinlan, J. R. (1986). "Induction of Decision Trees." Machine Learning.
3. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 9.
5. scikit-learn documentation: Decision Trees.
