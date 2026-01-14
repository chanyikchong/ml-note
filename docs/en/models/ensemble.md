# Ensemble Methods

## 1. Interview Summary

**Key Points to Remember:**
- **Bagging**: Train on bootstrap samples, reduce variance (Random Forest)
- **Boosting**: Train sequentially on residuals, reduce bias (XGBoost, AdaBoost)
- **Random Forest**: Bagged trees + random feature subset per split
- **Gradient Boosting**: Fit trees to negative gradient of loss
- **Key hyperparameters**: n_estimators, learning_rate, max_depth, subsample

**Common Interview Questions:**
- "What is the difference between bagging and boosting?"
- "Why does Random Forest work well?"
- "Explain gradient boosting step by step"

---

## 2. Core Definitions

### Bagging (Bootstrap Aggregating)
1. Create B bootstrap samples (sample with replacement)
2. Train model on each sample
3. Average predictions (regression) or vote (classification)

**Variance reduction:**

$$\text{Var}(\bar{f}) = \frac{1}{B}\text{Var}(f) + \frac{B-1}{B}\rho \cdot \text{Var}(f)$$

Where $\rho$ is correlation between models.

### Random Forest
Bagged decision trees with additional randomness:
- Each tree trained on bootstrap sample
- At each split, consider random subset of features
- Reduces correlation $\rho$ → better variance reduction

### Boosting
Sequentially combine weak learners:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $h_m$ is fitted to correct errors of $F_{m-1}$.

### AdaBoost
Reweight misclassified samples:

$$w_i^{(m)} = w_i^{(m-1)} \cdot \exp(\alpha_m \cdot \mathbb{1}[y_i \neq h_m(x_i)])$$

### Gradient Boosting
Fit new tree to pseudo-residuals:

$$r_i^{(m)} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

For MSE loss: $r_i = y_i - F_{m-1}(x_i)$ (actual residuals)

---

## 3. Math and Derivations

### Random Forest Variance Reduction

For B trees with correlation $\rho$:

$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

- As $B \to \infty$: $\text{Var} \to \rho \sigma^2$
- Lower $\rho$ → better (achieved by feature subsampling)
- More trees always help (never overfits from adding trees)

### Gradient Boosting Derivation

**Goal**: Minimize $\sum_i L(y_i, F(x_i))$

**Functional gradient descent**:

$$F_m = F_{m-1} - \eta \nabla_F L$$

**Pseudo-residuals** (negative gradient direction):

$$r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

| Loss | $L(y, F)$ | Pseudo-residual |
|------|-----------|-----------------|
| MSE | $\frac{1}{2}(y - F)^2$ | $y - F$ |
| MAE | $|y - F|$ | $\text{sign}(y - F)$ |
| Log-loss | $-y\log p - (1-y)\log(1-p)$ | $y - p$ |

### XGBoost Objective

$$\text{Obj} = \sum_i L(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

**Regularization:**

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $T$: number of leaves
- $w_j$: leaf weights
- $\gamma$: complexity penalty per leaf
- $\lambda$: L2 regularization on weights

### Feature Importance

**MDI (Mean Decrease in Impurity)**:

$$\text{Importance}(j) = \sum_{\text{nodes using } j} p(\text{node}) \cdot \Delta \text{impurity}$$

**Permutation Importance**:

$$\text{Importance}(j) = \text{score} - \text{score}_{\text{permuted } j}$$

---

## 4. Algorithm Sketch

### Random Forest

```
Train:
    For b = 1 to B:
        sample = bootstrap(training_data)
        tree[b] = train_tree(sample, max_features=sqrt(p))

Predict (classification):
    votes = [tree[b].predict(x) for b in 1..B]
    return majority_vote(votes)

Predict (regression):
    predictions = [tree[b].predict(x) for b in 1..B]
    return mean(predictions)
```

### Gradient Boosting

```
Initialize: F_0(x) = argmin_c sum(L(y_i, c))  # e.g., mean for MSE

For m = 1 to M:
    # Compute pseudo-residuals
    r_i = -dL/dF evaluated at F_{m-1}(x_i)

    # Fit tree to residuals
    h_m = fit_tree(X, r)

    # Line search for optimal step size (optional)
    gamma_m = argmin_gamma sum(L(y_i, F_{m-1}(x_i) + gamma * h_m(x_i)))

    # Update model
    F_m = F_{m-1} + learning_rate * gamma_m * h_m

Return F_M
```

### XGBoost Split Finding

```
For each node:
    For each feature j:
        Sort samples by feature j
        For each split point s:
            # Compute gradient statistics
            G_L = sum of gradients in left child
            H_L = sum of hessians in left child
            G_R, H_R = same for right

            # Score improvement
            gain = 0.5 * (G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - (G_L+G_R)^2/(H_L+H_R+λ)) - γ

        Select split with maximum gain
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Boosting overfits | Too many iterations, low learning rate | Early stopping, regularization |
| RF underfits | Trees too shallow | Increase max_depth |
| Slow training | Too many trees/deep trees | Reduce n_estimators, subsample |
| Memory issues | Storing all trees | Use lighter trees, pruning |
| Feature importance bias | Correlated features split importance | Use permutation importance |

### Hyperparameter Guidelines

| Method | Key Parameters | Tuning Strategy |
|--------|---------------|-----------------|
| Random Forest | n_estimators, max_depth, max_features | More trees rarely hurts; tune depth |
| Gradient Boosting | n_estimators, learning_rate, max_depth | Lower LR + more trees; early stopping |
| XGBoost | Same + reg_alpha, reg_lambda, subsample | Grid search with CV |

### When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Tabular data, quick baseline | Random Forest |
| Need best accuracy, have time | XGBoost/LightGBM with tuning |
| Interpretability matters | Random Forest (simpler) |
| Streaming/online learning | Consider other methods |
| Very large dataset | LightGBM (histogram-based) |

---

## 6. Mini Example

```python
import numpy as np
from collections import Counter

class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n = len(y)
        indices = np.random.choice(n, size=n, replace=True)
        return X[indices], y[indices]

    def _gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y, feature_indices):
        best_gain, best_feature, best_threshold = 0, None, None
        parent_gini = self._gini(y)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                if sum(left_mask) < 2 or sum(~left_mask) < 2:
                    continue

                weighted_gini = (sum(left_mask) * self._gini(y[left_mask]) +
                                 sum(~left_mask) * self._gini(y[~left_mask])) / len(y)
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth, n_features):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 4:
            return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}

        feature_indices = np.random.choice(X.shape[1], size=n_features, replace=False)
        feature, threshold = self._best_split(X, y, feature_indices)

        if feature is None:
            return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}

        left_mask = X[:, feature] <= threshold
        return {
            'leaf': False, 'feature': feature, 'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1, n_features),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1, n_features)
        }

    def fit(self, X, y):
        n_features = int(np.sqrt(X.shape[1])) if self.max_features == 'sqrt' else X.shape[1]
        self.trees = []
        for _ in range(self.n_trees):
            X_boot, y_boot = self._bootstrap_sample(X, y)
            tree = self._build_tree(X_boot, y_boot, 0, n_features)
            self.trees.append(tree)

    def _predict_tree(self, x, node):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        return self._predict_tree(x, node['right'])

    def predict(self, X):
        predictions = np.array([[self._predict_tree(x, tree) for tree in self.trees] for x in X])
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])


class SimpleGradientBoosting:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        residuals = y - self.initial_pred

        for _ in range(self.n_estimators):
            tree = self._fit_tree(X, residuals, 0)
            self.trees.append(tree)
            predictions = np.array([self._predict_tree(x, tree) for x in X])
            residuals = residuals - self.learning_rate * predictions

    def _fit_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 4:
            return {'leaf': True, 'prediction': np.mean(y)}

        best_mse, best_feature, best_threshold = float('inf'), None, None
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= threshold
                if sum(left_mask) < 2 or sum(~left_mask) < 2:
                    continue
                mse = (np.var(y[left_mask]) * sum(left_mask) +
                       np.var(y[~left_mask]) * sum(~left_mask))
                if mse < best_mse:
                    best_mse, best_feature, best_threshold = mse, feature, threshold

        if best_feature is None:
            return {'leaf': True, 'prediction': np.mean(y)}

        left_mask = X[:, best_feature] <= best_threshold
        return {
            'leaf': False, 'feature': best_feature, 'threshold': best_threshold,
            'left': self._fit_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._fit_tree(X[~left_mask], y[~left_mask], depth + 1)
        }

    def _predict_tree(self, x, node):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        return self._predict_tree(x, node['right'])

    def predict(self, X):
        pred = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            pred += self.learning_rate * np.array([self._predict_tree(x, tree) for x in X])
        return pred


# Example
np.random.seed(42)
X = np.random.randn(300, 4)
y_class = (X[:, 0] + X[:, 1] > 0).astype(int)
y_reg = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(300) * 0.1

# Random Forest
rf = SimpleRandomForest(n_trees=20, max_depth=5)
rf.fit(X, y_class)
rf_acc = np.mean(rf.predict(X) == y_class)
print(f"Random Forest accuracy: {rf_acc:.3f}")

# Gradient Boosting
gb = SimpleGradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
gb.fit(X, y_reg)
gb_mse = np.mean((gb.predict(X) - y_reg) ** 2)
print(f"Gradient Boosting MSE: {gb_mse:.4f}")
```

**Output:**
```
Random Forest accuracy: 0.987
Gradient Boosting MSE: 0.0098
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is the difference between bagging and boosting?</strong></summary>

**Bagging** (Bootstrap Aggregating):
- Trains models **independently** on bootstrap samples
- Combines by averaging/voting
- **Reduces variance**, keeps bias
- Parallelizable
- Example: Random Forest

**Boosting**:
- Trains models **sequentially**
- Each model corrects errors of previous
- **Reduces bias**, can increase variance
- Not parallelizable (sequential dependency)
- Examples: AdaBoost, Gradient Boosting, XGBoost
</details>

<details>
<summary><strong>Q2: Why does Random Forest reduce variance compared to a single tree?</strong></summary>

Two mechanisms:

1. **Bagging**: Average of B predictions has variance $\text{Var}/B$ if independent
2. **Feature subsampling**: Decorrelates trees (reduces $\rho$)

Combined variance: $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$

By making trees less correlated (lower $\rho$), variance approaches $\frac{\sigma^2}{B}$ as B increases.

Key insight: Can never overfit by adding more trees (variance only decreases).
</details>

<details>
<summary><strong>Q3: Explain gradient boosting step by step.</strong></summary>

1. **Initialize**: $F_0(x) = \bar{y}$ (mean prediction)

2. **For each iteration m**:
   - Compute pseudo-residuals: $r_i = -\frac{\partial L}{\partial F}|_{F=F_{m-1}}$
   - Fit tree $h_m$ to residuals
   - Update: $F_m = F_{m-1} + \eta \cdot h_m$

3. **Final model**: $F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)$

For MSE loss, pseudo-residuals are simply $y - F_{m-1}(x)$.

Key: We're doing gradient descent in function space, where each tree represents a step direction.
</details>

<details>
<summary><strong>Q4: What is the role of learning rate in gradient boosting?</strong></summary>

Learning rate $\eta$ (shrinkage) controls step size:

$$F_m = F_{m-1} + \eta \cdot h_m$$

**Effects**:
- Lower $\eta$ → slower learning → need more trees
- Lower $\eta$ → better generalization (regularization effect)
- Higher $\eta$ → faster convergence → risk of overfitting

**Best practice**: Use low learning rate (0.01-0.1) with early stopping based on validation performance.
</details>

<details>
<summary><strong>Q5: How does XGBoost differ from standard gradient boosting?</strong></summary>

XGBoost improvements:
1. **Regularization**: L1/L2 on leaf weights, tree complexity penalty
2. **Second-order approximation**: Uses Hessian for better splits
3. **Sparse-aware**: Handles missing values natively
4. **Parallel processing**: Parallel feature scanning
5. **Tree pruning**: Prunes splits with negative gain
6. **Hardware optimization**: Cache-aware, out-of-core computing

These make XGBoost faster and more regularized than sklearn's GradientBoostingClassifier.
</details>

<details>
<summary><strong>Q6: When would you prefer Random Forest over Gradient Boosting?</strong></summary>

**Prefer Random Forest when**:
- Need quick, reliable baseline
- Want robust model without much tuning
- Parallelization is important
- Interpretability needed (feature importance is cleaner)
- Training data has noise (RF is more robust)

**Prefer Gradient Boosting when**:
- Need best possible accuracy
- Have time for hyperparameter tuning
- Dataset is clean and well-prepared
- Can use early stopping for regularization

In practice, tuned XGBoost/LightGBM often beats Random Forest, but requires more effort.
</details>

---

## 8. References

1. Breiman, L. (2001). "Random Forests." Machine Learning.
2. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics.
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD.
4. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NIPS.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapters 10, 15.
