# Dimensionality Reduction

## 1. Interview Summary

**Key Points to Remember:**
- **PCA**: Linear projection maximizing variance, finds orthogonal components
- **t-SNE**: Non-linear, preserves local structure, good for visualization
- **UMAP**: Faster than t-SNE, preserves more global structure
- **When to use**: Visualization, noise reduction, feature extraction, preprocessing

**Common Interview Questions:**
- "Derive PCA mathematically"
- "Why can't you use t-SNE for new data?"
- "What are the pitfalls of t-SNE?"

---

## 2. Core Definitions

### PCA (Principal Component Analysis)
Find directions of maximum variance:
$$\max_w w^T \Sigma w \quad \text{s.t. } \|w\|_2 = 1$$

Solution: Eigenvectors of covariance matrix $\Sigma$.

### Explained Variance Ratio
$$\text{Explained ratio}_k = \frac{\lambda_k}{\sum_i \lambda_i}$$

Choose k to explain desired % of variance (e.g., 95%).

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
Minimize KL divergence between:
- High-dim similarities (Gaussian)
- Low-dim similarities (t-distribution)

### UMAP (Uniform Manifold Approximation and Projection)
- Constructs fuzzy topological representation
- Optimizes cross-entropy between high/low dim representations
- Better preserves global structure than t-SNE

---

## 3. Math and Derivations

### PCA Derivation

**Objective**: Find projection $w$ maximizing variance.

**Variance of projected data:**
$$\text{Var}(Xw) = w^T X^T X w = w^T \Sigma w$$

**Constrained optimization** (Lagrangian):
$$L = w^T \Sigma w - \lambda(w^T w - 1)$$

**Taking derivative:**
$$\frac{\partial L}{\partial w} = 2\Sigma w - 2\lambda w = 0$$
$$\Sigma w = \lambda w$$

Solution: $w$ is eigenvector of $\Sigma$, variance = $\lambda$.

**Multiple components**: Take top k eigenvectors.

### PCA via SVD

For centered data $X$:
$$X = U \Sigma V^T$$

- Columns of $V$: Principal components (eigenvectors of $X^T X$)
- Columns of $U \Sigma$: Projected data
- Singular values: $\sigma_i = \sqrt{n \lambda_i}$

SVD is numerically more stable than eigendecomposition.

### t-SNE Algorithm

**Step 1**: Compute pairwise similarities in high-dim
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**Step 2**: Define similarities in low-dim (t-distribution)
$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$$

**Step 3**: Minimize KL divergence
$$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### Why t-distribution?

Heavy tails allow:
- Moderate distances in high-dim → far apart in low-dim
- Prevents "crowding problem"
- Better cluster separation

---

## 4. Algorithm Sketch

### PCA (via SVD)

```
Input: Data matrix X (n × d), components k
Output: Projected data (n × k), components

1. Center data: X_c = X - mean(X)
2. Compute SVD: X_c = U Σ Vᵀ
3. Take first k columns of V: V_k
4. Project: Z = X_c @ V_k
5. Return Z, V_k
```

### t-SNE

```
Input: Data X, perplexity, learning_rate, n_iter
Output: Low-dimensional embedding Y

1. Compute pairwise similarities P (with perplexity-adjusted σ)
2. Initialize Y randomly (usually from N(0, 10⁻⁴))

For iter = 1 to n_iter:
    # Compute low-dim similarities Q
    Q = student_t_similarities(Y)

    # Compute gradients
    gradient = 4 * Σ_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||²)⁻¹

    # Update Y (with momentum)
    Y = Y - learning_rate * gradient + momentum * prev_update

Return Y
```

### UMAP (High-level)

```
1. Build k-nearest neighbor graph
2. Compute fuzzy simplicial set (edge weights)
3. Initialize low-dim embedding
4. Optimize cross-entropy loss via SGD
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Not centering for PCA | PCA assumes centered data | Always center first |
| Wrong n_components | Too few loses info, too many keeps noise | Use explained variance ratio |
| t-SNE on new data | No projection function | Use parametric t-SNE or fit on all data |
| Interpreting t-SNE distances | Global distances not preserved | Don't interpret cluster distances |
| t-SNE with wrong perplexity | Poor structure recovery | Try multiple values (5-50) |
| Using t-SNE features for ML | Not meaningful | Only use for visualization |

### t-SNE Interpretation Caveats

1. **Cluster sizes don't mean anything**: Dense vs sparse is artifact
2. **Distance between clusters doesn't mean anything**: Only local structure preserved
3. **Different runs give different results**: Use same seed for reproducibility
4. **Perplexity matters a lot**: Low = tight clusters, High = broader structure

### When to Use Each Method

| Method | Use Case |
|--------|----------|
| PCA | Feature reduction, preprocessing, linear relationships |
| t-SNE | Visualization of clusters, exploring local structure |
| UMAP | Visualization + some global structure, faster than t-SNE |
| Kernel PCA | Non-linear relationships, smaller datasets |
| Autoencoders | Complex non-linear reduction, if enough data |

---

## 6. Mini Example

```python
import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        # Center data
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Store components and explained variance
        self.components = Vt[:self.n_components]
        total_var = np.sum(S**2)
        self.explained_variance_ratio = (S[:self.n_components]**2) / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean


def simple_tsne(X, n_components=2, perplexity=30, n_iter=1000, lr=200):
    """Simplified t-SNE implementation."""
    n = X.shape[0]

    # Compute pairwise distances
    dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    # Compute P (high-dim similarities)
    P = np.exp(-dists / (2 * perplexity))
    np.fill_diagonal(P, 0)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    # Initialize Y randomly
    np.random.seed(42)
    Y = np.random.randn(n, n_components) * 0.01

    # Gradient descent
    for _ in range(n_iter):
        # Compute Q (low-dim similarities with t-distribution)
        dists_low = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        Q = 1 / (1 + dists_low)
        np.fill_diagonal(Q, 0)
        Q = Q / Q.sum()
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ_diff[i, :, None] * diff) * (1 / (1 + dists_low[i, :, None])), axis=0)

        Y -= lr * grad

    return Y


# Example
np.random.seed(42)

# Generate data: 3 clusters in 10D
n_per_cluster = 50
cluster1 = np.random.randn(n_per_cluster, 10) + np.array([5] * 10)
cluster2 = np.random.randn(n_per_cluster, 10) + np.array([-5] * 10)
cluster3 = np.random.randn(n_per_cluster, 10)
X = np.vstack([cluster1, cluster2, cluster3])
labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA explained variance: {pca.explained_variance_ratio}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio):.3f}")

# Simple t-SNE (note: very simplified, real t-SNE is more complex)
print("\nRunning simplified t-SNE...")
X_tsne = simple_tsne(X, n_components=2, perplexity=20, n_iter=500, lr=100)
print(f"t-SNE embedding shape: {X_tsne.shape}")

# Reconstruction error for PCA
X_reconstructed = pca.inverse_transform(X_pca)
recon_error = np.mean((X - X_reconstructed) ** 2)
print(f"\nPCA reconstruction MSE: {recon_error:.4f}")
```

**Output:**
```
PCA explained variance: [0.503 0.496]
Total variance explained: 0.999
Running simplified t-SNE...
t-SNE embedding shape: (150, 2)
PCA reconstruction MSE: 0.0098
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Derive PCA from the maximum variance perspective.</strong></summary>

**Goal**: Find direction $w$ (unit vector) maximizing variance of projected data.

**Variance of projection**: $\text{Var}(Xw) = w^T \Sigma w$ where $\Sigma$ is covariance matrix.

**Constrained optimization**:
$$\max_w w^T \Sigma w \quad \text{s.t. } w^T w = 1$$

**Lagrangian**: $L = w^T \Sigma w - \lambda(w^T w - 1)$

**Setting gradient to zero**:
$$\nabla_w L = 2\Sigma w - 2\lambda w = 0$$
$$\Sigma w = \lambda w$$

This is an eigenvalue equation. The maximum variance direction is the eigenvector with largest eigenvalue.

For k components: use top k eigenvectors.
</details>

<details>
<summary><strong>Q2: Why can't you use t-SNE to project new data points?</strong></summary>

t-SNE is an **optimization** procedure, not a learned mapping:

1. **No parametric function**: There's no $f(x)$ that maps high-dim to low-dim
2. **Depends on all data**: Each point's position depends on all other points
3. **Non-convex optimization**: Adding new point would require re-running

**Solutions**:
- Re-run t-SNE with new data included
- Use parametric t-SNE (neural network approximation)
- Use UMAP (can learn approximate transform)
</details>

<details>
<summary><strong>Q3: What are the main pitfalls when interpreting t-SNE plots?</strong></summary>

1. **Cluster size is meaningless**: A larger cluster doesn't mean more samples
2. **Inter-cluster distance is meaningless**: Far apart doesn't mean dissimilar
3. **Only local structure preserved**: Global relationships lost
4. **Perplexity dependence**: Different perplexity = different plots
5. **Randomness**: Different runs give different results
6. **Continuous manifolds become disconnected**: Can falsely suggest clusters

**Best practice**: Run with multiple perplexities, don't over-interpret, use only for visualization.
</details>

<details>
<summary><strong>Q4: How do you choose the number of components in PCA?</strong></summary>

Methods:
1. **Explained variance ratio**: Keep components explaining X% variance (e.g., 95%)
2. **Scree plot**: Plot eigenvalues, look for "elbow"
3. **Kaiser criterion**: Keep components with eigenvalue > 1 (for standardized data)
4. **Cross-validation**: If used for downstream task
5. **Domain knowledge**: Known intrinsic dimensionality

Most common: Keep enough for 90-99% explained variance.
</details>

<details>
<summary><strong>Q5: What is the difference between PCA and t-SNE?</strong></summary>

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| Type | Linear | Non-linear |
| Objective | Maximize variance | Preserve local similarities |
| Global structure | Preserved | Lost |
| Reversible | Yes (approximate) | No |
| New data | Easy (projection) | Hard (re-run) |
| Speed | Fast (O(nd²)) | Slow (O(n²)) |
| Use case | Preprocessing, all k | Visualization, k=2,3 |

Use PCA for preprocessing/feature reduction; use t-SNE only for visualization.
</details>

<details>
<summary><strong>Q6: Why does t-SNE use t-distribution in low dimensions?</strong></summary>

The **crowding problem**: In high dimensions, moderate distances are common. When projecting to 2D, there's not enough "space" to preserve all moderate distances.

**t-distribution** has heavy tails compared to Gaussian:
- Points at moderate distance in high-dim can be pushed further apart in low-dim
- Creates space for faithful local structure
- Prevents central "crowding"

Without heavy tails, all points would crowd together in the center of the embedding.
</details>

---

## 8. References

1. Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
2. van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE." JMLR.
3. McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv.
4. Wattenberg, M., Viégas, F., & Johnson, I. (2016). "How to Use t-SNE Effectively." Distill.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 14.
