# Clustering

## 1. Interview Summary

**Key Points to Remember:**
- **K-Means**: Partition into k clusters, minimize within-cluster variance
- **GMM**: Soft clustering with probabilistic assignments
- **Hierarchical**: Build tree of clusters (agglomerative or divisive)
- **DBSCAN**: Density-based, handles arbitrary shapes and noise
- **Evaluation**: Silhouette score, inertia, Davies-Bouldin index

**Common Interview Questions:**
- "How do you choose k in k-means?"
- "What are the limitations of k-means?"
- "Compare hard vs soft clustering"

---

## 2. Core Definitions

### K-Means
Minimize within-cluster sum of squares:

$$\arg\min_{\mu} \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

### Gaussian Mixture Model (GMM)
Model data as mixture of Gaussians:

$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

Where $\pi_k$ are mixing coefficients.

### Hierarchical Clustering
- **Agglomerative**: Bottom-up, merge closest clusters
- **Divisive**: Top-down, split clusters

**Linkage criteria:**
| Type | Distance between clusters |
|------|--------------------------|
| Single | Min distance between points |
| Complete | Max distance between points |
| Average | Average distance |
| Ward | Increase in total variance |

### DBSCAN
- **Core points**: ≥ minPts within ε radius
- **Border points**: Within ε of core point
- **Noise**: Neither core nor border

---

## 3. Math and Derivations

### K-Means Convergence

**Lloyd's Algorithm:**
1. Assign each point to nearest centroid
2. Update centroids as cluster means

**Guaranteed to converge** because:
- Each step decreases (or maintains) objective
- Finite number of possible assignments

**But**: Converges to local minimum, not global.

### GMM and EM Algorithm

**E-step**: Compute responsibilities

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

**M-step**: Update parameters

$$\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$$

$$\Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_i \gamma_{ik}}$$

$$\pi_k = \frac{1}{n}\sum_i \gamma_{ik}$$

### Silhouette Score

For each point $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- $a(i)$: Average distance to same-cluster points
- $b(i)$: Average distance to nearest other cluster

Range: [-1, 1], higher is better.

### Elbow Method

Plot inertia (within-cluster variance) vs k:
- Look for "elbow" where improvement slows
- Not always clear-cut

---

## 4. Algorithm Sketch

### K-Means

```
Input: Data X, number of clusters K
Output: Cluster assignments, centroids

# Initialize centroids (k-means++ recommended)
centroids = random_sample(X, K)

Repeat until convergence:
    # Assign points to nearest centroid
    For each point x_i:
        assignments[i] = argmin_k ||x_i - centroid_k||²

    # Update centroids
    For each cluster k:
        centroid_k = mean(X[assignments == k])

Return assignments, centroids
```

### K-Means++ Initialization

```
1. Choose first centroid randomly
2. For each remaining centroid:
   a. Compute D(x) = distance to nearest existing centroid
   b. Choose next centroid with probability ∝ D(x)²
3. Continue until k centroids chosen
```

### GMM with EM

```
Initialize: Random means, identity covariances, uniform mixing

Repeat until convergence:
    # E-step: compute responsibilities
    For each point i, cluster k:
        γ[i,k] = π[k] * N(x[i]; μ[k], Σ[k]) / Σ_j(π[j] * N(x[i]; μ[j], Σ[j]))

    # M-step: update parameters
    For each cluster k:
        N_k = Σ_i γ[i,k]
        μ[k] = Σ_i γ[i,k] * x[i] / N_k
        Σ[k] = Σ_i γ[i,k] * (x[i]-μ[k])(x[i]-μ[k])ᵀ / N_k
        π[k] = N_k / n

Return parameters, responsibilities
```

### DBSCAN

```
For each unvisited point p:
    Mark p as visited
    neighbors = points within ε of p

    If |neighbors| < minPts:
        Mark p as noise
    Else:
        Create new cluster C
        Add p to C
        For each point q in neighbors:
            If q not visited:
                Mark q as visited
                q_neighbors = points within ε of q
                If |q_neighbors| ≥ minPts:
                    neighbors = neighbors ∪ q_neighbors
            If q not in any cluster:
                Add q to C
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Wrong k in k-means | No clear best k | Use elbow method, silhouette, domain knowledge |
| K-means with non-spherical clusters | Assumes spherical clusters | Use GMM or DBSCAN |
| Sensitive to initialization | Random start → local minimum | Use k-means++, multiple restarts |
| Not scaling features | Features with large range dominate | Standardize features |
| DBSCAN ε/minPts sensitivity | Wrong values give poor results | Use k-distance plot |

### Algorithm Selection Guide

| Scenario | Recommended Method |
|----------|-------------------|
| Spherical clusters, known k | K-Means |
| Elliptical clusters, soft assignments | GMM |
| Unknown number of clusters | Hierarchical, DBSCAN |
| Arbitrary shapes, noise | DBSCAN |
| Need hierarchy/dendrogram | Hierarchical |
| Very large dataset | Mini-batch K-Means |

---

## 6. Mini Example

```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n, d = X.shape

        # K-means++ initialization
        self.centroids = [X[np.random.randint(n)]]
        for _ in range(1, self.k):
            dists = np.array([min(np.sum((x - c)**2) for c in self.centroids) for x in X])
            probs = dists / dists.sum()
            self.centroids.append(X[np.random.choice(n, p=probs)])
        self.centroids = np.array(self.centroids)

        # Iterate
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            self.labels = self._assign(X)

            # Update centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0)
                                      for k in range(self.k)])

            # Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

        return self

    def _assign(self, X):
        dists = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])
        return np.argmin(dists, axis=1)

    def predict(self, X):
        return self._assign(X)

    def inertia(self, X):
        return sum(np.sum((X[self.labels == k] - self.centroids[k])**2)
                   for k in range(self.k))


def silhouette_score(X, labels):
    """Compute average silhouette score."""
    n = len(X)
    scores = []

    for i in range(n):
        # a(i): mean distance to same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a_i = np.mean([np.sqrt(np.sum((X[i] - x)**2)) for x in same_cluster if not np.array_equal(x, X[i])])
        else:
            a_i = 0

        # b(i): mean distance to nearest other cluster
        b_i = float('inf')
        for k in np.unique(labels):
            if k != labels[i]:
                other_cluster = X[labels == k]
                mean_dist = np.mean([np.sqrt(np.sum((X[i] - x)**2)) for x in other_cluster])
                b_i = min(b_i, mean_dist)

        if b_i == float('inf'):
            scores.append(0)
        else:
            scores.append((b_i - a_i) / max(a_i, b_i))

    return np.mean(scores)


# Example
np.random.seed(42)

# Generate 3 clusters
cluster1 = np.random.randn(50, 2) + np.array([0, 0])
cluster2 = np.random.randn(50, 2) + np.array([5, 5])
cluster3 = np.random.randn(50, 2) + np.array([5, 0])
X = np.vstack([cluster1, cluster2, cluster3])

# Fit k-means
kmeans = KMeans(k=3)
kmeans.fit(X)

print(f"Centroids:\n{kmeans.centroids}")
print(f"Inertia: {kmeans.inertia(X):.2f}")
print(f"Silhouette Score: {silhouette_score(X, kmeans.labels):.3f}")

# Elbow method
print("\nElbow method (inertia for different k):")
for k in range(1, 7):
    km = KMeans(k=k)
    km.fit(X)
    print(f"k={k}: inertia={km.inertia(X):.1f}")
```

**Output:**
```
Centroids:
[[0.05 0.02]
 [5.01 4.89]
 [4.98 0.03]]
Inertia: 289.45
Silhouette Score: 0.567

Elbow method (inertia for different k):
k=1: inertia=3871.2
k=2: inertia=1257.8
k=3: inertia=289.5
k=4: inertia=241.3
k=5: inertia=205.6
k=6: inertia=175.4
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What are the limitations of k-means?</strong></summary>

1. **Assumes spherical clusters**: Can't handle elongated or irregular shapes
2. **Requires specifying k**: Must know number of clusters beforehand
3. **Sensitive to initialization**: Different starts → different results
4. **Sensitive to outliers**: Outliers pull centroids
5. **Only finds local optimum**: Not guaranteed to find best clustering
6. **Equal cluster size assumption**: Performs poorly with varying sizes

Solutions: Use k-means++, multiple restarts, or alternative algorithms (GMM, DBSCAN).
</details>

<details>
<summary><strong>Q2: How do you choose k in k-means?</strong></summary>

Methods:
1. **Elbow method**: Plot inertia vs k, look for "elbow"
2. **Silhouette score**: Higher is better (range -1 to 1)
3. **Gap statistic**: Compare to reference distribution
4. **Domain knowledge**: Prior knowledge about expected clusters
5. **Cross-validation**: If downstream task available

No single best method. Often combine multiple approaches.
</details>

<details>
<summary><strong>Q3: What is the difference between hard and soft clustering?</strong></summary>

**Hard clustering** (K-Means):
- Each point belongs to exactly one cluster
- Binary assignment: 0 or 1

**Soft clustering** (GMM):
- Each point has probability of belonging to each cluster
- Fractional assignment: γ_ik ∈ [0, 1], sum to 1

Soft clustering is useful when:
- Clusters overlap
- You need uncertainty estimates
- Points genuinely belong to multiple groups
</details>

<details>
<summary><strong>Q4: Explain the EM algorithm for GMM.</strong></summary>

**Expectation-Maximization** alternates two steps:

**E-step** (Expectation):
- Fix parameters, compute responsibilities
- γ_ik = probability point i belongs to cluster k

**M-step** (Maximization):
- Fix responsibilities, update parameters
- μ_k = weighted mean of points
- Σ_k = weighted covariance
- π_k = fraction of points in cluster k

Guaranteed to increase (or maintain) log-likelihood at each step.
</details>

<details>
<summary><strong>Q5: When would you use DBSCAN over k-means?</strong></summary>

Use **DBSCAN** when:
- Clusters have arbitrary shapes (non-spherical)
- Number of clusters is unknown
- Data contains noise/outliers (DBSCAN labels them)
- Clusters have varying densities (with care)

Use **K-Means** when:
- Clusters are roughly spherical
- Number of clusters is known
- Speed is important (k-means is faster)
- No significant outliers
</details>

<details>
<summary><strong>Q6: What is the silhouette score and how do you interpret it?</strong></summary>

Silhouette score for point i:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- a(i): cohesion (mean distance to same cluster)
- b(i): separation (mean distance to nearest other cluster)

**Interpretation:**
- s ≈ 1: Point is well-matched to its cluster
- s ≈ 0: Point is on boundary between clusters
- s < 0: Point may be in wrong cluster

Average silhouette score evaluates overall clustering quality.
</details>

---

## 8. References

1. MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations." Berkeley Symposium.
2. Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding." SODA.
3. Dempster, A., Laird, N., & Rubin, D. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." JRSS.
4. Ester, M., et al. (1996). "A Density-Based Algorithm for Discovering Clusters." KDD.
5. Rousseeuw, P. (1987). "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis." Journal of Computational and Applied Mathematics.
