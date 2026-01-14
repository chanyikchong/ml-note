# Learning Paradigms

## 1. Interview Summary

**Key Points to Remember:**
- **Supervised Learning**: Learn from labeled data; predict outputs for new inputs
- **Unsupervised Learning**: Find patterns in unlabeled data; no target variable
- **Self-Supervised Learning**: Create pseudo-labels from data structure itself
- Know examples of each and when to use them
- Understand the data requirements for each paradigm

**Common Interview Questions:**
- "What's the difference between supervised and unsupervised learning?"
- "Give an example of self-supervised learning"
- "When would you use unsupervised vs supervised methods?"

---

## 2. Core Definitions

### Supervised Learning
Learning a mapping $f: X \rightarrow Y$ from labeled training data $\{(x_i, y_i)\}_{i=1}^n$.

**Characteristics:**
- Requires labeled data (input-output pairs)
- Goal: Minimize prediction error on unseen data
- Types: Classification (discrete $Y$), Regression (continuous $Y$)

**Examples:**
- Email spam detection (classification)
- House price prediction (regression)
- Image classification (classification)

### Unsupervised Learning
Learning patterns from unlabeled data $\{x_i\}_{i=1}^n$ without target variables.

**Characteristics:**
- No labels required
- Goal: Discover structure, patterns, or representations
- Types: Clustering, dimensionality reduction, density estimation

**Examples:**
- Customer segmentation (clustering)
- Anomaly detection (density estimation)
- Feature extraction with PCA (dimensionality reduction)

### Self-Supervised Learning
Creating supervisory signals from the data itself, then learning representations.

**Characteristics:**
- Generates pseudo-labels automatically from data structure
- Bridge between supervised and unsupervised
- Powerful for representation learning

**Examples:**
- Language models predicting next word (NLP)
- Contrastive learning (computer vision)
- Masked autoencoding (BERT, MAE)

---

## 3. Math and Derivations

### Supervised Learning Formalization

Given dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ where $x_i \in \mathcal{X}$, $y_i \in \mathcal{Y}$.

**Empirical Risk Minimization (ERM):**

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(x_i), y_i)$$

where $\mathcal{L}$ is the loss function and $\mathcal{F}$ is the hypothesis class.

**True Risk (Generalization Error):**

$$R(f) = \mathbb{E}_{(x,y) \sim P}[\mathcal{L}(f(x), y)]$$

### Unsupervised Learning Formalization

Given dataset $\mathcal{D} = \{x_i\}_{i=1}^n$ where $x_i \in \mathcal{X}$.

**Clustering Objective (K-Means):**

$$\min_{C_1,...,C_k} \sum_{j=1}^{k} \sum_{x \in C_j} \|x - \mu_j\|^2$$

**Density Estimation:**

$$\hat{p}(x) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - x_i)$$

### Self-Supervised Learning

**Contrastive Loss (InfoNCE):**

$$\mathcal{L} = -\log \frac{\exp(sim(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(sim(z_i, z_k)/\tau)}$$

where $z_i, z_j$ are embeddings of augmented views of the same sample.

---

## 4. Algorithm Sketch

### Supervised Learning Pipeline
```
1. Collect labeled data {(x_i, y_i)}
2. Split into train/validation/test sets
3. Choose model family F
4. Train: minimize loss on training set
5. Validate: tune hyperparameters
6. Test: evaluate final performance
7. Deploy model
```

### Unsupervised Learning Pipeline
```
1. Collect unlabeled data {x_i}
2. Choose method (clustering, dim reduction, etc.)
3. Fit model to discover structure
4. Evaluate using internal metrics or downstream tasks
5. Interpret and use discovered patterns
```

### Self-Supervised Learning Pipeline
```
1. Collect unlabeled data {x_i}
2. Define pretext task (e.g., predict masked tokens)
3. Generate pseudo-labels from data structure
4. Train encoder on pretext task
5. Fine-tune or use features for downstream tasks
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Using supervised methods without enough labels | Insufficient labeled data | Consider semi-supervised or self-supervised |
| Expecting unsupervised to match supervised accuracy | No ground truth to guide learning | Set realistic expectations; use for exploration |
| Ignoring domain knowledge in clustering | Treating it as purely algorithmic | Incorporate domain expertise in design |
| Over-relying on self-supervised features | Features may not transfer perfectly | Fine-tune on target task |
| Wrong paradigm choice | Not analyzing problem requirements | Match paradigm to data availability and goals |

---

## 6. Mini Example

### Python Example: Comparing Paradigms

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Generate data
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

# Supervised: Use labels to learn classifier
clf = LogisticRegression()
clf.fit(X, y_true)
print(f"Supervised accuracy: {clf.score(X, y_true):.3f}")

# Unsupervised: Cluster without labels
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)
# Note: cluster labels may not match original labels
print(f"K-Means found {len(set(y_pred))} clusters")

# Dimensionality reduction (unsupervised)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Variance explained: {sum(pca.explained_variance_ratio_):.3f}")
```

**Output:**
```
Supervised accuracy: 0.993
K-Means found 3 clusters
Variance explained: 0.997
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What distinguishes supervised from unsupervised learning?</strong></summary>

Supervised learning uses labeled data (input-output pairs) to learn a mapping, while unsupervised learning works with unlabeled data to discover patterns or structure without explicit targets.
</details>

<details>
<summary><strong>Q2: Give three examples of self-supervised learning tasks.</strong></summary>

1. **Masked language modeling** (BERT): Predict masked tokens in text
2. **Contrastive learning** (SimCLR): Learn representations by distinguishing augmented views
3. **Next sentence prediction**: Predict if two sentences are consecutive
4. **Image rotation prediction**: Predict rotation angle applied to image
5. **Masked autoencoding** (MAE): Reconstruct masked image patches
</details>

<details>
<summary><strong>Q3: When would you prefer unsupervised over supervised methods?</strong></summary>

- When labeled data is unavailable or expensive to obtain
- For exploratory data analysis to discover unknown patterns
- For anomaly detection where anomalies are rare/undefined
- For data compression and dimensionality reduction
- When you want to understand the underlying data structure
</details>

<details>
<summary><strong>Q4: What is the key idea behind self-supervised learning?</strong></summary>

Self-supervised learning creates supervisory signals from the data itself by defining pretext tasks. The model learns useful representations by solving these tasks (e.g., predicting masked portions, distinguishing augmented views). These representations can then transfer to downstream supervised tasks with limited labeled data.
</details>

<details>
<summary><strong>Q5: What is Empirical Risk Minimization (ERM)?</strong></summary>

ERM is the principle of choosing a hypothesis that minimizes the average loss on the training data:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(x_i), y_i)$$

It approximates the true risk (expected loss over the data distribution) using the empirical distribution of training samples.
</details>

<details>
<summary><strong>Q6: Why might unsupervised clustering not achieve the same accuracy as supervised classification?</strong></summary>

Unsupervised clustering lacks access to ground truth labels, so:
- Cluster boundaries may not align with true class boundaries
- The algorithm optimizes for geometric/statistical coherence, not classification accuracy
- The number of clusters may not match the number of true classes
- Cluster assignments are arbitrary (permutation invariant)
</details>

---

## 8. References

1. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. Chen, T., et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML.
4. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
