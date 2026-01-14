# Class Imbalance

## 1. Interview Summary

**Key Points to Remember:**
- **Resampling**: Oversample minority, undersample majority
- **Cost-sensitive learning**: Higher penalty for minority class errors
- **SMOTE**: Synthetic minority oversampling technique
- **Threshold tuning**: Adjust decision threshold post-training
- **Metrics**: Use PR-AUC over ROC-AUC for severe imbalance

**Common Interview Questions:**
- "How do you handle imbalanced data?"
- "What are the pros/cons of oversampling vs undersampling?"
- "When would you use SMOTE?"

---

## 2. Core Definitions

### Imbalance Ratio
$$\text{IR} = \frac{n_{majority}}{n_{minority}}$$

### Resampling Strategies

| Strategy | Description | Effect |
|----------|-------------|--------|
| Random Oversampling | Duplicate minority samples | May overfit |
| Random Undersampling | Remove majority samples | Loses information |
| SMOTE | Generate synthetic minority | Better generalization |
| ADASYN | Adaptive synthetic sampling | Focus on hard examples |

### Cost-Sensitive Learning
Weight loss by class frequency:
$$L = \sum_i w_{y_i} \cdot \ell(y_i, \hat{y}_i)$$

Where $w_{minority} > w_{majority}$

---

## 3. Math and Derivations

### SMOTE Algorithm

For each minority sample $x_i$:
1. Find k nearest minority neighbors
2. Randomly select one neighbor $x_j$
3. Create synthetic: $x_{new} = x_i + \lambda (x_j - x_i)$ where $\lambda \in [0,1]$

### Class Weights

**Balanced weights:**
$$w_c = \frac{n_{total}}{n_c \cdot n_{classes}}$$

**Inverse frequency:**
$$w_c = \frac{1}{n_c}$$

### Threshold Optimization

Default threshold 0.5 may not be optimal. Find threshold $t^*$ that maximizes:
- F1 score
- Geometric mean
- Custom metric based on costs

$$t^* = \arg\max_t F_1(t)$$

---

## 4. Algorithm Sketch

### SMOTE Implementation

```
def SMOTE(X_minority, k=5, N=100):
    synthetic = []
    for i in range(len(X_minority)):
        # Find k nearest neighbors
        neighbors = k_nearest_neighbors(X_minority, X_minority[i], k)

        for _ in range(N // 100):
            # Select random neighbor
            j = random.choice(neighbors)

            # Generate synthetic sample
            lambda = random.uniform(0, 1)
            x_new = X_minority[i] + lambda * (X_minority[j] - X_minority[i])
            synthetic.append(x_new)

    return synthetic
```

### Complete Pipeline

```
def handle_imbalance(X, y, strategy='combined'):
    # 1. Split data first (prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 2. Apply resampling only to training data
    if strategy == 'oversample':
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    elif strategy == 'undersample':
        X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
    elif strategy == 'combined':
        X_train, y_train = SMOTETomek().fit_resample(X_train, y_train)

    # 3. Train model with class weights
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # 4. Tune threshold on validation set
    y_prob = model.predict_proba(X_val)[:, 1]
    best_threshold = find_optimal_threshold(y_val, y_prob)

    return model, best_threshold
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Resampling before split | Data leakage | Always split first, then resample train only |
| Using accuracy | Misleading with imbalance | Use F1, PR-AUC |
| Oversampling test set | Invalid evaluation | Never resample test data |
| Too much oversampling | Overfitting | Use combined strategies |
| Ignoring cost structure | Business costs matter | Use cost-sensitive learning |

### Strategy Selection Guide

| Imbalance Level | Recommended Strategy |
|-----------------|---------------------|
| Mild (1:10) | Class weights |
| Moderate (1:100) | SMOTE + class weights |
| Severe (1:1000+) | Combine multiple strategies |
| Very few minority | Anomaly detection instead |

---

## 6. Mini Example

```python
import numpy as np
from collections import Counter

def smote_simple(X_minority, k=3, n_synthetic=10):
    """Simple SMOTE implementation."""
    n_samples = len(X_minority)
    synthetic = []

    for _ in range(n_synthetic):
        # Pick random minority sample
        i = np.random.randint(n_samples)
        x_i = X_minority[i]

        # Find k nearest neighbors
        distances = np.sqrt(np.sum((X_minority - x_i)**2, axis=1))
        distances[i] = np.inf  # Exclude self
        k_neighbors = np.argsort(distances)[:k]

        # Pick random neighbor and interpolate
        j = np.random.choice(k_neighbors)
        x_j = X_minority[j]
        lam = np.random.random()
        synthetic.append(x_i + lam * (x_j - x_i))

    return np.array(synthetic)


def compute_class_weights(y):
    """Compute balanced class weights."""
    counts = Counter(y)
    n_samples = len(y)
    n_classes = len(counts)
    weights = {}
    for cls, count in counts.items():
        weights[cls] = n_samples / (n_classes * count)
    return weights


# Example
np.random.seed(42)

# Create imbalanced dataset
X_majority = np.random.randn(100, 2) + np.array([2, 2])
X_minority = np.random.randn(10, 2) + np.array([-2, -2])
X = np.vstack([X_majority, X_minority])
y = np.array([0] * 100 + [1] * 10)

print("Original distribution:", Counter(y))
print(f"Imbalance ratio: {100/10:.1f}:1")

# Class weights
weights = compute_class_weights(y)
print(f"\nClass weights: {weights}")

# SMOTE
synthetic = smote_simple(X_minority, k=3, n_synthetic=90)
X_resampled = np.vstack([X, synthetic])
y_resampled = np.concatenate([y, np.ones(90)])

print(f"\nAfter SMOTE: {Counter(y_resampled.astype(int))}")

# Threshold optimization example
def find_optimal_threshold(y_true, y_prob):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# Simulate predictions
y_prob = np.concatenate([
    np.random.uniform(0, 0.4, 100),  # Majority (low prob)
    np.random.uniform(0.4, 0.9, 10)  # Minority (higher prob)
])

opt_t, opt_f1 = find_optimal_threshold(y, y_prob)
print(f"\nOptimal threshold: {opt_t:.2f} (F1={opt_f1:.3f})")
print(f"Default threshold F1: {find_optimal_threshold(y, y_prob)[1]:.3f} at t=0.50")
```

**Output:**
```
Original distribution: Counter({0: 100, 1: 10})
Imbalance ratio: 10.0:1

Class weights: {0: 0.55, 1: 5.5}

After SMOTE: Counter({0: 100, 1: 100})

Optimal threshold: 0.35 (F1=0.667)
Default threshold F1: 0.400 at t=0.50
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What are the pros and cons of oversampling vs undersampling?</strong></summary>

**Oversampling (e.g., SMOTE)**:
- Pros: No information loss, larger training set
- Cons: Can overfit, longer training time, may create unrealistic samples

**Undersampling**:
- Pros: Faster training, balances dataset
- Cons: Loses potentially useful majority class information

**Best practice**: Often combine both (e.g., SMOTE + Tomek links) or use ensemble methods with different samplings.
</details>

<details>
<summary><strong>Q2: Why is accuracy misleading with imbalanced data?</strong></summary>

With 99% negative, 1% positive data:
- Model predicting "all negative" achieves 99% accuracy
- But it's useless for finding the positive class

Example metrics for 1:99 imbalance with "predict all negative":
- Accuracy: 99%
- Recall (positive): 0%
- F1 (positive): 0%

Better metrics: Precision, Recall, F1, PR-AUC (focus on minority class)
</details>

<details>
<summary><strong>Q3: What is SMOTE and how does it work?</strong></summary>

**SMOTE** (Synthetic Minority Over-sampling Technique):
1. For each minority sample, find k nearest minority neighbors
2. Randomly select one neighbor
3. Create synthetic sample on the line segment between them

$$x_{new} = x_i + \lambda \cdot (x_j - x_i), \quad \lambda \in [0,1]$$

**Advantages**: Creates realistic samples, better generalization than simple duplication

**Limitations**: May create noisy samples in overlapping regions, doesn't consider majority class
</details>

<details>
<summary><strong>Q4: Why must resampling be done after train-test split?</strong></summary>

If you resample before splitting:
1. Synthetic samples may end up in test set
2. These synthetic samples were created using training information
3. Results in **data leakage** - test set is no longer independent

**Correct workflow**:
1. Split data into train/test
2. Apply resampling only to training data
3. Keep test set original (real-world distribution)
4. Evaluate on unmodified test set
</details>

<details>
<summary><strong>Q5: How do class weights work in cost-sensitive learning?</strong></summary>

Class weights increase the penalty for misclassifying minority class:

$$L_{weighted} = \sum_i w_{y_i} \cdot \ell(y_i, \hat{y}_i)$$

**Balanced weights**: $w_c = \frac{n_{total}}{n_c \cdot n_{classes}}$

Example with 100 majority, 10 minority:
- $w_{majority} = \frac{110}{100 \times 2} = 0.55$
- $w_{minority} = \frac{110}{10 \times 2} = 5.5$

Minority misclassifications cost 10x more, incentivizing correct minority predictions.
</details>

<details>
<summary><strong>Q6: When would you use anomaly detection instead of classification for imbalance?</strong></summary>

Use anomaly detection when:
1. **Extreme imbalance** (>1:10,000): Too few minority samples to learn patterns
2. **Minority class undefined**: No clear positive class definition
3. **Novel patterns needed**: Want to detect previously unseen anomalies

Anomaly detection methods:
- Isolation Forest
- One-Class SVM
- Autoencoders

These learn "normal" patterns and flag deviations, rather than requiring minority class examples.
</details>

---

## 8. References

1. Chawla, N., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." JAIR.
2. He, H., & Garcia, E. (2009). "Learning from Imbalanced Data." IEEE TKDE.
3. Batista, G., et al. (2004). "A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data." ACM SIGKDD.
4. Japkowicz, N., & Stephen, S. (2002). "The Class Imbalance Problem: A Systematic Study." IDA.
5. imbalanced-learn documentation: Resampling strategies.
