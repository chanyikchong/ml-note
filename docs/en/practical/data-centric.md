# Data-Centric ML Issues

## 1. Interview Summary

**Key Points to Remember:**
- **Label noise**: Mislabeled data degrades model performance
- **Dataset shift**: Train and test distributions differ
- **Outliers**: Extreme values that may or may not be errors
- **Data quality > model complexity**: Better data often beats better models

**Common Interview Questions:**
- "How do you handle noisy labels in your dataset?"
- "What is dataset shift and how do you detect it?"
- "How do you decide whether to remove or keep outliers?"

---

## 2. Core Definitions

### Types of Dataset Shift

| Type | Description | Example |
|------|-------------|---------|
| Covariate shift | $P(X)$ changes, $P(Y|X)$ same | Different demographics |
| Label shift | $P(Y)$ changes, $P(X|Y)$ same | Class proportion changes |
| Concept drift | $P(Y|X)$ changes | Fraud patterns evolve |
| Domain shift | Both $P(X)$ and $P(Y|X)$ change | New data source |

### Label Noise Types

| Type | Description | Impact |
|------|-------------|--------|
| Random noise | Random mislabeling | Increased variance |
| Systematic noise | Consistent errors (annotator bias) | Biased model |
| Class-dependent | Some classes more noisy | Unfair predictions |

### Outlier Types

| Type | Description | Action |
|------|-------------|--------|
| Error outliers | Data entry mistakes | Remove or correct |
| Natural outliers | True extreme values | Keep (usually) |
| Influential outliers | High leverage points | Investigate |

---

## 3. Math and Derivations

### Covariate Shift Correction

If $P_{train}(X) \neq P_{test}(X)$ but $P(Y|X)$ is same:

**Importance weighting:**
$$w(x) = \frac{P_{test}(x)}{P_{train}(x)}$$

Weighted loss:
$$L_{corrected} = \sum_i w(x_i) \cdot \ell(y_i, \hat{y}_i)$$

### Label Noise Model

With noise rate $\eta$ (probability of label flip):
$$P(\tilde{y}|x) = (1-\eta) P(y|x) + \eta P(y_{wrong}|x)$$

**Forward correction:**
$$P(y|x) = \frac{P(\tilde{y}|x) - \eta P(y_{wrong}|x)}{1 - \eta}$$

### Outlier Detection (Z-score)

$$z_i = \frac{x_i - \mu}{\sigma}$$

Flag as outlier if $|z_i| > 3$ (assuming Gaussian).

### IQR Method

$$\text{Outlier if } x < Q_1 - 1.5 \cdot IQR \text{ or } x > Q_3 + 1.5 \cdot IQR$$

Where $IQR = Q_3 - Q_1$.

---

## 4. Algorithm Sketch

### Confident Learning (Label Noise Detection)

```
def confident_learning(X, y, model):
    # Step 1: Get predicted probabilities
    probs = cross_val_predict(model, X, y, method='predict_proba')

    # Step 2: Estimate threshold per class
    thresholds = []
    for c in classes:
        # Average probability for samples labeled c
        thresholds.append(np.mean(probs[y == c, c]))

    # Step 3: Create confident joint matrix
    C = np.zeros((n_classes, n_classes))
    for i, (yi, pi) in enumerate(zip(y, probs)):
        predicted = np.argmax(pi > thresholds)
        C[yi, predicted] += 1

    # Step 4: Find label errors
    errors = []
    for i, (yi, pi) in enumerate(zip(y, probs)):
        if pi[yi] < thresholds[yi] and max(pi) > thresholds[np.argmax(pi)]:
            errors.append(i)

    return errors
```

### Dataset Shift Detection

```
def detect_covariate_shift(X_train, X_test):
    # Train classifier to distinguish train vs test
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    X = np.vstack([X_train, X_test])

    model = LogisticRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    # AUC near 0.5 = no shift; near 1.0 = significant shift
    return np.mean(scores)
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Removing all outliers | Some are valid | Domain knowledge first |
| Ignoring label noise | Assumes labels are correct | Use confident learning |
| Training on shifted data | Distribution mismatch | Check for shift, reweight |
| Overfitting to noise | Model memorizes errors | Regularization, noise-robust loss |
| Single annotator | No quality check | Multiple annotators, agreement scores |

### Data Quality Checklist

```
1. Check for duplicates
2. Validate data types and ranges
3. Check for missing values patterns
4. Analyze class balance
5. Look for label consistency
6. Check train/test distribution similarity
7. Identify potential leakage features
8. Validate temporal ordering (if applicable)
```

---

## 6. Mini Example

```python
import numpy as np

def detect_outliers_iqr(data):
    """Detect outliers using IQR method."""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = (data < lower) | (data > upper)
    return outliers, lower, upper


def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using z-score."""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold


def estimate_label_noise(y_true, y_noisy):
    """Estimate noise rate from known labels."""
    disagreement = np.sum(y_true != y_noisy)
    noise_rate = disagreement / len(y_true)
    return noise_rate


def add_label_noise(y, noise_rate=0.1):
    """Add random label noise."""
    n_flip = int(len(y) * noise_rate)
    flip_idx = np.random.choice(len(y), n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]  # Binary flip
    return y_noisy, flip_idx


def detect_shift_simple(X_train, X_test):
    """Simple covariate shift detection using feature statistics."""
    train_stats = {'mean': np.mean(X_train, axis=0), 'std': np.std(X_train, axis=0)}
    test_stats = {'mean': np.mean(X_test, axis=0), 'std': np.std(X_test, axis=0)}

    # Compare distributions
    mean_diff = np.abs(train_stats['mean'] - test_stats['mean'])
    std_ratio = test_stats['std'] / (train_stats['std'] + 1e-8)

    return mean_diff, std_ratio


# Example
np.random.seed(42)

# Outlier detection
print("=== Outlier Detection ===")
data = np.concatenate([np.random.randn(100), [10, -8, 15]])  # Add outliers
outliers_iqr, lower, upper = detect_outliers_iqr(data)
outliers_zscore = detect_outliers_zscore(data)

print(f"Data shape: {data.shape}")
print(f"IQR bounds: [{lower:.2f}, {upper:.2f}]")
print(f"IQR outliers: {np.sum(outliers_iqr)} samples")
print(f"Z-score outliers: {np.sum(outliers_zscore)} samples")

# Label noise
print("\n=== Label Noise ===")
y_true = np.random.randint(0, 2, 100)
y_noisy, flipped = add_label_noise(y_true, noise_rate=0.15)
estimated_noise = estimate_label_noise(y_true, y_noisy)
print(f"True noise rate: 0.15")
print(f"Estimated noise rate: {estimated_noise:.2f}")
print(f"Flipped indices (first 5): {flipped[:5]}")

# Dataset shift
print("\n=== Dataset Shift Detection ===")
X_train = np.random.randn(100, 3)
X_test_no_shift = np.random.randn(50, 3)  # Same distribution
X_test_shift = np.random.randn(50, 3) + np.array([2, 0, -1])  # Shifted

mean_diff_no, std_ratio_no = detect_shift_simple(X_train, X_test_no_shift)
mean_diff_yes, std_ratio_yes = detect_shift_simple(X_train, X_test_shift)

print("No shift - mean differences:", mean_diff_no.round(2))
print("With shift - mean differences:", mean_diff_yes.round(2))
```

**Output:**
```
=== Outlier Detection ===
Data shape: (103,)
IQR bounds: [-2.25, 2.31]
IQR outliers: 3 samples
Z-score outliers: 3 samples

=== Label Noise ===
True noise rate: 0.15
Estimated noise rate: 0.15
Flipped indices (first 5): [51 92 14 71 60]

=== Dataset Shift Detection ===
No shift - mean differences: [0.12 0.08 0.15]
With shift - mean differences: [2.05 0.11 1.12]
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is the difference between covariate shift and concept drift?</strong></summary>

**Covariate shift**:
- Input distribution $P(X)$ changes
- Relationship $P(Y|X)$ stays the same
- Example: Training on young users, testing on old users

**Concept drift**:
- Relationship $P(Y|X)$ changes over time
- Same input may map to different outputs
- Example: Fraud patterns evolving

**Key insight**: Covariate shift can be corrected by importance weighting; concept drift requires model updates.
</details>

<details>
<summary><strong>Q2: How do you handle noisy labels in training data?</strong></summary>

**Strategies**:

1. **Noise-robust losses**: Use MAE instead of cross-entropy (less sensitive)

2. **Confident learning**: Detect and remove likely mislabeled samples

3. **Co-teaching**: Train two networks, each teaches the other on "clean" samples

4. **Label smoothing**: Soft labels reduce overconfidence on wrong labels

5. **Multiple annotators**: Use majority vote or model annotator reliability

**Key**: Don't assume labels are 100% correct; build in noise tolerance.
</details>

<details>
<summary><strong>Q3: When should you remove outliers vs keep them?</strong></summary>

**Remove when**:
- Clear data entry errors (e.g., age = 999)
- Impossible values (negative prices)
- Measurement equipment failure

**Keep when**:
- Natural extreme values (wealthy customers)
- Important edge cases for the model
- Outlier is the thing you're trying to detect (fraud)

**Investigate when**:
- High leverage points that affect model significantly
- Pattern of outliers suggests systematic issue

**Best practice**: Document decisions; consider robust methods (median, quantile regression).
</details>

<details>
<summary><strong>Q4: How do you detect dataset shift between train and test?</strong></summary>

**Methods**:

1. **Classifier two-sample test**: Train model to distinguish train vs test
   - AUC ~ 0.5: No shift
   - AUC ~ 1.0: Significant shift

2. **Feature distribution comparison**:
   - Compare means, variances, quantiles
   - KS test, Chi-squared test for each feature

3. **Model prediction distribution**:
   - Compare $P(\hat{y})$ on train vs test
   - Should be similar if no shift

4. **Time-based analysis**: Plot metrics over time for gradual drift

**Action**: If shift detected, consider importance weighting or collecting new training data.
</details>

<details>
<summary><strong>Q5: What is importance weighting for covariate shift?</strong></summary>

When $P_{train}(X) \neq P_{test}(X)$, weight training samples:

$$w(x) = \frac{P_{test}(x)}{P_{train}(x)}$$

**Estimation methods**:
1. Train classifier to distinguish train/test
2. Use density ratio estimation (KLIEP, uLSIF)
3. Propensity scores

**Weighted training**:
$$\min_\theta \sum_i w(x_i) \cdot L(y_i, f_\theta(x_i))$$

**Caveat**: Unstable when $P_{train}(x)$ is very small (high variance weights).
</details>

<details>
<summary><strong>Q6: How does label noise affect different models?</strong></summary>

**More robust** (to random noise):
- Tree-based models (decision boundaries from data)
- K-NN (local majority voting helps)
- Regularized models (prevents memorization)

**Less robust**:
- Neural networks (can memorize noise)
- Models with high capacity
- Loss functions that penalize confident wrong predictions

**Mitigation**:
- Use noise-robust loss (MAE, truncated loss)
- Early stopping (prevents memorization)
- Confident learning to clean data
- Ensemble methods
</details>

---

## 8. References

1. Northcutt, C., et al. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels." JAIR.
2. Sugiyama, M., & Kawanabe, M. (2012). *Machine Learning in Non-Stationary Environments*. MIT Press.
3. Quinonero-Candela, J., et al. (2009). *Dataset Shift in Machine Learning*. MIT Press.
4. Frenay, B., & Verleysen, M. (2014). "Classification in the Presence of Label Noise." IEEE TNNLS.
5. Nettleton, D., et al. (2010). "A Study of the Effect of Different Types of Noise on the Precision of Supervised Learning Techniques." AIR.
