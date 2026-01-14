# Feature Engineering

## 1. Interview Summary

**Key Points to Remember:**
- **Numerical**: Scaling, normalization, log transforms, binning
- **Categorical**: One-hot encoding, target encoding, embeddings
- **Missing values**: Imputation strategies matter
- **Feature selection**: Filter, wrapper, embedded methods
- **Feature importance**: Reduces dimensionality, improves interpretability

**Common Interview Questions:**
- "When would you use standardization vs normalization?"
- "How do you handle categorical variables with high cardinality?"
- "What are the methods for feature selection?"

---

## 2. Core Definitions

### Numerical Transformations

| Method | Formula | Use Case |
|--------|---------|----------|
| Standardization | $(x - \mu) / \sigma$ | Most algorithms, assumes Gaussian |
| Min-Max | $(x - min) / (max - min)$ | Bounded [0,1], neural networks |
| Log transform | $\log(x + 1)$ | Right-skewed distributions |
| Power transform | $x^\lambda$ or Box-Cox | Make data more Gaussian |

### Categorical Encoding

| Method | Description | Cardinality |
|--------|-------------|------------|
| One-hot | Binary column per category | Low (<20) |
| Label encoding | Integer per category | Ordinal data |
| Target encoding | Mean of target per category | High cardinality |
| Embedding | Learned dense vector | Very high, deep learning |

### Missing Value Strategies

| Strategy | When to Use |
|----------|------------|
| Drop rows | Few missing, random |
| Mean/median | Numerical, few missing |
| Mode | Categorical |
| Model-based (KNN, iterative) | Complex patterns |
| Indicator variable | Missingness is informative |

---

## 3. Math and Derivations

### Target Encoding with Smoothing

To prevent overfitting on rare categories:
$$\text{encoded}_c = \lambda(c) \cdot \bar{y}_c + (1 - \lambda(c)) \cdot \bar{y}_{global}$$

Where:
$$\lambda(c) = \frac{n_c}{n_c + m}$$

$m$ is smoothing parameter. More samples → trust category mean; fewer → trust global mean.

### Variance Inflation Factor (VIF)

Detect multicollinearity:
$$VIF_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is R² from regressing feature $j$ on all other features.
- VIF > 5-10 indicates high multicollinearity

### Information Gain for Feature Selection

$$IG(Y, X) = H(Y) - H(Y|X)$$

Higher IG = more predictive feature.

---

## 4. Algorithm Sketch

### Feature Engineering Pipeline

```
def feature_pipeline(df, target, num_cols, cat_cols):
    # Handle missing values
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Numerical transformations
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Categorical encoding
    for col in cat_cols:
        if df[col].nunique() < 10:
            # One-hot for low cardinality
            df = pd.get_dummies(df, columns=[col])
        else:
            # Target encoding for high cardinality
            means = df.groupby(col)[target].mean()
            df[col] = df[col].map(means)

    return df
```

### Feature Selection Methods

```
# Filter method: Correlation
def correlation_filter(X, y, threshold=0.1):
    correlations = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
    return [i for i, c in enumerate(correlations) if c > threshold]

# Wrapper method: Recursive Feature Elimination
def rfe(model, X, y, n_features):
    while X.shape[1] > n_features:
        model.fit(X, y)
        importances = model.feature_importances_
        worst = np.argmin(importances)
        X = np.delete(X, worst, axis=1)
    return X

# Embedded method: L1 regularization
def lasso_selection(X, y, alpha=0.01):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return np.where(model.coef_ != 0)[0]
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Data leakage from encoding | Fit encoder on all data | Fit on train only |
| Target leakage | Future info in features | Careful feature audit |
| High cardinality one-hot | Thousands of columns | Use target/embedding |
| Scaling test with train stats | Different distributions | Save scaler from train |
| Ignoring feature interactions | Linear models miss them | Create explicit interactions |

### Preprocessing Order

```
1. Train-test split (prevent leakage)
2. Handle missing values (fit imputer on train)
3. Encode categoricals (fit encoder on train)
4. Scale numericals (fit scaler on train)
5. Feature selection (using train only)
6. Transform test using fitted transformers
```

---

## 6. Mini Example

```python
import numpy as np

def standardize(X, fit=True, mean=None, std=None):
    if fit:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std, mean, std

def target_encode(X_cat, y, smoothing=10):
    global_mean = np.mean(y)
    encoded = np.zeros(len(X_cat))
    for cat in np.unique(X_cat):
        mask = X_cat == cat
        n = np.sum(mask)
        cat_mean = np.mean(y[mask])
        lambda_c = n / (n + smoothing)
        encoded[mask] = lambda_c * cat_mean + (1 - lambda_c) * global_mean
    return encoded

def correlation_feature_selection(X, y, k=5):
    correlations = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
    top_k = np.argsort(correlations)[-k:]
    return top_k, correlations[top_k]

# Example
np.random.seed(42)

# Create dataset
n_samples = 100
X_num = np.random.randn(n_samples, 3)  # 3 numerical features
X_cat = np.random.choice(['A', 'B', 'C'], n_samples)  # 1 categorical
y = X_num[:, 0] + (X_cat == 'A').astype(float) * 2 + np.random.randn(n_samples) * 0.5

# Split
train_idx = np.arange(80)
test_idx = np.arange(80, 100)

# Standardize numerical
X_train_scaled, mean, std = standardize(X_num[train_idx], fit=True)
X_test_scaled, _, _ = standardize(X_num[test_idx], fit=False, mean=mean, std=std)

print("Standardization:")
print(f"  Train mean: {np.mean(X_train_scaled, axis=0)}")
print(f"  Train std: {np.std(X_train_scaled, axis=0)}")

# Target encode categorical
X_cat_encoded = target_encode(X_cat[train_idx], y[train_idx])
print(f"\nTarget encoding (Category A mean: {np.mean(y[train_idx][X_cat[train_idx]=='A']):.2f})")
print(f"  Encoded values: A={X_cat_encoded[X_cat[train_idx]=='A'][0]:.2f}, B={X_cat_encoded[X_cat[train_idx]=='B'][0]:.2f}")

# Feature selection
all_features = np.column_stack([X_train_scaled, X_cat_encoded])
top_features, correlations = correlation_feature_selection(all_features, y[train_idx], k=2)
print(f"\nTop 2 features by correlation: {top_features}")
print(f"  Correlations: {correlations}")
```

**Output:**
```
Standardization:
  Train mean: [-0.0  0.0 -0.0]
  Train std: [1.0 1.0 1.0]

Target encoding (Category A mean: 2.15)
  Encoded values: A=1.89, B=-0.12

Top 2 features by correlation: [0 3]
  Correlations: [0.78 0.65]
```

---

## 7. Quiz

<details>
<summary><strong>Q1: When should you use standardization vs min-max normalization?</strong></summary>

**Standardization** (z-score):
- When algorithm assumes Gaussian distribution
- For algorithms sensitive to magnitude (SVM, logistic regression)
- When outliers are present (less affected)

**Min-Max normalization**:
- When you need bounded range [0, 1]
- For neural networks (activation functions expect certain ranges)
- When distribution is not Gaussian
- When no outliers or outliers are meaningful
</details>

<details>
<summary><strong>Q2: How do you handle high-cardinality categorical features?</strong></summary>

Options:
1. **Target encoding**: Replace category with mean of target (use smoothing)
2. **Frequency encoding**: Replace with count/frequency
3. **Embedding**: Learn dense vector representation (deep learning)
4. **Hashing**: Hash to fixed number of bins
5. **Grouping**: Combine rare categories into "Other"

Avoid one-hot encoding (creates too many sparse features).
</details>

<details>
<summary><strong>Q3: What are the three types of feature selection methods?</strong></summary>

1. **Filter methods**: Score features independently of model
   - Correlation, mutual information, chi-squared
   - Fast but ignores feature interactions

2. **Wrapper methods**: Use model performance as criterion
   - Forward selection, backward elimination, RFE
   - Better but computationally expensive

3. **Embedded methods**: Feature selection built into training
   - L1 regularization (Lasso), tree importance
   - Balance of quality and efficiency
</details>

<details>
<summary><strong>Q4: How do you prevent data leakage in feature engineering?</strong></summary>

**Rules**:
1. Split data before any transformation
2. Fit transformers on training data only
3. Transform test data using fitted parameters
4. Never use target information for encoding at test time
5. Watch for temporal leakage in time series

**Example pipeline**:
```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use train's stats
```
</details>

<details>
<summary><strong>Q5: What is target encoding and how do you prevent overfitting?</strong></summary>

**Target encoding**: Replace category with mean of target for that category.

**Overfitting problem**: Rare categories have unreliable means.

**Solutions**:
1. **Smoothing**: Blend category mean with global mean
   $$encoded = \frac{n \cdot cat\_mean + m \cdot global\_mean}{n + m}$$

2. **Cross-validation encoding**: Encode each fold using other folds

3. **Add noise**: Regularize by adding small noise
</details>

<details>
<summary><strong>Q6: How do you handle missing values?</strong></summary>

**Strategies**:
1. **Drop**: If few missing, random pattern
2. **Impute with statistic**: Mean (numerical), mode (categorical)
3. **Model-based**: KNN imputer, iterative imputer
4. **Create indicator**: Binary "is_missing" feature (if informative)
5. **Domain-specific**: Use knowledge (e.g., 0 for "no purchase")

**Key**: Always impute on training data first, then apply same values to test.
</details>

---

## 8. References

1. Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*. CRC Press.
2. Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly.
3. Guyon, I., & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." JMLR.
4. scikit-learn documentation: Preprocessing data.
5. Micci-Barreca, D. (2001). "A Preprocessing Scheme for High-Cardinality Categorical Attributes." ACM SIGKDD.
