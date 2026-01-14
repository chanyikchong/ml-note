# Data Splits & Validation

## 1. Interview Summary

**Key Points to Remember:**
- **Training set**: Used to fit model parameters
- **Validation set**: Used to tune hyperparameters and model selection
- **Test set**: Used only once for final unbiased evaluation
- **Data leakage**: When information from test/validation leaks into training
- **Cross-validation**: K-fold for robust hyperparameter tuning
- Know when CV is invalid (time series, grouped data)

**Common Interview Questions:**
- "Why do we need separate train/val/test sets?"
- "What is data leakage? Give examples."
- "When should you NOT use k-fold cross-validation?"

---

## 2. Core Definitions

### Train/Validation/Test Split
- **Training Set (~60-80%)**: Data used to learn model parameters
- **Validation Set (~10-20%)**: Data used for hyperparameter tuning, early stopping, model selection
- **Test Set (~10-20%)**: Held-out data for final evaluation; never used during training

### Data Leakage
Information from outside the training dataset that provides unintended predictive signal.

**Types:**
1. **Target leakage**: Features containing information about the target
2. **Train-test leakage**: Test information leaking into training
3. **Temporal leakage**: Using future data to predict past

### Cross-Validation
Technique to estimate model performance by partitioning data into multiple train/validation splits.

**K-Fold CV:**
- Split data into K equal parts (folds)
- Train on K-1 folds, validate on remaining fold
- Repeat K times, average results

---

## 3. Math and Derivations

### Generalization Error Decomposition

True generalization error cannot be computed directly. We estimate it:

$$\text{Test Error} \approx \mathbb{E}[\mathcal{L}(f(x), y)]$$

**Bias-Variance Decomposition** (for squared loss):

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

### Cross-Validation Estimator

K-fold CV estimate of generalization error:

$$\hat{R}_{CV} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{|D_k|} \sum_{(x,y) \in D_k} \mathcal{L}(f^{(-k)}(x), y)$$

where $f^{(-k)}$ is trained on all data except fold $k$.

### Standard Error of CV Estimate

$$SE = \sqrt{\frac{1}{K(K-1)} \sum_{k=1}^{K} (R_k - \hat{R}_{CV})^2}$$

**One-Standard-Error Rule**: Choose simplest model within one SE of the best.

---

## 4. Algorithm Sketch

### Standard Train/Val/Test Split
```
1. Shuffle data (if i.i.d. assumption holds)
2. Split: 60% train, 20% validation, 20% test
3. Train model on training set
4. Tune hyperparameters using validation set
5. Select best model based on validation performance
6. Evaluate ONCE on test set
7. Report test performance as final result
```

### K-Fold Cross-Validation
```
1. Shuffle data randomly
2. Split into K equal folds
3. For k = 1 to K:
   a. Use fold k as validation
   b. Use remaining K-1 folds as training
   c. Train model and record validation score
4. Average K validation scores
5. (Optional) Retrain on all data with best hyperparameters
```

### Stratified K-Fold (for Classification)
```
1. Ensure each fold maintains class proportions
2. For each class:
   a. Distribute samples across folds proportionally
3. Proceed with standard K-fold procedure
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Using test set for tuning | Desperation to improve scores | Strict discipline; never touch test until final eval |
| Feature scaling before split | Fitting scaler on all data | Fit scaler only on training data |
| Leaking future information | Not respecting time order | Use temporal splits for time series |
| Random split on grouped data | Ignoring group structure | Use GroupKFold or group-aware splits |
| Too few CV folds | Computational constraints | Use at least K=5; K=10 is common |
| Overfitting to validation | Excessive hyperparameter search | Use nested CV; limit search iterations |

### Data Leakage Examples

**Example 1: Target Leakage**
```python
# BAD: Hospital_discharge_date leaks patient outcome
features = ['age', 'admission_date', 'discharge_date']  # discharge implies survival
```

**Example 2: Preprocessing Leakage**
```python
# BAD: Scaling on full dataset
scaler.fit(X)  # Includes test data!
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GOOD: Scale only on training data
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 6. Mini Example

### Python Example: Proper Validation Setup

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Proper train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline (scaling happens inside CV)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# 5-fold cross-validation on training data
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Final evaluation on test set
pipeline.fit(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {test_score:.3f}")
```

**Output:**
```
CV Accuracy: 0.856 (+/- 0.038)
Test Accuracy: 0.870
```

### When NOT to Use Standard K-Fold

```python
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

# Time Series: Use TimeSeriesSplit
ts_cv = TimeSeriesSplit(n_splits=5)
# Ensures training only on past data

# Grouped Data: Use GroupKFold
groups = [0, 0, 1, 1, 2, 2, 3, 3]  # Patient IDs
group_cv = GroupKFold(n_splits=4)
# Ensures same patient isn't in both train and val
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Why can't we use the test set for hyperparameter tuning?</strong></summary>

Using the test set for tuning would cause the model selection process to overfit to the test data. The test set is meant to provide an unbiased estimate of generalization performance. If we tune on it, we're effectively "training" our model selection on that data, making the final test score an optimistic estimate.
</details>

<details>
<summary><strong>Q2: What is data leakage and give two examples?</strong></summary>

Data leakage occurs when information from outside the training set influences model training, leading to overly optimistic performance estimates.

**Examples:**
1. **Preprocessing leakage**: Fitting a scaler or encoder on the entire dataset before splitting
2. **Target leakage**: Including features that contain target information (e.g., "treatment_successful" when predicting patient outcomes)
3. **Temporal leakage**: Using future data to predict past events
</details>

<details>
<summary><strong>Q3: When is k-fold cross-validation invalid?</strong></summary>

K-fold CV is invalid when the i.i.d. assumption is violated:
1. **Time series data**: Future data would leak into training; use TimeSeriesSplit instead
2. **Grouped/clustered data**: Observations from the same group in both train/val; use GroupKFold
3. **Spatial data**: Nearby points may be correlated; use spatial blocking
4. **When samples are not independent**: Any dependency structure requires special handling
</details>

<details>
<summary><strong>Q4: What is the one-standard-error rule?</strong></summary>

The one-standard-error rule suggests selecting the simplest model whose performance is within one standard error of the best-performing model. This promotes parsimony and helps avoid overfitting to the validation set while accepting a small decrease in estimated performance for a simpler model.
</details>

<details>
<summary><strong>Q5: How should you handle preprocessing in cross-validation?</strong></summary>

Preprocessing (scaling, encoding, imputation) should be fit only on the training fold and applied to the validation fold. This is achieved by:
1. Using sklearn `Pipeline` to encapsulate preprocessing and model
2. Fitting preprocessors inside each CV fold
3. Never fitting on the full dataset before splitting

This prevents leaking information from validation data into training.
</details>

<details>
<summary><strong>Q6: What is nested cross-validation and when is it needed?</strong></summary>

Nested CV uses an outer loop for performance estimation and an inner loop for hyperparameter tuning. The outer loop provides unbiased generalization estimates while the inner loop tunes hyperparameters.

It's needed when:
- Reporting final performance estimates with hyperparameter tuning
- You want to avoid optimistic bias from using the same data for tuning and evaluation
- Dataset is small and a separate test set would waste data
</details>

---

## 8. References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Kohavi, R. (1995). "A Study of Cross-Validation and Bootstrap for Accuracy Estimation." IJCAI.
3. Kaufman, S., et al. (2012). "Leakage in Data Mining: Formulation, Detection, and Avoidance." TKDD.
4. Varma, S., & Simon, R. (2006). "Bias in Error Estimation When Using Cross-Validation for Model Selection." BMC Bioinformatics.
5. Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." JMLR.
