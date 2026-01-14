# Evaluation Metrics

## 1. Interview Summary

**Key Points to Remember:**
- **Accuracy**: Overall correctness, misleading with class imbalance
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many are found
- **F1**: Harmonic mean of precision and recall
- **ROC-AUC**: Ranking quality, threshold-independent
- **PR-AUC**: Better for imbalanced data

**Common Interview Questions:**
- "When is accuracy not a good metric?"
- "Explain precision vs recall tradeoff"
- "When would you use PR-AUC over ROC-AUC?"

---

## 2. Core Definitions

### Confusion Matrix
```
                Predicted
                Neg    Pos
Actual Neg     TN     FP
       Pos     FN     TP
```

### Classification Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness |
| Precision | $\frac{TP}{TP + FP}$ | Positive predictive value |
| Recall | $\frac{TP}{TP + FN}$ | True positive rate, sensitivity |
| Specificity | $\frac{TN}{TN + FP}$ | True negative rate |
| F1 Score | $\frac{2 \cdot Prec \cdot Rec}{Prec + Rec}$ | Harmonic mean |

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penalizes large errors |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Robust to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Variance explained |

---

## 3. Math and Derivations

### ROC Curve

Plot of True Positive Rate vs False Positive Rate at various thresholds:
- **TPR** (y-axis) = $\frac{TP}{TP + FN}$ = Recall
- **FPR** (x-axis) = $\frac{FP}{FP + TN}$ = 1 - Specificity

**AUC Interpretation:**
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC < 0.5: Worse than random

**Probabilistic Interpretation:**
$$\text{AUC} = P(\text{score}_{positive} > \text{score}_{negative})$$

### PR Curve

Plot of Precision vs Recall:
- More informative for imbalanced data
- Focus on positive class performance

### F-beta Score

Generalized F1 allowing precision-recall weighting:
$$F_\beta = \frac{(1 + \beta^2) \cdot Prec \cdot Rec}{\beta^2 \cdot Prec + Rec}$$

- $\beta = 1$: Equal weight (F1)
- $\beta = 2$: Recall twice as important
- $\beta = 0.5$: Precision twice as important

---

## 4. Algorithm Sketch

### Choosing Metrics

```
If balanced classes:
    → Accuracy is reasonable
    → F1 for single number
    → ROC-AUC for ranking

If imbalanced classes:
    → Avoid accuracy!
    → Use Precision-Recall
    → Use PR-AUC

If cost-sensitive:
    → Custom weighted metrics
    → Consider business cost matrix

For regression:
    If outliers matter: → MSE
    If outliers should be ignored: → MAE
    If relative fit matters: → R²
```

### Threshold Selection

```
1. Train classifier with probability outputs
2. Compute precision and recall at various thresholds
3. Plot precision-recall curve
4. Choose threshold based on:
   - Business requirements
   - Cost of FP vs FN
   - Desired precision/recall tradeoff
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Using accuracy with imbalance | 99% accuracy with 1% positive class | Use F1, PR-AUC |
| Ignoring threshold choice | Default 0.5 may not be optimal | Tune based on requirements |
| ROC-AUC with severe imbalance | ROC can be misleading | Use PR-AUC instead |
| Averaging metrics wrong | Micro vs macro vs weighted | Understand each type |
| Optimizing wrong metric | Training on accuracy, evaluating on AUC | Train and evaluate on same |

### When ROC-AUC Fails

With severe class imbalance (e.g., 99.9% negative):
- ROC can look good due to many true negatives
- PR curve better shows performance on rare positives
- A model predicting "all negative" has 0 FPR, looks good on ROC

---

## 6. Mini Example

```python
import numpy as np

def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_roc_auc(y_true, y_scores):
    """Compute ROC-AUC using trapezoidal rule."""
    # Sort by score descending
    order = np.argsort(y_scores)[::-1]
    y_true = y_true[order]

    # Compute TPR and FPR at each threshold
    tpr = np.cumsum(y_true) / np.sum(y_true)
    fpr = np.cumsum(1 - y_true) / np.sum(1 - y_true)

    # AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc

# Example
np.random.seed(42)
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.7, 0.3, 0.4, 0.85, 0.15])

metrics = compute_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"ROC-AUC: {compute_roc_auc(y_true, y_scores):.3f}")
```

---

## 7. Quiz

<details>
<summary><strong>Q1: When is accuracy a poor metric?</strong></summary>

Accuracy fails with class imbalance. Example:
- Dataset: 99% negative, 1% positive
- Model predicting "all negative": 99% accuracy!
- But it's useless for finding positives

Use precision, recall, F1, or PR-AUC instead.
</details>

<details>
<summary><strong>Q2: Explain the precision-recall tradeoff.</strong></summary>

As threshold increases:
- **Precision increases**: Fewer predictions, but more confident
- **Recall decreases**: Miss more actual positives

Trade-off depends on costs:
- High precision needed: Spam detection (don't lose real emails)
- High recall needed: Medical diagnosis (don't miss disease)
</details>

<details>
<summary><strong>Q3: When should you use PR-AUC over ROC-AUC?</strong></summary>

Use **PR-AUC** when:
- Class imbalance is severe (e.g., <10% positive)
- You care primarily about positive class
- True negatives are less important

Use **ROC-AUC** when:
- Classes are roughly balanced
- Both positive and negative predictions matter
- You want threshold-independent comparison
</details>

<details>
<summary><strong>Q4: What does ROC-AUC measure probabilistically?</strong></summary>

ROC-AUC = probability that a randomly chosen positive example has a higher predicted score than a randomly chosen negative example.

$$\text{AUC} = P(\text{score}_{pos} > \text{score}_{neg})$$

AUC = 1 means perfect ranking; AUC = 0.5 means random ranking.
</details>

<details>
<summary><strong>Q5: Explain micro, macro, and weighted averaging.</strong></summary>

For multi-class classification:
- **Micro**: Compute metrics globally (sum all TP, FP, FN)
- **Macro**: Compute per class, then average (treats all classes equally)
- **Weighted**: Compute per class, average weighted by support

Use macro when all classes equally important regardless of size.
Use weighted when you want to reflect class distribution.
</details>

<details>
<summary><strong>Q6: What is the F-beta score and when to use it?</strong></summary>

$$F_\beta = \frac{(1 + \beta^2) \cdot Prec \cdot Rec}{\beta^2 \cdot Prec + Rec}$$

- F1 ($\beta=1$): Equal importance
- F2 ($\beta=2$): Recall twice as important as precision
- F0.5 ($\beta=0.5$): Precision twice as important

Use when you want to explicitly weight precision vs recall based on business needs.
</details>

---

## 8. References

1. Davis, J., & Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves." ICML.
2. Saito, T., & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." PLOS ONE.
3. Powers, D. (2011). "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation." JMLT.
4. Fawcett, T. (2006). "An Introduction to ROC Analysis." Pattern Recognition Letters.
5. scikit-learn documentation: Metrics and scoring.
