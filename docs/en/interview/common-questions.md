# Common ML Interview Questions

A collection of frequently asked machine learning interview questions with concise answers.

---

## Fundamentals

### Q: What is the bias-variance tradeoff?

**A**: The bias-variance tradeoff describes the tension between model simplicity and complexity:
- **High bias**: Model is too simple, underfits (misses patterns)
- **High variance**: Model is too complex, overfits (memorizes noise)

**Total error** = Bias² + Variance + Irreducible noise

Simple models → high bias, low variance
Complex models → low bias, high variance

**Solution**: Find the sweet spot via cross-validation, regularization, or ensemble methods.

---

### Q: Explain overfitting and how to prevent it.

**A**: Overfitting occurs when a model learns noise in training data instead of general patterns.

**Signs**: Large gap between train and validation performance.

**Prevention**:
1. More training data
2. Regularization (L1, L2)
3. Early stopping
4. Dropout (neural networks)
5. Cross-validation
6. Simpler model architecture
7. Data augmentation

---

### Q: What's the difference between L1 and L2 regularization?

**A**:

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|-----------|
| Penalty | $\lambda\sum|w_i|$ | $\lambda\sum w_i^2$ |
| Effect | Sparse weights (feature selection) | Small weights (shrinkage) |
| Solution | Non-differentiable at 0 | Closed-form |
| Use when | Feature selection needed | All features relevant |

---

### Q: When would you use cross-validation vs. a held-out test set?

**A**:
- **Cross-validation**: When data is limited, need robust estimate
- **Held-out test**: When data is abundant, final unbiased evaluation

**Important**: Test set should only be used once for final evaluation. Never tune hyperparameters on test set.

---

## Models

### Q: Explain how a decision tree splits.

**A**: Decision trees recursively split data to maximize purity:

1. Calculate impurity (Gini or entropy) for current node
2. For each feature and threshold, compute information gain
3. Select split that maximizes information gain
4. Repeat until stopping criteria (max depth, min samples)

**Gini**: $1 - \sum p_c^2$
**Entropy**: $-\sum p_c \log p_c$

---

### Q: Why use Random Forest over a single decision tree?

**A**:

| Aspect | Single Tree | Random Forest |
|--------|------------|---------------|
| Variance | High | Low (ensemble averaging) |
| Interpretability | High | Lower |
| Overfitting | Prone | Resistant |
| Training | Fast | Parallelizable |

Random Forest reduces variance through:
1. **Bagging**: Train on bootstrap samples
2. **Feature randomness**: Consider subset of features at each split

---

### Q: Explain gradient boosting conceptually.

**A**: Gradient boosting builds models sequentially:

1. Start with simple prediction (e.g., mean)
2. Compute residuals (errors)
3. Train new model to predict residuals
4. Add new model to ensemble (with learning rate)
5. Repeat

**Key formula**: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$

Each new tree corrects previous errors. Ensemble grows additively.

---

### Q: When would you use SVM vs Logistic Regression?

**A**:

| Aspect | Logistic Regression | SVM |
|--------|-------------------|-----|
| Output | Probabilities | Class only (unless calibrated) |
| Decision boundary | Considers all points | Only support vectors |
| Non-linearity | Feature engineering | Kernel trick |
| High dimensions | May overfit | Works well (max-margin) |
| Training speed | Fast | Slower (especially kernel) |

**Rule of thumb**: Start with logistic regression; use SVM for high-dimensional, small datasets.

---

### Q: How does PCA work?

**A**: PCA finds orthogonal directions of maximum variance:

1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors and eigenvalues
4. Sort by eigenvalue (descending)
5. Project data onto top k eigenvectors

**Key insight**: First principal component = direction of maximum variance.

---

## Deep Learning

### Q: Why use activation functions?

**A**: Activation functions introduce non-linearity. Without them, multiple layers collapse to a single linear transformation.

**Common activations**:
- **ReLU**: $\max(0, x)$ — fast, avoids vanishing gradient
- **Sigmoid**: $(1 + e^{-x})^{-1}$ — outputs probabilities, vanishing gradient problem
- **Tanh**: Centered output, still has gradient issues

---

### Q: Explain vanishing/exploding gradients.

**A**:
- **Vanishing**: Gradients shrink exponentially through layers (sigmoid, deep networks)
- **Exploding**: Gradients grow exponentially (unstable training)

**Solutions**:
1. ReLU activations (no saturation)
2. Batch/Layer normalization
3. Skip connections (ResNet)
4. Proper initialization (He, Xavier)
5. Gradient clipping (for exploding)

---

### Q: What does BatchNorm do?

**A**: BatchNorm normalizes layer inputs across the batch:

$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

**Benefits**:
1. Faster training (higher learning rates)
2. Regularization effect
3. Reduces internal covariate shift

**Note**: Behavior differs at train vs test time.

---

### Q: When to use CNN vs RNN vs Transformer?

**A**:

| Architecture | Use Case | Key Feature |
|--------------|----------|-------------|
| CNN | Images, grid data | Local patterns, translation invariant |
| RNN/LSTM | Sequences (short) | Memory, sequential processing |
| Transformer | Text, long sequences | Attention, parallelizable |

Modern trend: Transformers increasingly used for images (ViT) and audio too.

---

## Practical ML

### Q: How do you handle imbalanced datasets?

**A**:

1. **Resampling**: Oversample minority (SMOTE) or undersample majority
2. **Class weights**: Higher penalty for minority errors
3. **Threshold tuning**: Adjust decision boundary post-training
4. **Different metrics**: Use PR-AUC, F1 instead of accuracy
5. **Anomaly detection**: For extreme imbalance

**Key**: Always resample after train-test split (only on training data).

---

### Q: When would you use precision vs recall?

**A**:
- **Precision** (of positives, how many correct): Use when false positives are costly (spam filter, medical screening second stage)
- **Recall** (of actual positives, how many found): Use when false negatives are costly (disease detection, fraud)

**F1 score** balances both: $F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$

---

### Q: What is data leakage?

**A**: Data leakage occurs when information from outside the training set influences the model:

**Types**:
1. **Target leakage**: Feature derived from target
2. **Train-test contamination**: Test data seen during training

**Prevention**:
1. Split data first
2. Fit transformers on train only
3. No future information in features
4. Careful with time series

---

### Q: How do you explain model predictions?

**A**:

**Methods**:
1. **Feature importance**: Which features matter most
2. **SHAP values**: Contribution of each feature to prediction
3. **Partial dependence plots**: How features affect output
4. **LIME**: Local linear approximation

**For stakeholders**: Use concrete examples; explain in business terms.

---

### Q: How do you handle missing values?

**A**:

| Method | When to Use |
|--------|------------|
| Drop rows | Few missing, random pattern |
| Mean/Median | Numerical, simple |
| Mode | Categorical |
| Model-based (KNN) | Complex patterns |
| Indicator variable | Missingness is informative |

**Always**: Fit imputer on train, transform test with same values.

---

## System Design

### Q: How would you design a recommendation system?

**A**:

**Approaches**:
1. **Collaborative filtering**: User-item interactions (matrix factorization)
2. **Content-based**: Item features match user preferences
3. **Hybrid**: Combine both

**Considerations**:
- Cold start problem (new users/items)
- Scalability (millions of users/items)
- Real-time vs batch
- Evaluation: Click-through rate, engagement

---

### Q: How do you monitor models in production?

**A**:

**Monitor for**:
1. **Data drift**: Input distribution changes (PSI, KL divergence)
2. **Concept drift**: Relationship between X and Y changes
3. **Performance degradation**: When labels are available
4. **System health**: Latency, errors, throughput

**Actions**: Alerts, automatic retraining triggers, A/B testing for updates.

---

### Q: What's the difference between batch and online learning?

**A**:

| Aspect | Batch | Online |
|--------|-------|--------|
| Training | All data at once | One sample at a time |
| Updates | Periodic retraining | Continuous |
| Memory | Needs all data | Minimal |
| Use case | Static problems | Changing patterns |

**Online learning**: Useful for streaming data, large datasets that don't fit in memory.

---

## Behavioral

### Q: Describe a challenging ML project and how you solved it.

**Framework**:
1. **Situation**: What was the problem?
2. **Task**: What was your role?
3. **Action**: What approaches did you try?
4. **Result**: What was the outcome? Metrics improved?

**Tips**: Be specific about technical details, show iterative problem-solving.

---

### Q: How do you decide what model to use?

**A**:

1. **Start simple**: Linear/logistic regression as baseline
2. **Consider data**: Size, type (tabular/image/text), quality
3. **Consider requirements**: Interpretability, speed, accuracy
4. **Iterate**: Try multiple approaches, compare fairly

**Common choices**:
- Tabular: Start with gradient boosting (XGBoost, LightGBM)
- Text/Image: Deep learning (Transformer, CNN)
- Small data: Simpler models, transfer learning
