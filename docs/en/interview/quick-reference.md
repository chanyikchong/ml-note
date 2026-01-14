# ML Quick Reference Card

Essential formulas and concepts for quick review before interviews.

---

## Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Regression |
| MAE | $\frac{1}{n}\sum|y - \hat{y}|$ | Robust regression |
| Cross-Entropy | $-\sum y \log \hat{y}$ | Classification |
| Hinge | $\max(0, 1 - y \cdot \hat{y})$ | SVM |

---

## Regularization

| Type | Penalty | Effect |
|------|---------|--------|
| L1 (Lasso) | $\lambda\sum|w_i|$ | Sparse weights |
| L2 (Ridge) | $\lambda\sum w_i^2$ | Small weights |
| Elastic Net | $\alpha L1 + (1-\alpha) L2$ | Combined |

---

## Evaluation Metrics

### Classification

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

$$\text{Accuracy} = \frac{TP + TN}{Total}$$

### ROC vs PR Curves

| Curve | X-axis | Y-axis | Use When |
|-------|--------|--------|----------|
| ROC | FPR | TPR | Balanced data |
| PR | Recall | Precision | Imbalanced data |

---

## Model Formulas

### Linear Regression

**OLS**: $\hat{w} = (X^TX)^{-1}X^Ty$

**Ridge**: $\hat{w} = (X^TX + \lambda I)^{-1}X^Ty$

### Logistic Regression

$$P(y=1|x) = \sigma(w^Tx) = \frac{1}{1 + e^{-w^Tx}}$$

### SVM

**Primal**: $\min_w \frac{1}{2}||w||^2 + C\sum\xi_i$

Subject to: $y_i(w^Tx_i + b) \geq 1 - \xi_i$

### Decision Tree Splits

**Gini**: $1 - \sum_c p_c^2$

**Entropy**: $-\sum_c p_c \log_2 p_c$

**Information Gain**: $IG = H(parent) - \sum \frac{n_i}{n} H(child_i)$

### Gradient Boosting

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

where $h_m$ fits negative gradient of loss.

---

## Neural Networks

### Activation Functions

| Function | Formula | Derivative |
|----------|---------|------------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(1-\sigma)$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2$ |
| ReLU | $\max(0, x)$ | $\mathbb{1}_{x>0}$ |

### Backpropagation

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Chain rule through layers.

### BatchNorm

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

### CNN Output Size

$$O = \frac{W - K + 2P}{S} + 1$$

W=input, K=kernel, P=padding, S=stride

---

## Dimensionality Reduction

### PCA

1. Center data: $X_c = X - \mu$
2. Covariance: $C = \frac{1}{n}X_c^TX_c$
3. Eigendecomposition: $Cv = \lambda v$
4. Project: $Z = X_c V_k$

**Variance explained**: $\frac{\lambda_k}{\sum_i \lambda_i}$

---

## Clustering

### K-Means Objective

$$\min_{\mu} \sum_{i=1}^n \sum_{k=1}^K r_{ik} ||x_i - \mu_k||^2$$

### Silhouette Score

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

a(i) = avg intra-cluster distance
b(i) = avg nearest-cluster distance

---

## Probabilistic Models

### Bayes' Theorem

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

### Naive Bayes

$$P(y|x) \propto P(y) \prod_i P(x_i|y)$$

### Gaussian Distribution

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

---

## Feature Engineering

### Standardization

$$z = \frac{x - \mu}{\sigma}$$

### Min-Max Normalization

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Target Encoding (with smoothing)

$$\text{encoded} = \frac{n \cdot \bar{y}_{cat} + m \cdot \bar{y}_{global}}{n + m}$$

---

## Bias-Variance

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

| Model Complexity | Bias | Variance |
|------------------|------|----------|
| Low (simple) | High | Low |
| High (complex) | Low | High |

---

## Class Imbalance

### SMOTE

$$x_{new} = x_i + \lambda(x_j - x_i), \quad \lambda \in [0,1]$$

### Class Weights

$$w_c = \frac{n_{total}}{n_c \cdot n_{classes}}$$

---

## Optimization

### Gradient Descent Update

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

### Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

---

## Key Numbers to Remember

| Concept | Typical Value |
|---------|---------------|
| Learning rate | 1e-3 to 1e-4 |
| Batch size | 32, 64, 128 |
| Dropout rate | 0.2 to 0.5 |
| L2 regularization | 1e-4 to 1e-2 |
| Train/Val/Test split | 70/15/15 or 80/10/10 |
| CV folds | 5 or 10 |
| VIF threshold | 5-10 |
| PSI threshold | 0.1-0.25 |

---

## Common Pitfalls Checklist

- [ ] Data leakage (fit on train only)
- [ ] Imbalanced classes (use appropriate metrics)
- [ ] Correlated features (check VIF)
- [ ] Missing values (impute properly)
- [ ] Feature scaling (required for many algorithms)
- [ ] Random seeds (for reproducibility)
- [ ] Overfitting (check train vs val gap)
- [ ] Training-serving skew (same preprocessing)
