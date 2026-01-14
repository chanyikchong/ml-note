# Naive Bayes

## 1. Interview Summary

**Key Points to Remember:**
- **Bayes' theorem**: $P(y|x) \propto P(x|y)P(y)$
- **Naive assumption**: Features are conditionally independent given class
- **Generative model**: Models $P(x|y)$ instead of $P(y|x)$ directly
- **Fast and simple**: Training is counting, prediction is multiplication
- **Works well with**: Text classification, spam filtering, high-dimensional sparse data

**Common Interview Questions:**
- "What is the naive assumption and when does it fail?"
- "Why does Naive Bayes work well despite the strong assumption?"
- "Compare Gaussian vs Multinomial vs Bernoulli Naive Bayes"

---

## 2. Core Definitions

### Bayes' Theorem

$$P(y|x) = \frac{P(x|y)P(y)}{P(x)}$$

For classification, we need:

$$\hat{y} = \arg\max_y P(y|x) = \arg\max_y P(x|y)P(y)$$

### Naive Independence Assumption
Assume features are conditionally independent given class:

$$P(x_1, x_2, ..., x_d | y) = \prod_{j=1}^{d} P(x_j | y)$$

### Types of Naive Bayes

| Type | Likelihood Model | Use Case |
|------|------------------|----------|
| Gaussian | $P(x_j|y) = \mathcal{N}(\mu_{jy}, \sigma_{jy}^2)$ | Continuous features |
| Multinomial | $P(x_j|y) = \theta_{jy}^{x_j}$ | Word counts, frequencies |
| Bernoulli | $P(x_j|y) = \theta_{jy}^{x_j}(1-\theta_{jy})^{1-x_j}$ | Binary features |

---

## 3. Math and Derivations

### Full Derivation

Given training data $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$:

**Step 1**: Estimate class priors

$$P(y = c) = \frac{\text{count}(y = c)}{n}$$

**Step 2**: Estimate likelihoods (Gaussian example)

$$\mu_{jc} = \frac{1}{n_c}\sum_{i: y^{(i)}=c} x_j^{(i)}$$

$$\sigma_{jc}^2 = \frac{1}{n_c}\sum_{i: y^{(i)}=c} (x_j^{(i)} - \mu_{jc})^2$$

**Step 3**: Prediction

$$\hat{y} = \arg\max_c \left[ \log P(y=c) + \sum_{j=1}^d \log P(x_j | y=c) \right]$$

### Multinomial Naive Bayes (for text)

For document with word counts $x = (x_1, ..., x_V)$:

$$P(x|y=c) \propto \prod_{j=1}^{V} \theta_{jc}^{x_j}$$

Where $\theta_{jc} = P(\text{word } j | \text{class } c)$

**MLE estimate**:

$$\theta_{jc} = \frac{\text{count}(j, c)}{\sum_{k=1}^V \text{count}(k, c)}$$

### Laplace Smoothing

Problem: If word never seen in class, $P(x_j|y) = 0$, making entire product 0.

Solution: Add pseudocounts $\alpha$ (typically 1):

$$\theta_{jc} = \frac{\text{count}(j, c) + \alpha}{\sum_{k=1}^V \text{count}(k, c) + \alpha V}$$

### Log-Space Computation

To avoid underflow from multiplying many small probabilities:

$$\log P(y|x) = \log P(y) + \sum_j \log P(x_j|y) - \log P(x)$$

Since $P(x)$ is constant across classes, we compare:

$$\arg\max_y \left[ \log P(y) + \sum_j \log P(x_j|y) \right]$$

---

## 4. Algorithm Sketch

### Training

```
Input: Training data (X, y)
Output: Model parameters

For each class c:
    # Prior
    prior[c] = count(y == c) / n

    # Likelihoods (Gaussian)
    For each feature j:
        mean[j, c] = mean(X[y == c, j])
        var[j, c] = variance(X[y == c, j])
```

### Prediction

```
Input: New point x
Output: Predicted class

For each class c:
    log_prob[c] = log(prior[c])

    For each feature j:
        log_prob[c] += log(P(x[j] | c))
        # Gaussian: log(N(x[j]; mean[j,c], var[j,c]))

Return argmax(log_prob)
```

### Text Classification Pipeline

```
1. Preprocess text: tokenize, lowercase, remove stopwords
2. Build vocabulary: map words to indices
3. Convert documents to count vectors
4. Train Multinomial NB with Laplace smoothing
5. Predict: compute log-probabilities for each class
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Zero probabilities | Unseen feature values | Use Laplace smoothing |
| Numerical underflow | Product of many small numbers | Use log-probabilities |
| Wrong NB variant | Using Gaussian for counts | Match variant to data type |
| Correlated features | Violates independence assumption | Still often works; try if correlated |
| Unbalanced classes | Prior dominates prediction | Consider balanced priors |

### When Naive Assumption Fails

The assumption fails when features are correlated:
- "hot" and "dog" in "hot dog" (word pairs)
- Pixel neighbors in images
- Time series with autocorrelation

**Why it often still works:**
- Classification only needs correct ranking, not calibrated probabilities
- Errors in probability estimates may cancel out
- Even wrong model can have correct decision boundary

---

## 6. Mini Example

```python
import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        # Compute priors and likelihoods
        self.priors = {}
        self.means = {}
        self.vars = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0) + 1e-9  # Add small value for stability

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        log_probs = {}
        for c in self.classes:
            # Log prior
            log_prob = np.log(self.priors[c])
            # Log likelihood (Gaussian)
            log_prob += np.sum(-0.5 * np.log(2 * np.pi * self.vars[c])
                               - 0.5 * (x - self.means[c])**2 / self.vars[c])
            log_probs[c] = log_prob
        return max(log_probs, key=log_probs.get)

# Example
np.random.seed(42)

# Generate data: two Gaussian classes
X0 = np.random.randn(50, 2) + np.array([0, 0])
X1 = np.random.randn(50, 2) + np.array([3, 3])
X = np.vstack([X0, X1])
y = np.array([0] * 50 + [1] * 50)

# Train and predict
nb = GaussianNB()
nb.fit(X, y)

# Test
X_test = np.array([[0, 0], [3, 3], [1.5, 1.5]])
predictions = nb.predict(X_test)
print(f"Test points: {X_test.tolist()}")
print(f"Predictions: {predictions}")

# Training accuracy
train_pred = nb.predict(X)
accuracy = np.mean(train_pred == y)
print(f"Training accuracy: {accuracy:.3f}")
```

**Output:**
```
Test points: [[0, 0], [3, 3], [1.5, 1.5]]
Predictions: [0 1 1]
Training accuracy: 0.970
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is the "naive" assumption in Naive Bayes?</strong></summary>

The naive assumption is that all features are **conditionally independent** given the class label:

$$P(x_1, x_2, ..., x_d | y) = \prod_{j=1}^{d} P(x_j | y)$$

This allows us to estimate each $P(x_j|y)$ separately, making training simple and fast. Without this assumption, we'd need to estimate the full joint distribution, requiring exponentially more data.
</details>

<details>
<summary><strong>Q2: Why does Naive Bayes often work well despite the independence assumption being violated?</strong></summary>

Several reasons:
1. **Classification only needs ranking**: We don't need accurate probabilities, just the correct $\arg\max$
2. **Error cancellation**: Overestimates and underestimates may balance out
3. **Decision boundary**: Even with wrong probabilities, the decision boundary can be correct
4. **Regularization effect**: The strong assumption acts as a regularizer, reducing variance
5. **High-dimensional success**: In high dimensions, the assumption becomes less harmful
</details>

<details>
<summary><strong>Q3: When would you use Multinomial vs Gaussian vs Bernoulli Naive Bayes?</strong></summary>

- **Multinomial NB**: Word counts, document classification, any count data
- **Gaussian NB**: Continuous real-valued features, assuming normal distribution
- **Bernoulli NB**: Binary features (presence/absence), binary text classification

Example:
- Email spam (word counts) → Multinomial
- Iris flower classification (continuous) → Gaussian
- Binary features (has_link, has_attachment) → Bernoulli
</details>

<details>
<summary><strong>Q4: What is Laplace smoothing and why is it needed?</strong></summary>

**Problem**: If a feature value never appears with a class in training, $P(x_j|y) = 0$, making the entire product 0.

**Solution**: Add pseudocounts $\alpha$ (typically 1):

$$\theta_{jc} = \frac{\text{count}(j, c) + \alpha}{\text{total count}(c) + \alpha \cdot V}$$

This ensures no probability is ever exactly 0. Also called "additive smoothing" or "add-one smoothing."
</details>

<details>
<summary><strong>Q5: Is Naive Bayes a discriminative or generative model?</strong></summary>

Naive Bayes is a **generative** model:
- Models the joint distribution $P(x, y) = P(x|y)P(y)$
- Learns how data is "generated" for each class
- Can generate new samples (sample $y$, then sample $x|y$)

Contrast with **discriminative** models (logistic regression, SVM):
- Model $P(y|x)$ directly
- Don't model how features are distributed
- Often better for classification when assumption is wrong
</details>

<details>
<summary><strong>Q6: How do you handle continuous features in Naive Bayes?</strong></summary>

Options:
1. **Gaussian NB**: Assume each feature follows normal distribution per class
2. **Discretization**: Bin continuous values into categories
3. **Kernel density estimation**: Non-parametric density estimation

Gaussian NB is most common:

$$P(x_j|y=c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)$$
</details>

---

## 8. References

1. Mitchell, T. (1997). *Machine Learning*. Chapter 6: Bayesian Learning.
2. Manning, C., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Chapter 13.
3. McCallum, A., & Nigam, K. (1998). "A Comparison of Event Models for Naive Bayes Text Classification." AAAI Workshop.
4. Zhang, H. (2004). "The Optimality of Naive Bayes." FLAIRS.
5. Ng, A., & Jordan, M. (2002). "On Discriminative vs. Generative Classifiers." NIPS.
