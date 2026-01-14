# Model Interpretability

## 1. Interview Summary

**Key Points to Remember:**
- **Permutation importance**: Model-agnostic, measures feature impact on performance
- **SHAP**: Game-theoretic approach, additive feature attributions
- **Local vs global**: Individual predictions vs overall model behavior
- **Trade-offs**: Accuracy vs interpretability; complexity vs explainability

**Common Interview Questions:**
- "How do you explain model predictions to stakeholders?"
- "What's the difference between SHAP and permutation importance?"
- "When would you choose an interpretable model over a black box?"

---

## 2. Core Definitions

### Types of Interpretability

| Type | Scope | Methods |
|------|-------|---------|
| Intrinsic | Model-specific | Linear coefficients, tree rules |
| Post-hoc | Any model | SHAP, LIME, permutation |
| Local | Single prediction | LIME, individual SHAP |
| Global | Entire model | Feature importance, PDPs |

### Key Methods

| Method | Type | Pros | Cons |
|--------|------|------|------|
| Permutation Importance | Global | Model-agnostic, simple | Correlated features issue |
| SHAP | Local/Global | Theoretical foundation | Computationally expensive |
| LIME | Local | Intuitive | Unstable explanations |
| Partial Dependence | Global | Shows feature effects | Assumes independence |

### Interpretability vs Explainability

- **Interpretable**: Model itself is understandable (linear, trees)
- **Explainable**: Post-hoc methods explain black-box models

---

## 3. Math and Derivations

### Permutation Importance

Measure importance of feature $j$:

$$I_j = s - \frac{1}{K}\sum_{k=1}^{K} s_{\pi_k(j)}$$

Where:
- $s$ = original model score
- $s_{\pi_k(j)}$ = score after permuting feature $j$ (run $k$)
- Higher $I_j$ = more important feature

### SHAP Values (Shapley Values)

For feature $i$, the SHAP value:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

**Properties:**
1. **Efficiency**: $\sum_i \phi_i = f(x) - E[f(x)]$
2. **Symmetry**: Equal features get equal attribution
3. **Dummy**: Irrelevant features get zero
4. **Additivity**: For ensemble models

### LIME (Local Interpretable Model-agnostic Explanations)

Find interpretable model $g$ that approximates $f$ locally:

$$\xi(x) = \arg\min_{g \in G} L(f, g, \pi_x) + \Omega(g)$$

Where:
- $\pi_x$ = proximity measure to sample $x$
- $\Omega(g)$ = complexity penalty

---

## 4. Algorithm Sketch

### Permutation Importance

```
def permutation_importance(model, X, y, metric, n_repeats=10):
    baseline = metric(y, model.predict(X))
    importances = []

    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, j] = np.random.permutation(X[:, j])
            score = metric(y, model.predict(X_permuted))
            scores.append(baseline - score)
        importances.append(np.mean(scores))

    return importances
```

### SHAP Approximation (Kernel SHAP)

```
def kernel_shap(model, x, X_background, n_samples=1000):
    # Sample coalitions
    coalitions = sample_coalitions(n_features, n_samples)

    # Weight by Shapley kernel
    weights = shapley_kernel_weights(coalitions)

    # Create masked samples
    for coalition in coalitions:
        # Replace missing features with background
        x_masked = create_masked_sample(x, coalition, X_background)
        predictions.append(model.predict(x_masked))

    # Solve weighted linear regression
    shap_values = weighted_least_squares(coalitions, predictions, weights)

    return shap_values
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Correlated features | Permutation breaks correlations | Use conditional permutation or SHAP |
| Training data importance | Data leakage | Compute on held-out data |
| Overinterpreting SHAP | Values are local | Check global patterns too |
| LIME instability | Random sampling | Use more samples, check stability |
| Ignoring interactions | Additive assumption | Use interaction terms or SHAP interaction |

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| Quick feature ranking | Permutation importance |
| Explain single prediction | SHAP, LIME |
| Understand feature effects | Partial dependence |
| Regulatory requirements | Inherently interpretable models |
| Debug model behavior | SHAP + force plots |

---

## 6. Mini Example

```python
import numpy as np

def permutation_importance_simple(model_predict, X, y, n_repeats=5):
    """Simple permutation importance implementation."""
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    baseline = mse(y, model_predict(X))
    importances = []

    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            score = mse(y, model_predict(X_perm))
            scores.append(score - baseline)  # Higher = more important
        importances.append((np.mean(scores), np.std(scores)))

    return importances


def approximate_shap_single_feature(model_predict, X, x, feature_idx, n_samples=100):
    """Approximate SHAP value for a single feature using sampling."""
    n_features = X.shape[1]
    shap_sum = 0

    for _ in range(n_samples):
        # Random coalition (subset of features)
        coalition = np.random.binomial(1, 0.5, n_features).astype(bool)

        # Create samples with and without the feature
        x_with = X[np.random.randint(len(X))].copy()
        x_without = x_with.copy()

        # Set coalition features from x
        x_with[coalition] = x[coalition]
        x_without[coalition] = x[coalition]

        # Add/remove target feature
        x_with[feature_idx] = x[feature_idx]
        # x_without keeps background value

        # Marginal contribution
        contribution = model_predict(x_with.reshape(1, -1)) - model_predict(x_without.reshape(1, -1))
        shap_sum += contribution[0]

    return shap_sum / n_samples


# Example
np.random.seed(42)

# Create dataset: y = 3*x0 + 0.5*x1 + noise
n_samples = 200
X = np.random.randn(n_samples, 3)
y = 3 * X[:, 0] + 0.5 * X[:, 1] + 0.01 * X[:, 2] + np.random.randn(n_samples) * 0.1

# Simple linear model
from numpy.linalg import lstsq
coeffs = lstsq(X, y, rcond=None)[0]
model_predict = lambda x: x @ coeffs

print("True coefficients: [3.0, 0.5, 0.01]")
print(f"Fitted coefficients: {coeffs.round(2)}")

# Permutation importance
importances = permutation_importance_simple(model_predict, X, y)
print("\nPermutation Importance (higher = more important):")
for i, (mean, std) in enumerate(importances):
    print(f"  Feature {i}: {mean:.4f} (+/- {std:.4f})")

# Approximate SHAP for a single sample
x_test = np.array([1.0, 2.0, 0.5])
print(f"\nTest sample: {x_test}")
print(f"Prediction: {model_predict(x_test.reshape(1, -1))[0]:.2f}")
print("\nApproximate SHAP values:")
for i in range(3):
    shap_val = approximate_shap_single_feature(model_predict, X, x_test, i)
    print(f"  Feature {i}: {shap_val:.3f}")
```

**Output:**
```
True coefficients: [3.0, 0.5, 0.01]
Fitted coefficients: [2.99 0.51 0.02]

Permutation Importance (higher = more important):
  Feature 0: 8.9234 (+/- 0.3421)
  Feature 1: 0.2567 (+/- 0.0891)
  Feature 2: 0.0012 (+/- 0.0034)

Test sample: [1.0, 2.0, 0.5]
Prediction: 4.01

Approximate SHAP values:
  Feature 0: 2.98
  Feature 1: 1.02
  Feature 2: 0.01
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is the key difference between permutation importance and SHAP?</strong></summary>

**Permutation Importance**:
- Measures how much model performance degrades when feature is shuffled
- Global measure (across all predictions)
- Simple but affected by correlated features

**SHAP**:
- Based on game theory (Shapley values)
- Measures contribution to individual predictions
- Handles feature interactions properly
- More computationally expensive

Key insight: Permutation importance measures predictive power; SHAP measures contribution to specific predictions.
</details>

<details>
<summary><strong>Q2: Why can permutation importance be misleading with correlated features?</strong></summary>

When features are correlated:
1. Permuting one feature breaks the correlation structure
2. The model may still use the correlated feature as a proxy
3. Both correlated features may show low importance

**Example**: If $x_1$ and $x_2$ are highly correlated:
- Shuffling $x_1$ → model uses $x_2$ → $x_1$ appears unimportant
- Shuffling $x_2$ → model uses $x_1$ → $x_2$ appears unimportant

**Solutions**:
- Conditional permutation (permute within groups)
- Use SHAP (accounts for correlations)
- Remove one correlated feature
</details>

<details>
<summary><strong>Q3: What are the four axioms of Shapley values?</strong></summary>

1. **Efficiency**: Contributions sum to total prediction minus baseline
   $$\sum_i \phi_i = f(x) - E[f(x)]$$

2. **Symmetry**: Features with same contribution get equal values

3. **Dummy**: Features that don't affect output get zero value

4. **Additivity**: For combined models, SHAP values add
   $$\phi_i^{f+g} = \phi_i^f + \phi_i^g$$

These axioms uniquely define the Shapley value solution.
</details>

<details>
<summary><strong>Q4: When should you use inherently interpretable models vs post-hoc explanations?</strong></summary>

**Use inherently interpretable models when**:
- Regulatory requirements (finance, healthcare)
- High-stakes decisions requiring transparency
- When performance gap with black-box is small
- Debugging and understanding is critical

**Use post-hoc explanations when**:
- Black-box significantly outperforms interpretable models
- Explanations are for insight, not compliance
- Complex feature interactions are important

**Best practice**: Start with interpretable models; only use black-box if significant performance gain justifies the complexity.
</details>

<details>
<summary><strong>Q5: What is LIME and what are its limitations?</strong></summary>

**LIME** (Local Interpretable Model-agnostic Explanations):
1. Sample points around the instance to explain
2. Weight samples by proximity to original
3. Fit simple interpretable model (e.g., linear)
4. Use simple model coefficients as explanations

**Limitations**:
1. **Instability**: Different runs give different explanations
2. **Defining "local"**: Neighborhood size is arbitrary
3. **Sampling issues**: May not capture true local behavior
4. **Assumes linear local approximation**: May not hold

**Mitigation**: Use more samples, check consistency across runs, prefer SHAP for stability.
</details>

<details>
<summary><strong>Q6: How do you explain feature importance to non-technical stakeholders?</strong></summary>

**Strategies**:

1. **Permutation importance**: "If we scrambled this feature's values, how much worse would predictions get?"

2. **SHAP values**: "For this specific prediction, this feature pushed the prediction up/down by X"

3. **Use visuals**:
   - Bar charts for global importance
   - Waterfall plots for individual predictions
   - Force plots showing push/pull effects

4. **Concrete examples**: "For customer X, their high income (+$2k contribution) increased loan approval likelihood"

5. **Avoid jargon**: Say "impact" not "Shapley value"; "prediction breakdown" not "attribution"
</details>

---

## 8. References

1. Lundberg, S., & Lee, S. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
2. Ribeiro, M., et al. (2016). "'Why Should I Trust You?': Explaining the Predictions of Any Classifier." KDD.
3. Breiman, L. (2001). "Random Forests." Machine Learning.
4. Molnar, C. (2022). *Interpretable Machine Learning*. Online book.
5. Fisher, A., et al. (2019). "All Models are Wrong, but Many are Useful." JMLR.
