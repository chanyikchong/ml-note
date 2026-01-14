# Linear Regression

## 1. Interview Summary

**Key Points to Remember:**
- **OLS**: Ordinary Least Squares - closed-form solution
- **Ridge**: L2 regularization - handles multicollinearity
- **Lasso**: L1 regularization - feature selection
- Key assumptions: linearity, independence, homoscedasticity, normality of errors
- Know the closed-form solution and when to use gradient descent

**Common Interview Questions:**
- "What are the assumptions of linear regression?"
- "Derive the closed-form solution for OLS"
- "When would you use Ridge vs Lasso?"

---

## 2. Core Definitions

### Model

$$y = X\beta + \epsilon$$

where:
- $y \in \mathbb{R}^n$: target vector
- $X \in \mathbb{R}^{n \times p}$: design matrix (with intercept column)
- $\beta \in \mathbb{R}^p$: coefficients
- $\epsilon \in \mathbb{R}^n$: error terms

### OLS Objective

$$\min_\beta \|y - X\beta\|_2^2 = \min_\beta \sum_{i=1}^n (y_i - x_i^T\beta)^2$$

### Ridge Regression

$$\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2$$

### Lasso Regression

$$\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_1$$

---

## 3. Math and Derivations

### OLS Closed-Form Solution

**Derivation:**

$$\mathcal{L}(\beta) = (y - X\beta)^T(y - X\beta)$$

Expand:

$$\mathcal{L}(\beta) = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$$

Take gradient and set to zero:

$$\nabla_\beta \mathcal{L} = -2X^Ty + 2X^TX\beta = 0$$

Solve for $\beta$:

$$\boxed{\hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty}$$

**Requires**: $X^TX$ invertible (full column rank)

### Ridge Closed-Form Solution

$$\mathcal{L}(\beta) = \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2$$

$$\nabla_\beta \mathcal{L} = -2X^Ty + 2X^TX\beta + 2\lambda\beta = 0$$

$$\boxed{\hat{\beta}_{Ridge} = (X^TX + \lambda I)^{-1}X^Ty}$$

**Advantage**: Always invertible for $\lambda > 0$

### Gauss-Markov Theorem

Under assumptions:
1. $\mathbb{E}[\epsilon] = 0$
2. $\text{Var}(\epsilon) = \sigma^2 I$ (homoscedasticity, uncorrelated)
3. $X$ is fixed/non-stochastic

**Result**: OLS is BLUE (Best Linear Unbiased Estimator)
- Minimum variance among all linear unbiased estimators

### Assumptions for Inference

For hypothesis testing and confidence intervals:
1. **Linearity**: $y = X\beta + \epsilon$
2. **Independence**: Errors are independent
3. **Homoscedasticity**: $\text{Var}(\epsilon_i) = \sigma^2$ constant
4. **Normality**: $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

---

## 4. Algorithm Sketch

### OLS with Normal Equations
```
Input: X (n×p), y (n×1)
1. Compute X^T X (p×p matrix)
2. Compute X^T y (p×1 vector)
3. Solve (X^T X) β = X^T y
   - Use Cholesky decomposition if X^T X is positive definite
   - Or use QR decomposition for numerical stability
4. Return β
```

### Gradient Descent for Linear Regression
```
Input: X, y, learning_rate η, iterations T
Initialize: β = 0 or random

For t = 1 to T:
    predictions = X @ β
    errors = predictions - y
    gradient = (2/n) * X.T @ errors
    β = β - η * gradient

Return β
```

### Choosing Between OLS, Ridge, Lasso
```
If p < n and X has full rank:
    → OLS (closed-form)

If multicollinearity or p ≈ n:
    → Ridge (shrinks correlated features together)

If you want feature selection:
    → Lasso (sets some coefficients to zero)

If high-dimensional (p >> n):
    → Lasso or Elastic Net
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Not checking assumptions | Just applying OLS blindly | Plot residuals, check diagnostics |
| Ignoring multicollinearity | Correlated predictors | Check VIF, use Ridge/PCA |
| Extrapolation | Predictions outside training range | Be cautious about extrapolation |
| Forgetting to standardize for Ridge/Lasso | Different scales affect penalty | Standardize features |
| Using R² as only metric | R² always increases with more features | Use adjusted R², cross-validation |

### Diagnosing Problems

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| Large coefficient variance | Multicollinearity | Ridge, VIF check |
| Heteroscedastic residuals | Non-constant variance | Weighted LS, transform y |
| Non-normal residuals | Outliers or wrong model | Robust regression, transform |
| High training R², low test R² | Overfitting | Regularization, more data |

---

## 6. Mini Example

### Python Example: OLS, Ridge, Lasso Comparison

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate data with multicollinearity
np.random.seed(42)
n, p = 100, 10
X = np.random.randn(n, p)
X[:, 1] = X[:, 0] + np.random.randn(n) * 0.1  # Correlated feature
true_beta = np.array([3, 0, -2, 1, 0, 0, 0, 0, 0, 0])  # Sparse
y = X @ true_beta + np.random.randn(n) * 0.5

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Fit models
ols = LinearRegression().fit(X_train_s, y_train)
ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
lasso = Lasso(alpha=0.1).fit(X_train_s, y_train)

# Evaluate
for name, model in [('OLS', ols), ('Ridge', ridge), ('Lasso', lasso)]:
    pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    nonzero = np.sum(np.abs(model.coef_) > 0.01)
    print(f"{name:6s}: MSE={mse:.3f}, R²={r2:.3f}, Non-zero={nonzero}")

# Output:
# OLS   : MSE=0.312, R²=0.946, Non-zero=10
# Ridge : MSE=0.298, R²=0.948, Non-zero=10
# Lasso : MSE=0.287, R²=0.950, Non-zero=3
```

### Closed-Form Implementation

```python
def ols_closed_form(X, y):
    """Closed-form OLS solution."""
    return np.linalg.solve(X.T @ X, X.T @ y)

def ridge_closed_form(X, y, lambda_):
    """Closed-form Ridge solution."""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lambda_ * np.eye(p), X.T @ y)

# Test
X_with_intercept = np.c_[np.ones(len(X_train)), X_train_s]
beta_ols = ols_closed_form(X_with_intercept, y_train)
print(f"Closed-form coefficients: {beta_ols[:3]}")
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Derive the OLS closed-form solution.</strong></summary>

Start with loss: $\mathcal{L}(\beta) = \|y - X\beta\|_2^2$

Expand: $\mathcal{L} = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$

Gradient: $\nabla_\beta \mathcal{L} = -2X^Ty + 2X^TX\beta$

Set to zero: $X^TX\beta = X^Ty$

Solution: $\hat{\beta} = (X^TX)^{-1}X^Ty$
</details>

<details>
<summary><strong>Q2: What are the four assumptions of linear regression for inference?</strong></summary>

1. **Linearity**: True relationship is linear in parameters
2. **Independence**: Errors are independent of each other
3. **Homoscedasticity**: Error variance is constant across all X values
4. **Normality**: Errors are normally distributed

Mnemonic: LINE (Linearity, Independence, Normality, Equal variance)
</details>

<details>
<summary><strong>Q3: When would you choose Ridge over Lasso?</strong></summary>

Choose **Ridge** when:
- Features are correlated and you want to keep all of them
- You believe all features contribute (no true zeros)
- You want stable predictions (Ridge solution is unique)

Choose **Lasso** when:
- You want automatic feature selection
- You believe some features are irrelevant
- Interpretability via sparse models is important
</details>

<details>
<summary><strong>Q4: What is multicollinearity and how do you detect/handle it?</strong></summary>

**Multicollinearity**: High correlation between predictor variables

**Detection**:
- Correlation matrix
- Variance Inflation Factor (VIF): VIF > 10 indicates problem
- Condition number of X'X

**Handling**:
- Remove one of correlated features
- Ridge regression (most common)
- PCA to create uncorrelated components
- Collect more data
</details>

<details>
<summary><strong>Q5: What is the Gauss-Markov theorem?</strong></summary>

Under the assumptions:
1. $\mathbb{E}[\epsilon] = 0$
2. $\text{Var}(\epsilon) = \sigma^2 I$ (homoscedasticity, uncorrelated)
3. Fixed design matrix X

The OLS estimator is **BLUE**: Best Linear Unbiased Estimator.
- "Best" = minimum variance among all linear unbiased estimators
- Does NOT require normality (that's only for inference)
</details>

<details>
<summary><strong>Q6: Why does Ridge regression always have a solution but OLS might not?</strong></summary>

OLS requires inverting $X^TX$, which fails when:
- More features than samples (p > n)
- Features are linearly dependent

Ridge adds $\lambda I$: $(X^TX + \lambda I)$ is always invertible for $\lambda > 0$ because:
- All eigenvalues of $X^TX$ are ≥ 0
- Adding $\lambda$ makes all eigenvalues > 0
- Positive definite matrices are always invertible
</details>

---

## 8. References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 3.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 3.
3. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." JRSS-B.
4. Hoerl, A. E., & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." Technometrics.
5. Seber, G. A., & Lee, A. J. (2012). *Linear Regression Analysis*. Wiley.
