# 线性回归

## 1. 面试摘要

**关键要点：**
- **OLS**：普通最小二乘 - 闭式解
- **Ridge**：L2正则化 - 处理多重共线性
- **Lasso**：L1正则化 - 特征选择
- 关键假设：线性、独立、同方差、误差正态
- 知道闭式解以及何时使用梯度下降

**常见面试问题：**
- "线性回归的假设是什么？"
- "推导OLS的闭式解"
- "什么时候用Ridge vs Lasso？"

---

## 2. 核心定义

### 模型
$$y = X\beta + \epsilon$$

其中：
- $y \in \mathbb{R}^n$：目标向量
- $X \in \mathbb{R}^{n \times p}$：设计矩阵（带截距列）
- $\beta \in \mathbb{R}^p$：系数
- $\epsilon \in \mathbb{R}^n$：误差项

### OLS目标
$$\min_\beta \|y - X\beta\|_2^2 = \min_\beta \sum_{i=1}^n (y_i - x_i^T\beta)^2$$

### Ridge回归
$$\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2$$

### Lasso回归
$$\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_1$$

---

## 3. 数学与推导

### OLS闭式解

**推导：**
$$\mathcal{L}(\beta) = (y - X\beta)^T(y - X\beta)$$

展开：
$$\mathcal{L}(\beta) = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$$

求梯度并令其为零：
$$\nabla_\beta \mathcal{L} = -2X^Ty + 2X^TX\beta = 0$$

求解$\beta$：
$$\boxed{\hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty}$$

**要求**：$X^TX$可逆（满列秩）

### Ridge闭式解

$$\mathcal{L}(\beta) = \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2$$

$$\nabla_\beta \mathcal{L} = -2X^Ty + 2X^TX\beta + 2\lambda\beta = 0$$

$$\boxed{\hat{\beta}_{Ridge} = (X^TX + \lambda I)^{-1}X^Ty}$$

**优势**：对于$\lambda > 0$总是可逆

### 高斯-马尔可夫定理

在假设下：
1. $\mathbb{E}[\epsilon] = 0$
2. $\text{Var}(\epsilon) = \sigma^2 I$（同方差，不相关）
3. $X$是固定/非随机的

**结果**：OLS是BLUE（最佳线性无偏估计量）
- 所有线性无偏估计量中方差最小

### 推断的假设

对于假设检验和置信区间：
1. **线性**：$y = X\beta + \epsilon$
2. **独立**：误差相互独立
3. **同方差**：$\text{Var}(\epsilon_i) = \sigma^2$恒定
4. **正态**：$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

---

## 4. 算法框架

### 使用正规方程的OLS
```
输入：X (n×p)，y (n×1)
1. 计算 X^T X (p×p矩阵)
2. 计算 X^T y (p×1向量)
3. 求解 (X^T X) β = X^T y
   - 如果X^T X正定使用Cholesky分解
   - 或使用QR分解以获得数值稳定性
4. 返回 β
```

### 线性回归的梯度下降
```
输入：X，y，学习率η，迭代次数T
初始化：β = 0或随机

对于 t = 1 到 T：
    predictions = X @ β
    errors = predictions - y
    gradient = (2/n) * X.T @ errors
    β = β - η * gradient

返回 β
```

### 选择OLS、Ridge、Lasso
```
如果 p < n 且 X 满秩：
    → OLS（闭式解）

如果多重共线性或 p ≈ n：
    → Ridge（将相关特征一起收缩）

如果你想要特征选择：
    → Lasso（将一些系数设为零）

如果高维 (p >> n)：
    → Lasso或弹性网络
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 不检查假设 | 盲目应用OLS | 绘制残差，检查诊断 |
| 忽略多重共线性 | 相关预测变量 | 检查VIF，使用Ridge/PCA |
| 外推 | 预测超出训练范围 | 对外推保持谨慎 |
| Ridge/Lasso忘记标准化 | 不同尺度影响惩罚 | 标准化特征 |
| 只用R²作为指标 | R²随特征增加而增加 | 使用调整R²，交叉验证 |

### 诊断问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 系数方差大 | 多重共线性 | Ridge，VIF检查 |
| 异方差残差 | 非恒定方差 | 加权LS，变换y |
| 非正态残差 | 异常值或错误模型 | 稳健回归，变换 |
| 训练R²高，测试R²低 | 过拟合 | 正则化，更多数据 |

---

## 6. 迷你示例

### Python示例：OLS、Ridge、Lasso比较

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成带多重共线性的数据
np.random.seed(42)
n, p = 100, 10
X = np.random.randn(n, p)
X[:, 1] = X[:, 0] + np.random.randn(n) * 0.1  # 相关特征
true_beta = np.array([3, 0, -2, 1, 0, 0, 0, 0, 0, 0])  # 稀疏
y = X @ true_beta + np.random.randn(n) * 0.5

# 划分和标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 拟合模型
ols = LinearRegression().fit(X_train_s, y_train)
ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
lasso = Lasso(alpha=0.1).fit(X_train_s, y_train)

# 评估
for name, model in [('OLS', ols), ('Ridge', ridge), ('Lasso', lasso)]:
    pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    nonzero = np.sum(np.abs(model.coef_) > 0.01)
    print(f"{name:6s}: MSE={mse:.3f}, R²={r2:.3f}, 非零={nonzero}")

# 输出：
# OLS   : MSE=0.312, R²=0.946, 非零=10
# Ridge : MSE=0.298, R²=0.948, 非零=10
# Lasso : MSE=0.287, R²=0.950, 非零=3
```

### 闭式解实现

```python
def ols_closed_form(X, y):
    """闭式OLS解。"""
    return np.linalg.solve(X.T @ X, X.T @ y)

def ridge_closed_form(X, y, lambda_):
    """闭式Ridge解。"""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lambda_ * np.eye(p), X.T @ y)

# 测试
X_with_intercept = np.c_[np.ones(len(X_train)), X_train_s]
beta_ols = ols_closed_form(X_with_intercept, y_train)
print(f"闭式解系数：{beta_ols[:3]}")
```

---

## 7. 测验

<details>
<summary><strong>Q1: 推导OLS闭式解。</strong></summary>

从损失开始：$\mathcal{L}(\beta) = \|y - X\beta\|_2^2$

展开：$\mathcal{L} = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$

梯度：$\nabla_\beta \mathcal{L} = -2X^Ty + 2X^TX\beta$

令其为零：$X^TX\beta = X^Ty$

解：$\hat{\beta} = (X^TX)^{-1}X^Ty$
</details>

<details>
<summary><strong>Q2: 推断的线性回归四个假设是什么？</strong></summary>

1. **线性**：真实关系对参数是线性的
2. **独立**：误差相互独立
3. **同方差**：所有X值的误差方差恒定
4. **正态**：误差服从正态分布

记忆：LINE（线性、独立、正态、等方差）
</details>

<details>
<summary><strong>Q3: 什么时候选择Ridge而不是Lasso？</strong></summary>

选择**Ridge**当：
- 特征相关且你想保留所有特征
- 你相信所有特征都有贡献（没有真正的零）
- 你想要稳定预测（Ridge解唯一）

选择**Lasso**当：
- 你想要自动特征选择
- 你相信一些特征不相关
- 通过稀疏模型的可解释性很重要
</details>

<details>
<summary><strong>Q4: 什么是多重共线性？如何检测/处理？</strong></summary>

**多重共线性**：预测变量之间高度相关

**检测**：
- 相关矩阵
- 方差膨胀因子（VIF）：VIF > 10表示有问题
- X'X的条件数

**处理**：
- 移除相关特征之一
- Ridge回归（最常见）
- PCA创建不相关成分
- 收集更多数据
</details>

<details>
<summary><strong>Q5: 什么是高斯-马尔可夫定理？</strong></summary>

在假设下：
1. $\mathbb{E}[\epsilon] = 0$
2. $\text{Var}(\epsilon) = \sigma^2 I$（同方差，不相关）
3. 固定设计矩阵X

OLS估计量是**BLUE**：最佳线性无偏估计量。
- "最佳" = 所有线性无偏估计量中方差最小
- 不需要正态性（那只是推断需要的）
</details>

<details>
<summary><strong>Q6: 为什么Ridge回归总有解但OLS可能没有？</strong></summary>

OLS需要求逆$X^TX$，当以下情况时失败：
- 特征比样本多（p > n）
- 特征线性相关

Ridge添加$\lambda I$：$(X^TX + \lambda I)$对于$\lambda > 0$总是可逆因为：
- $X^TX$的所有特征值都≥0
- 添加$\lambda$使所有特征值>0
- 正定矩阵总是可逆的
</details>

---

## 8. 参考文献

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 3.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 3.
3. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." JRSS-B.
4. Hoerl, A. E., & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." Technometrics.
5. Seber, G. A., & Lee, A. J. (2012). *Linear Regression Analysis*. Wiley.
