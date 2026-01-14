# 逻辑回归

## 1. 面试摘要

**关键要点：**
- 使用sigmoid函数的**二分类器**
- **决策边界**是线性的（超平面）
- 输出**概率**，不只是类别标签
- 使用**交叉熵损失**（对数损失）
- 没有闭式解；需要**迭代优化**
- 正则化（L1/L2）防止过拟合

**常见面试问题：**
- "为什么分类用逻辑回归而不是线性回归？"
- "推导逻辑回归的梯度"
- "什么是决策边界？"

---

## 2. 核心定义

### 模型
$$P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是sigmoid函数。

### 决策边界
如果 $P(y=1|x) > 0.5$ 则预测类别1，等价于 $w^Tx + b > 0$。

决策边界是超平面：$w^Tx + b = 0$

### 对数几率（Logit）
$$\log\frac{P(y=1|x)}{P(y=0|x)} = w^Tx + b$$

对特征线性，因此叫"逻辑回归"。

### 损失函数（二元交叉熵）
$$\mathcal{L}(w) = -\frac{1}{n}\sum_{i=1}^n [y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

---

## 3. 数学与推导

### Sigmoid性质

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**关键性质：**
- 值域：$(0, 1)$
- $\sigma(0) = 0.5$
- $\sigma(-z) = 1 - \sigma(z)$
- 导数：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### 梯度推导

对于单个样本，特征$x$和标签$y$：

$$\mathcal{L} = -[y\log(\sigma(w^Tx)) + (1-y)\log(1-\sigma(w^Tx))]$$

令 $z = w^Tx$ 且 $\hat{p} = \sigma(z)$。

$$\frac{\partial \mathcal{L}}{\partial z} = -\frac{y}{\hat{p}}\cdot\hat{p}(1-\hat{p}) + \frac{1-y}{1-\hat{p}}\cdot\hat{p}(1-\hat{p})$$
$$= -y(1-\hat{p}) + (1-y)\hat{p} = \hat{p} - y$$

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w} = (\hat{p} - y)x$$

**所有样本的梯度：**
$$\nabla_w \mathcal{L} = \frac{1}{n}\sum_{i=1}^n (\hat{p}_i - y_i)x_i = \frac{1}{n}X^T(\hat{p} - y)$$

### 最大似然解释

似然：$\prod_i \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}$

对数似然：$\sum_i [y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$

最小化交叉熵 = 最大化对数似然。

### 多分类：Softmax回归

对于$K$个类别：
$$P(y=k|x) = \frac{e^{w_k^Tx}}{\sum_{j=1}^K e^{w_j^Tx}}$$

损失：$K$个类别的交叉熵。

---

## 4. 算法框架

### 逻辑回归的梯度下降
```
输入：X (n×p)，y (n×1)，学习率η，迭代次数T
初始化：w = 0，b = 0

对于 t = 1 到 T：
    # 前向传播
    z = X @ w + b
    p_hat = sigmoid(z)

    # 计算梯度
    error = p_hat - y
    grad_w = (1/n) * X.T @ error
    grad_b = (1/n) * sum(error)

    # 更新
    w = w - η * grad_w
    b = b - η * grad_b

返回 w, b
```

### 牛顿-拉夫森（更快收敛）
```
输入：X，y，迭代次数T
初始化：w = 0

对于 t = 1 到 T：
    p_hat = sigmoid(X @ w)
    gradient = X.T @ (p_hat - y)

    # Hessian：H = X.T @ diag(p*(1-p)) @ X
    S = diag(p_hat * (1 - p_hat))
    Hessian = X.T @ S @ X

    # 牛顿更新
    w = w - inv(Hessian) @ gradient

返回 w
```

### 正则化逻辑回归
```
# L2正则化
Loss = cross_entropy + (λ/2) * ||w||²
grad_w = (1/n) * X.T @ (p_hat - y) + λ * w

# L1正则化（需要特殊求解器）
Loss = cross_entropy + λ * ||w||₁
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 完美分离 | 权重发散到无穷 | 使用正则化 |
| 使用准确率作为损失 | 不可微 | 使用交叉熵 |
| 忘记标准化 | 收敛慢，正则化不公平 | 标准化特征 |
| 阈值总是0.5 | 可能不是最优 | 基于精确率-召回率权衡调整 |
| 忽略类别不平衡 | 偏向多数类 | 加权损失，重采样 |

### 为什么分类不用线性回归？

| 问题 | 线性回归 | 逻辑回归 |
|------|----------|----------|
| 输出范围 | $(-\infty, +\infty)$ | $(0, 1)$（概率） |
| 异常值 | 敏感，移动边界 | 鲁棒 |
| 损失函数 | MSE（不适合分类） | 交叉熵（合适） |
| 解释 | 没有概率意义 | 类别概率 |

---

## 6. 迷你示例

### Python示例：从头实现逻辑回归

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression_gd(X, y, lr=0.1, epochs=1000, reg=0.01):
    """带L2正则化的梯度下降。"""
    n, p = X.shape
    w = np.zeros(p)
    b = 0

    for _ in range(epochs):
        z = X @ w + b
        p_hat = sigmoid(z)

        # 梯度
        dw = (1/n) * X.T @ (p_hat - y) + reg * w
        db = (1/n) * np.sum(p_hat - y)

        # 更新
        w -= lr * dw
        b -= lr * db

    return w, b

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)

# 生成数据
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 标准化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 训练
w, b = logistic_regression_gd(X_train_s, y_train)

# 评估
train_acc = np.mean(predict(X_train_s, w, b) == y_train)
test_acc = np.mean(predict(X_test_s, w, b) == y_test)
print(f"训练准确率：{train_acc:.3f}")
print(f"测试准确率：{test_acc:.3f}")

# 输出：
# 训练准确率：0.872
# 测试准确率：0.850
```

### 决策边界可视化

```python
import matplotlib.pyplot as plt

# 用于可视化的2D示例
X_2d, y_2d = make_classification(n_samples=200, n_features=2,
                                  n_redundant=0, random_state=42)
scaler_2d = StandardScaler()
X_2d_s = scaler_2d.fit_transform(X_2d)

w_2d, b_2d = logistic_regression_gd(X_2d_s, y_2d)

# 绘制决策边界：w[0]*x1 + w[1]*x2 + b = 0
# => x2 = -(w[0]*x1 + b) / w[1]
x1_range = np.linspace(X_2d_s[:, 0].min()-1, X_2d_s[:, 0].max()+1, 100)
x2_boundary = -(w_2d[0] * x1_range + b_2d) / w_2d[1]

plt.scatter(X_2d_s[y_2d==0, 0], X_2d_s[y_2d==0, 1], label='类别 0')
plt.scatter(X_2d_s[y_2d==1, 0], X_2d_s[y_2d==1, 1], label='类别 1')
plt.plot(x1_range, x2_boundary, 'k--', label='决策边界')
plt.legend()
plt.title('逻辑回归决策边界')
```

---

## 7. 测验

<details>
<summary><strong>Q1: 为什么分类不能用线性回归？</strong></summary>

线性回归用于分类有问题：
1. **输出范围**：预测可以<0或>1，不是有效概率
2. **异常值敏感性**：极端值移动决策边界
3. **损失函数**：MSE惩罚置信的正确预测
4. **没有概率解释**：不能解释为类别概率

逻辑回归使用sigmoid将输出限制在(0,1)，并使用交叉熵损失，这对分类是合适的。
</details>

<details>
<summary><strong>Q2: 推导对数损失对权重的梯度。</strong></summary>

样本$i$的损失：$\mathcal{L}_i = -[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$

其中 $\hat{p}_i = \sigma(w^Tx_i)$。

使用链式法则和 $\sigma'(z) = \sigma(z)(1-\sigma(z))$：

$$\frac{\partial \mathcal{L}_i}{\partial w} = (\hat{p}_i - y_i)x_i$$

所有样本：$\nabla_w\mathcal{L} = \frac{1}{n}X^T(\hat{p} - y)$

优雅的结果：梯度 = 预测 - 标签，按特征缩放。
</details>

<details>
<summary><strong>Q3: 逻辑回归的决策边界是什么？</strong></summary>

决策边界是 $P(y=1|x) = 0.5$ 的地方，发生在：
$$w^Tx + b = 0$$

这是特征空间中的**线性**超平面。在2D中：一条线。在3D中：一个平面。

一边的点：$w^Tx + b > 0$ → 预测类别1
另一边的点：$w^Tx + b < 0$ → 预测类别0
</details>

<details>
<summary><strong>Q4: 完美分离时会发生什么？</strong></summary>

当类别完美可分时：
- 任何分离它们的边界都实现零训练损失
- 权重增长到无穷大以最大化置信度
- 模型变得过度自信，校准差
- 梯度下降永远不收敛

**解决方案**：
- L2正则化（限制权重幅度）
- 早停
- 贝叶斯逻辑回归
</details>

<details>
<summary><strong>Q5: 逻辑回归和对数几率有什么关系？</strong></summary>

逻辑回归将**对数几率**（logit）建模为特征的线性函数：

$$\log\frac{P(y=1|x)}{P(y=0|x)} = w^Tx + b$$

这意味着：
- $x_j$每增加一个单位，对数几率变化$w_j$
- 等价地，几率乘以$e^{w_j}$
- 为特征重要性提供可解释的系数
</details>

<details>
<summary><strong>Q6: 多分类逻辑回归（softmax）如何工作？</strong></summary>

对于$K$个类别，使用softmax：
$$P(y=k|x) = \frac{e^{w_k^Tx}}{\sum_{j=1}^K e^{w_j^Tx}}$$

性质：
- 输出和为1（有效概率分布）
- K=2时退化为sigmoid（一对多）
- 损失：分类交叉熵
- 梯度：类似形式，每个类别$(\hat{p} - y)$
</details>

---

## 8. 参考文献

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 4.3.
2. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
3. Ng, A. (2012). "Machine Learning." Coursera. 逻辑回归讲座。
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Section 4.4.
5. Cox, D. R. (1958). "The Regression Analysis of Binary Sequences." JRSS-B.
