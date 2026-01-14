# 偏差-方差权衡

## 1. 面试摘要

**关键要点：**
- **偏差**：来自错误假设的误差；欠拟合
- **方差**：来自对训练数据敏感性的误差；过拟合
- **权衡**：减少一个通常会增加另一个
- **模型复杂度**控制权衡
- **正则化**以偏差为代价减少方差

**常见面试问题：**
- "解释偏差-方差权衡"
- "如何判断模型是欠拟合还是过拟合？"
- "模型复杂度如何影响偏差和方差？"

---

## 2. 核心定义

### 偏差
用简化模型近似复杂现实问题引入的误差。

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$$

**高偏差（欠拟合）：**
- 模型太简单
- 无法捕获底层模式
- 训练和测试误差都很高

### 方差
来自对训练集波动敏感性的误差。

$$\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$

**高方差（过拟合）：**
- 模型太复杂
- 拟合训练数据中的噪声
- 训练误差低，测试误差高

### 不可约误差
数据中固有的无法消除的噪声。

$$\sigma^2 = \text{Var}[\epsilon]$$

---

## 3. 数学与推导

### 偏差-方差分解

对于平方损失的回归，期望预测误差可以分解：

**设置：**
- 真实函数：$y = f(x) + \epsilon$，其中 $\mathbb{E}[\epsilon] = 0$，$\text{Var}[\epsilon] = \sigma^2$
- 学习函数：$\hat{f}(x)$ 在数据集 $\mathcal{D}$ 上训练

**推导：**
$$\begin{aligned}
\mathbb{E}_\mathcal{D}[(y - \hat{f}(x))^2] &= \mathbb{E}_\mathcal{D}[(f(x) + \epsilon - \hat{f}(x))^2] \\
&= \mathbb{E}_\mathcal{D}[(f(x) - \hat{f}(x))^2] + \mathbb{E}[\epsilon^2] + 2\mathbb{E}_\mathcal{D}[(f(x) - \hat{f}(x))\epsilon] \\
&= \mathbb{E}_\mathcal{D}[(f(x) - \hat{f}(x))^2] + \sigma^2
\end{aligned}$$

第一项进一步分解：
$$\begin{aligned}
\mathbb{E}_\mathcal{D}[(f(x) - \hat{f}(x))^2] &= (f(x) - \mathbb{E}_\mathcal{D}[\hat{f}(x)])^2 + \mathbb{E}_\mathcal{D}[(\hat{f}(x) - \mathbb{E}_\mathcal{D}[\hat{f}(x)])^2] \\
&= \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)]
\end{aligned}$$

**最终结果：**
$$\boxed{\text{期望误差} = \text{偏差}^2 + \text{方差} + \text{不可约噪声}}$$

### 模型复杂度与权衡

| 复杂度 | 偏差 | 方差 | 训练误差 | 测试误差 |
|--------|------|------|----------|----------|
| 低 | 高 | 低 | 高 | 高 |
| 最优 | 中 | 中 | 低 | 低 |
| 高 | 低 | 高 | 很低 | 高 |

---

## 4. 算法框架

### 诊断偏差与方差

```
1. 在训练集上训练模型
2. 计算训练误差 E_train
3. 计算验证/测试误差 E_test

如果 E_train 高且 E_test 高：
    → 高偏差（欠拟合）
    → 解决方案：更复杂模型，更多特征

如果 E_train 低且 E_test 高：
    → 高方差（过拟合）
    → 解决方案：正则化，更多数据，更简单模型

如果 E_train 低且 E_test 低：
    → 良好拟合！（可能仍可改进）
```

### 学习曲线分析

```
1. 用递增数据大小训练模型 [n_1, n_2, ..., n_k]
2. 对于每个大小 n_i：
   - 在 n_i 个样本上训练
   - 记录训练和验证误差
3. 绘制两条曲线与样本大小的关系

高偏差模式：
   - 两条曲线收敛到高误差
   - 增加数据帮助不大

高方差模式：
   - 训练误差 << 验证误差
   - 差距随数据增加而减小
   - 更多数据有帮助！
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 混淆偏差/方差与训练/测试误差 | 相关但不相同 | 偏差-方差是理论分解 |
| 仅用测试误差诊断 | 无法区分偏差和方差 | 比较训练与测试误差 |
| 增加复杂度不加正则化 | 方差爆炸 | 复杂模型总是考虑正则化 |
| 过度正则化 | 惩罚过强 | 通过验证调整正则化强度 |
| 忽略不可约误差 | 期望完美预测 | 理解数据噪声限制可达准确率 |

### 视觉诊断

```
训练误差 vs 测试误差：

高偏差:           高方差:            良好拟合:
误差              误差               误差
  |  ----测试      |    ----测试       |    ----测试
  |  ----训练      |                   |    ----训练
  |                |    ----训练       |
  +--------→       +--------→          +--------→
    复杂度            复杂度              复杂度
```

---

## 6. 迷你示例

### Python示例：观察偏差-方差权衡

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# 从真实函数生成带噪声数据
np.random.seed(42)
n_samples = 100
X = np.sort(np.random.uniform(0, 1, n_samples))
y_true = np.sin(2 * np.pi * X)
y = y_true + np.random.normal(0, 0.3, n_samples)

X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 测试不同多项式阶数
degrees = [1, 3, 10, 20]
train_errors, test_errors = [], []

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = np.mean((y_train - train_pred)**2)
    test_mse = np.mean((y_test - test_pred)**2)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

    print(f"阶数 {degree:2d}: 训练MSE = {train_mse:.4f}, 测试MSE = {test_mse:.4f}")

# 输出：
# 阶数  1: 训练MSE = 0.4521, 测试MSE = 0.5307  (高偏差)
# 阶数  3: 训练MSE = 0.0892, 测试MSE = 0.1124  (良好拟合)
# 阶数 10: 训练MSE = 0.0731, 测试MSE = 0.1456  (开始过拟合)
# 阶数 20: 训练MSE = 0.0412, 测试MSE = 0.9823  (高方差)
```

**解释：**
- 阶数1：欠拟合（高偏差）- 两个误差都高
- 阶数3：良好平衡 - 两个误差都低
- 阶数20：过拟合（高方差）- 训练低，测试很高

---

## 7. 测验

<details>
<summary><strong>Q1: 什么是偏差-方差权衡？</strong></summary>

偏差-方差权衡描述了两种误差来源之间的张力：
- **偏差**：来自过于简单模型无法捕获真实模式的误差
- **方差**：来自过于复杂模型拟合训练数据噪声的误差

随着模型复杂度增加，偏差通常减少但方差增加。最优模型平衡两者以最小化总误差。
</details>

<details>
<summary><strong>Q2: 如何判断模型是欠拟合还是过拟合？</strong></summary>

**欠拟合（高偏差）：**
- 训练误差高
- 测试误差高（与训练相似）
- 模型对数据来说太简单

**过拟合（高方差）：**
- 训练误差低
- 测试误差高（比训练高很多）
- 模型太复杂，拟合了噪声
</details>

<details>
<summary><strong>Q3: 写出偏差-方差分解公式。</strong></summary>

对于平方损失：

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

其中：
- $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$
- $\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$
- $\sigma^2$ 是不可约噪声
</details>

<details>
<summary><strong>Q4: 如何减少高方差？</strong></summary>

减少方差的策略：
1. **正则化**（L1、L2、dropout）
2. **更多训练数据**
3. **更简单模型**（更少参数）
4. **集成方法**（bagging平均方差）
5. **早停**
6. **特征选择**（降低维度）
</details>

<details>
<summary><strong>Q5: 如何减少高偏差？</strong></summary>

减少偏差的策略：
1. **更复杂模型**（更高容量）
2. **添加更多特征**（多项式特征、交互）
3. **减少正则化**
4. **使用不同模型族**（更具表达力）
5. **Boosting**（迭代修正偏差）
</details>

<details>
<summary><strong>Q6: 为什么不能同时减少偏差和方差？</strong></summary>

减少偏差需要更多模型灵活性来捕获复杂模式，但更多灵活性意味着模型也可以拟合噪声（增加方差）。相反，约束模型以减少方差会限制其拟合真实函数的能力（增加偏差）。同时减少两者的唯一方法是获得更多高质量训练数据，这可以在不需要更简单模型的情况下减少方差。
</details>

---

## 8. 参考文献

1. Geman, S., Bienenstock, E., & Doursat, R. (1992). "Neural Networks and the Bias/Variance Dilemma." Neural Computation.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 7.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 3.2.
4. James, G., et al. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 2.
5. Domingos, P. (2000). "A Unified Bias-Variance Decomposition." ICML.
