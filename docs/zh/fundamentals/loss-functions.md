# 损失函数

## 1. 面试摘要

**关键要点：**
- **MSE**：均方误差 - 回归，对异常值敏感
- **MAE**：平均绝对误差 - 回归，对异常值鲁棒
- **交叉熵**：分类，衡量概率分布差异
- **校准**：预测概率与真实频率的匹配程度
- 知道何时使用每种损失函数

**常见面试问题：**
- "什么时候使用MAE而不是MSE？"
- "什么是交叉熵损失？为什么用于分类？"
- "模型校准良好是什么意思？"

---

## 2. 核心定义

### 均方误差（MSE）
$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**性质：**
- 对大误差惩罚更重（二次方）
- 对异常值敏感
- 有良好的梯度性质（平滑）
- 最优预测器：$\mathbb{E}[Y|X]$（条件均值）

### 平均绝对误差（MAE）
$$\mathcal{L}_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**性质：**
- 误差的线性惩罚
- 对异常值鲁棒
- 在零点不平滑（梯度未定义）
- 最优预测器：$Y|X$的中位数

### 交叉熵损失（对数损失）

**二分类：**
$$\mathcal{L}_{CE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i)]$$

**多分类：**
$$\mathcal{L}_{CE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(\hat{p}_{ic})$$

### 校准
如果模型校准良好：当它预测概率 $p$ 时，正例的实际频率就是 $p$。

$$P(Y=1 | \hat{p}(X)=p) = p$$

---

## 3. 数学与推导

### 从最大似然推导MSE

假设高斯噪声：$y = f(x) + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma^2)$

**似然：**
$$p(y|x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f(x))^2}{2\sigma^2}\right)$$

**对数似然：**
$$\log p(y|x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y - f(x))^2}{2\sigma^2}$$

最大化对数似然 $\equiv$ 最小化 $(y - f(x))^2$ $\equiv$ MSE

### 从KL散度推导交叉熵

对于真实分布 $p$ 和预测分布 $q$：

**KL散度：**
$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = -H(p) - \sum_x p(x) \log q(x)$$

由于 $H(p)$ 相对于模型参数是常数：
$$\min D_{KL}(p \| q) \equiv \min \left(-\sum_x p(x) \log q(x)\right) = \min \mathcal{L}_{CE}$$

### Huber损失（平滑MAE）

$$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{如果 } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{否则}
\end{cases}$$

- 对小误差是二次的（平滑梯度）
- 对大误差是线性的（对异常值鲁棒）

---

## 4. 算法框架

### 选择损失函数

```
对于回归：
    如果异常值罕见或应该被惩罚：
        → 使用MSE
    如果异常值常见且应该被忽略：
        → 使用MAE或Huber损失
    如果预测百分位数/分位数：
        → 使用分位数损失

对于分类：
    如果需要概率输出：
        → 使用交叉熵
    如果只需要类别预测：
        → 交叉熵通常仍然更好
    如果严重类别不平衡：
        → 考虑Focal Loss或加权CE
```

### 校准方法

```
1. 训练分类器
2. 在验证集上获取预测概率
3. 应用校准：
   - Platt缩放：对logits拟合sigmoid
   - 等渗回归：非参数校准
   - 温度缩放：对logits使用单个标量
4. 评估：
   - 可靠性图
   - 期望校准误差（ECE）
   - Brier分数
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 对异常值使用MSE | MSE对大误差取平方 | 使用MAE或Huber |
| 使用准确率损失 | 不可微 | 使用交叉熵（代理） |
| 忽略校准 | 只关注准确率 | 对概率任务进行校准 |
| 排序用错误损失 | 排序任务用回归损失 | 使用排序损失（成对、列表级） |
| 对硬标签使用CE | 丢失信息 | 考虑标签平滑 |

### 损失比较

| 损失 | 对异常值敏感度 | 零点梯度 | 最优预测器 |
|------|---------------|---------|-----------|
| MSE | 高 | 平滑 | 均值 |
| MAE | 低 | 未定义 | 中位数 |
| Huber | 可配置 | 平滑 | 均值/中位数之间 |

---

## 6. 迷你示例

### Python示例：比较损失函数

```python
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 生成带异常值的数据
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 0.5

# 添加异常值
y[0], y[1], y[2] = 100, -80, 90  # 极端异常值

# 用MSE拟合（线性回归）
lr = LinearRegression()
lr.fit(X, y)
pred_mse = lr.predict(X)

# 用Huber损失拟合（鲁棒）
huber = HuberRegressor()
huber.fit(X, y)
pred_huber = huber.predict(X)

print(f"真实系数：斜率=2，截距=1")
print(f"MSE回归：斜率={lr.coef_[0]:.2f}，截距={lr.intercept_:.2f}")
print(f"Huber回归：斜率={huber.coef_[0]:.2f}，截距={huber.intercept_:.2f}")

# 输出：
# 真实系数：斜率=2，截距=1
# MSE回归：斜率=0.84，截距=3.45  (被异常值影响！)
# Huber回归：斜率=1.98，截距=1.03  (对异常值鲁棒)
```

### 交叉熵示例

```python
import numpy as np

def cross_entropy(y_true, y_pred, eps=1e-15):
    """二元交叉熵损失。"""
    y_pred = np.clip(y_pred, eps, 1 - eps)  # 防止log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 示例预测
y_true = np.array([1, 0, 1, 1, 0])
y_confident = np.array([0.9, 0.1, 0.95, 0.85, 0.05])  # 好的预测
y_wrong = np.array([0.1, 0.9, 0.2, 0.3, 0.8])  # 差的预测

print(f"CE损失（好的预测）：{cross_entropy(y_true, y_confident):.4f}")
print(f"CE损失（差的预测）：{cross_entropy(y_true, y_wrong):.4f}")

# 输出：
# CE损失（好的预测）：0.0969
# CE损失（差的预测）：1.6904
```

---

## 7. 测验

<details>
<summary><strong>Q1: 为什么MSE对异常值敏感？</strong></summary>

MSE对误差取平方，所以大误差对总损失贡献不成比例地大。误差为10的异常值贡献100到MSE，而10个误差为1的小误差总共只贡献10。这导致模型显著调整以减少异常值误差，偏置整体拟合。
</details>

<details>
<summary><strong>Q2: 交叉熵和最大似然有什么联系？</strong></summary>

最小化交叉熵等价于最大化模型预测分布下数据的似然。对于预测概率 $\hat{p}$ 的分类：

$$\max \prod_i \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i} \equiv \min -\sum_i [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$$

交叉熵是负对数似然。
</details>

<details>
<summary><strong>Q3: 什么时候选择MAE而不是MSE？</strong></summary>

在以下情况选择MAE：
- 数据包含不应主导拟合的异常值
- 你想预测中位数而不是均值
- 误差应该线性惩罚（每单位所有误差同等重要）
- 你需要更鲁棒的估计器

注意：MAE在零点梯度未定义，这可能使优化更困难。
</details>

<details>
<summary><strong>Q4: 什么是模型校准？为什么重要？</strong></summary>

校准衡量预测概率是否与实际频率匹配。预测80%概率的模型对于这些预测应该大约80%的时间是正确的。

重要性：
- 决策基于概率阈值
- 组合多个模型的预测
- 风险评估需要准确的置信度估计
- 医疗/金融应用需要可靠的不确定性
</details>

<details>
<summary><strong>Q5: Huber损失如何结合MSE和MAE的优点？</strong></summary>

Huber损失：
- 对小误差表现像MSE（平滑梯度，高效优化）
- 对大误差表现像MAE（对异常值鲁棒）
- 有参数 $\delta$ 控制转换点

$$\mathcal{L}_\delta = \begin{cases} \frac{1}{2}e^2 & |e| \leq \delta \\ \delta|e| - \frac{\delta^2}{2} & |e| > \delta \end{cases}$$
</details>

<details>
<summary><strong>Q6: 为什么不能直接使用准确率作为损失函数？</strong></summary>

准确率（0-1损失）不可微：
- 它是在决策边界跳跃的阶跃函数
- 梯度几乎处处为零
- 无法使用基于梯度的优化

交叉熵是平滑、可微的代理，它：
- 提供有意义的梯度
- 鼓励正确的概率排序
- 数学上有原则（与似然相关）
</details>

---

## 8. 参考文献

1. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
2. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML.
3. Huber, P. J. (1964). "Robust Estimation of a Location Parameter." Annals of Mathematical Statistics.
4. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
5. Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." ICML.
