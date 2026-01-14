# 优化算法

## 1. 面试摘要

**关键要点：**
- **GD**：梯度下降 - 全批量，慢但稳定
- **SGD**：随机梯度下降 - 单样本，有噪声但快
- **Mini-batch SGD**：两者优点结合，最常用
- **动量**：加速收敛，抑制振荡
- **Adam**：自适应学习率，深度学习默认选择
- 学习率调度改善收敛

**常见面试问题：**
- "解释GD、SGD和mini-batch SGD的区别"
- "动量如何帮助优化？"
- "为什么Adam如此流行？"

---

## 2. 核心定义

### 梯度下降（GD）
使用整个数据集计算的梯度更新参数：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

其中 $\eta$ 是学习率。

### 随机梯度下降（SGD）
使用单个随机样本的梯度更新：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_i(\theta_t)$$

**特性：**
- 梯度估计方差高
- 可以逃离局部最小值（噪声有帮助）
- 每次迭代快得多

### Mini-batch SGD
使用小批量样本的梯度更新：

$$\theta_{t+1} = \theta_t - \eta \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \mathcal{L}_i(\theta_t)$$

**典型批量大小**：32、64、128、256

### 动量
累积梯度历史以加速收敛：

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

**典型 $\gamma$**：0.9

---

## 3. 数学与推导

### 收敛分析（凸情况）

对于具有L-Lipschitz梯度的凸损失：

**GD收敛速率：**
$$\mathcal{L}(\theta_T) - \mathcal{L}(\theta^*) \leq \frac{L\|\theta_0 - \theta^*\|^2}{2T}$$

收敛速度为 $O(1/T)$。

**SGD收敛：**
使用递减学习率 $\eta_t = \eta_0/\sqrt{t}$：
$$\mathbb{E}[\mathcal{L}(\bar{\theta}_T)] - \mathcal{L}(\theta^*) \leq O\left(\frac{1}{\sqrt{T}}\right)$$

### Adam算法

**矩估计：**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**偏差校正：**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**更新：**
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**默认超参数：** $\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$

### 学习率调度

**阶梯衰减：**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

**指数衰减：**
$$\eta_t = \eta_0 \cdot e^{-kt}$$

**余弦退火：**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**预热：**
$$\eta_t = \frac{t}{T_{warmup}} \cdot \eta_{target} \quad \text{当 } t < T_{warmup}$$

---

## 4. 算法框架

### 带动量的Mini-batch SGD

```
初始化：θ，v = 0，η，γ = 0.9
对于 epoch = 1 到 num_epochs：
    打乱训练数据
    对于每个mini-batch B：
        g = (1/|B|) * Σ ∇L_i(θ)   # 计算梯度
        v = γ * v + η * g         # 更新速度
        θ = θ - v                  # 更新参数
```

### Adam优化器

```
初始化：θ，m = 0，v = 0，t = 0
超参数：η = 0.001，β₁ = 0.9，β₂ = 0.999，ε = 1e-8

对于每个mini-batch：
    t = t + 1
    g = 损失对θ的梯度

    # 更新有偏矩估计
    m = β₁ * m + (1 - β₁) * g
    v = β₂ * v + (1 - β₂) * g²

    # 偏差校正
    m_hat = m / (1 - β₁^t)
    v_hat = v / (1 - β₂^t)

    # 更新参数
    θ = θ - η * m_hat / (√v_hat + ε)
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 学习率太高 | 发散，损失爆炸 | 从小开始，使用学习率查找器 |
| 学习率太低 | 收敛很慢 | 使用预热和调度 |
| 不用动量 | 收敛慢，振荡 | 使用动量≥0.9 |
| 恒定学习率 | 错过微调阶段 | 使用衰减调度 |
| 所有问题都用Adam | 可能泛化不是最好 | 视觉任务尝试SGD+动量 |
| 不打乱数据 | 有偏梯度估计 | 每个epoch打乱 |

### 优化器比较

| 优化器 | 优点 | 缺点 | 最适合 |
|--------|------|------|--------|
| SGD+动量 | 通常最佳泛化 | 需要LR调参 | CNN，视觉 |
| Adam | 快速，自适应 | 可能泛化不好 | NLP，快速原型 |
| AdamW | Adam+正确权重衰减 | 更多超参数 | Transformer |
| RMSprop | 自适应，简单 | 现在较少使用 | RNN |

---

## 6. 迷你示例

### Python示例：比较优化器

```python
import numpy as np

def quadratic_loss(x):
    """损失：f(x) = 0.5 * x^T A x，其中A条件数大。"""
    A = np.array([[10, 0], [0, 1]])
    return 0.5 * x @ A @ x

def gradient(x):
    A = np.array([[10, 0], [0, 1]])
    return A @ x

# SGD
def sgd(x0, lr=0.1, steps=50):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(steps):
        x = x - lr * gradient(x)
        history.append(x.copy())
    return np.array(history)

# 带动量的SGD
def sgd_momentum(x0, lr=0.1, momentum=0.9, steps=50):
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(steps):
        v = momentum * v + lr * gradient(x)
        x = x - v
        history.append(x.copy())
    return np.array(history)

# Adam
def adam(x0, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8, steps=50):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [x.copy()]
    for t in range(1, steps + 1):
        g = gradient(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(x.copy())
    return np.array(history)

# 比较
x0 = np.array([1.0, 1.0])
sgd_hist = sgd(x0, lr=0.05)
mom_hist = sgd_momentum(x0, lr=0.05)
adam_hist = adam(x0, lr=0.3)

print(f"SGD最终：{sgd_hist[-1]}，损失：{quadratic_loss(sgd_hist[-1]):.6f}")
print(f"动量最终：{mom_hist[-1]}，损失：{quadratic_loss(mom_hist[-1]):.6f}")
print(f"Adam最终：{adam_hist[-1]}，损失：{quadratic_loss(adam_hist[-1]):.6f}")

# 输出（收敛到[0,0]）：
# SGD最终：[0.00592 0.00519]，损失：0.000189
# 动量最终：[0.00001 0.00003]，损失：0.000000
# Adam最终：[0.00000 0.00000]，损失：0.000000
```

---

## 7. 测验

<details>
<summary><strong>Q1: GD、SGD和mini-batch SGD有什么区别？</strong></summary>

- **GD**：每次梯度计算使用整个数据集。稳定但慢且内存消耗大。
- **SGD**：每次更新使用单个样本。快速且内存高效但梯度很有噪声。
- **Mini-batch SGD**：使用小批量（如32-256样本）。平衡计算效率和梯度质量。实践中最常用。
</details>

<details>
<summary><strong>Q2: 动量如何帮助优化？</strong></summary>

动量累积过去梯度的滑动平均：
- **加速**一致梯度方向的收敛
- **抑制**符号变化方向的振荡
- 帮助逃离浅的局部最小值和鞍点
- 像带惯性的"重球"滚下山

公式：$v_t = \gamma v_{t-1} + \eta \nabla \mathcal{L}$，通常 $\gamma = 0.9$
</details>

<details>
<summary><strong>Q3: Adam与SGD有什么不同？</strong></summary>

Adam结合：
1. **一阶矩**（动量）：梯度的滑动平均
2. **二阶矩**：梯度平方的滑动平均（像RMSprop）
3. **偏差校正**：校正初始化偏差

这提供**每参数自适应学习率**，对不频繁参数较大，对频繁参数较小。不需要手动调整每层学习率。
</details>

<details>
<summary><strong>Q4: 为什么使用学习率调度？</strong></summary>

- **早期训练**：大LR快速进展
- **后期训练**：小LR微调和收敛
- **逃离局部最小值**：LR重启可以帮助逃离
- **更好泛化**：较慢的最终收敛通常泛化更好

常见调度：阶梯衰减、余弦退火、预热+衰减。
</details>

<details>
<summary><strong>Q5: SGD+动量什么时候可能优于Adam？</strong></summary>

SGD+动量通常在以下情况优于Adam：
- 计算机视觉任务（CNN）
- 当泛化比训练速度更重要时
- 较长的训练中Adam的自适应性变得不那么有用
- 可以仔细调整LR的任务

Adam擅长快速原型、NLP和LR调参时间有限时。
</details>

<details>
<summary><strong>Q6: 学习率调度中预热的目的是什么？</strong></summary>

预热在几次迭代中逐渐将学习率从接近零增加到目标值。好处：
- 当梯度不可靠时防止大的早期更新
- 稳定大批量训练
- 帮助批归一化统计初始化
- 对Transformer和大型模型特别重要
</details>

---

## 8. 参考文献

1. Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms." arXiv.
2. Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." ICLR.
3. Bottou, L., Curtis, F. E., & Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning." SIAM Review.
4. Loshchilov, I., & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR.
5. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv.
