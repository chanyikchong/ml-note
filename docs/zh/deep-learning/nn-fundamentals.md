# 神经网络基础

## 1. 面试摘要

**关键要点：**
- **MLP**：多层感知机 - 全连接前馈网络
- **反向传播**：通过链式法则高效计算梯度
- **激活函数**：ReLU（最常用）、sigmoid、tanh
- **通用近似**：足够宽的网络可以近似任何函数
- 知道前向传播、反向传播和梯度流

**常见面试问题：**
- "解释反向传播"
- "为什么需要非线性激活函数？"
- "什么是梯度消失问题？"

---

## 2. 核心定义

### 多层感知机（MLP）
前馈神经网络包含：
- 输入层
- 一个或多个隐藏层
- 输出层
- 全连接（密集）层

### 前向传播

$$h^{(l)} = \sigma(W^{(l)}h^{(l-1)} + b^{(l)})$$

### 反向传播
使用链式法则高效计算 $\frac{\partial \mathcal{L}}{\partial W}$：

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

### 常见激活函数

| 函数 | 公式 | 值域 | 导数 |
|------|------|------|------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | (0, 1) | $\sigma(z)(1-\sigma(z))$ |
| Tanh | $\tanh(z)$ | (-1, 1) | $1 - \tanh^2(z)$ |
| ReLU | $\max(0, z)$ | [0, ∞) | z > 0时为1，否则为0 |

---

## 3. 数学与推导

### 反向传播推导

对于损失 $\mathcal{L}$ 和层输出 $a^{(l)} = \sigma(z^{(l)})$，其中 $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$：

**输出层梯度：**

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}}$$

**隐藏层梯度（递归）：**

$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

**权重梯度：**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

### 通用近似定理

具有单个隐藏层和有限神经元的前馈网络，在对激活函数的温和假设下，可以在$\mathbb{R}^n$的紧子集上近似任何连续函数。

---

## 4. 算法框架

### 前向传播
```
输入：x，权重W，偏置b
a[0] = x

对于 l = 1 到 L：
    z[l] = W[l] @ a[l-1] + b[l]
    a[l] = activation(z[l])

输出：a[L]
```

### 反向传播
```
# 计算输出层delta
delta[L] = loss_gradient(a[L], y) * activation_derivative(z[L])

# 反向传播
对于 l = L-1 向下到 1：
    delta[l] = (W[l+1].T @ delta[l+1]) * activation_derivative(z[l])

# 计算梯度
对于 l = 1 到 L：
    dW[l] = delta[l] @ a[l-1].T
    db[l] = delta[l]
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 梯度消失 | Sigmoid/tanh饱和 | 使用ReLU，正确初始化 |
| 梯度爆炸 | 深层网络，大权重 | 梯度裁剪，归一化 |
| 死亡ReLU | 神经元卡在0 | LeakyReLU，正确初始化 |
| 不使用非线性 | 线性网络退化为单层 | 总是使用激活函数 |
| 错误初始化 | 初始权重太大/太小 | He/Xavier初始化 |

---

## 6. 迷你示例

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return z1, a1, z2

def backward(X, y, z1, a1, z2, W2):
    m = len(y)
    dz2 = z2 - y  # 假设MSE损失
    dW2 = (1/m) * a1.T @ dz2
    db2 = (1/m) * np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = (1/m) * X.T @ dz1
    db1 = (1/m) * np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2
```

---

## 7. 测验

<details>
<summary><strong>Q1: 为什么需要非线性激活函数？</strong></summary>

没有非线性激活，多层网络会退化为单个线性变换。线性函数的组合还是线性函数：
$f(x) = W_2(W_1 x) = (W_2 W_1)x = Wx$

非线性激活允许网络学习复杂的非线性决策边界。
</details>

<details>
<summary><strong>Q2: 解释梯度消失问题。</strong></summary>

使用sigmoid或tanh激活时，梯度乘以小于1的导数（如sigmoid'最大为0.25）。在深层网络中，这些相乘：

$\frac{\partial \mathcal{L}}{\partial W^{(1)}} \propto \prod_{l=1}^{L} \sigma'(z^{(l)})$

这个乘积指数级变小，使早期层权重更新非常慢。

**解决方案**：ReLU（导数为1）、跳跃连接、正确初始化。
</details>

<details>
<summary><strong>Q3: 反向传播中的链式法则是什么？</strong></summary>

反向传播使用链式法则分解梯度：

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}} \cdots \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

这允许每个样本$O(n)$时间内高效计算梯度。
</details>

<details>
<summary><strong>Q4: 比较ReLU、sigmoid和tanh激活。</strong></summary>

| 属性 | ReLU | Sigmoid | Tanh |
|------|------|---------|------|
| 值域 | [0, ∞) | (0, 1) | (-1, 1) |
| 梯度 | 1或0 | ≤ 0.25 | ≤ 1 |
| 零中心 | 否 | 否 | 是 |
| 梯度消失 | 较少 | 是 | 是 |
| 死亡神经元 | 是 | 否 | 否 |
| 速度 | 快 | 慢(exp) | 慢(exp) |
</details>

<details>
<summary><strong>Q5: 什么是Xavier/Glorot初始化？</strong></summary>

初始化权重为：

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

这使激活的方差在各层之间大致恒定，防止前向传播中信号消失/爆炸。

**He初始化**（用于ReLU）：$W \sim \mathcal{N}(0, \frac{2}{n_{in}})$
</details>

<details>
<summary><strong>Q6: 什么是通用近似定理？</strong></summary>

具有有限神经元和非多项式激活的单隐藏层网络可以在紧集上将任何连续函数近似到任意精度。

**注意事项**：
- 没有说需要多少神经元
- 没有说学习（优化）
- 仅紧集
- 实践中深层网络通常更高效
</details>

---

## 8. 参考文献

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Rumelhart, D., Hinton, G., & Williams, R. (1986). "Learning Representations by Back-propagating Errors." Nature.
3. Glorot, X., & Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." AISTATS.
4. He, K., et al. (2015). "Delving Deep into Rectifiers." ICCV.
5. Hornik, K. (1991). "Approximation Capabilities of Multilayer Feedforward Networks." Neural Networks.
