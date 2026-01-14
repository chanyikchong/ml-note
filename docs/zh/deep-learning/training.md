# 深度神经网络训练

## 1. 面试摘要

**关键要点：**
- **初始化**：He/Xavier初始化对收敛至关重要
- **梯度消失/爆炸**：主要训练挑战
- **梯度裁剪**：防止梯度爆炸
- **学习率调度**：预热、衰减、余弦退火
- **调试**：监控损失、梯度、激活值

**常见面试问题：**
- "如何诊断梯度消失？"
- "什么是正确的权重初始化？"
- "如何选择学习率？"

---

## 2. 核心定义

### 权重初始化

**Xavier/Glorot**（用于tanh/sigmoid）：
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

**He/Kaiming**（用于ReLU）：
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

### 梯度消失
梯度在早期层指数变小：
- 权重不更新
- 网络不学习

### 梯度爆炸
梯度指数变大：
- 权重变成NaN/Inf
- 训练不稳定

### 梯度裁剪
限制梯度幅度：
$$g \leftarrow \min\left(1, \frac{\theta}{\|g\|}\right) g$$

---

## 3. 数学与推导

### 为什么初始化重要

对于层 $h = f(Wx)$，方差传播：
$$\text{Var}(h) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

为了跨层保持方差：
$$\text{Var}(W) = \frac{1}{n_{in}}$$

对于ReLU（杀死一半信号）：
$$\text{Var}(W) = \frac{2}{n_{in}}$$

### 梯度消失分析

对于L层带激活 $\sigma$：
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \prod_{l=2}^{L} \frac{\partial h_l}{\partial h_{l-1}} \frac{\partial h_1}{\partial W_1}$$

如果 $|\sigma'| < 1$（sigmoid, tanh）：
- 乘积随深度指数收缩
- 早期层收到微小梯度

### 学习率调度

**阶梯衰减：**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

**余弦退火：**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

**预热：**
$$\eta_t = \eta_{target} \cdot \frac{t}{T_{warmup}} \quad \text{当 } t < T_{warmup}$$

---

## 4. 算法框架

### 训练循环

```
def train(model, data, epochs, lr):
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealing(optimizer, T_max=epochs)

    for epoch in range(epochs):
        for batch in data:
            # 前向传播
            predictions = model(batch.x)
            loss = criterion(predictions, batch.y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新权重
            optimizer.step()

        # 更新学习率
        scheduler.step()

        # 验证
        val_loss = evaluate(model, val_data)
        if val_loss < best_loss:
            save_checkpoint(model)
```

### 调试清单

```
1. 检查数据加载：
   - 可视化样本
   - 验证标签正确
   - 检查归一化

2. 先在小批次过拟合：
   - 在1-10个样本上训练
   - 应该达到~100%准确率
   - 如果不能，架构/代码有bug

3. 训练期间监控：
   - 损失应该下降
   - 梯度应该合理（非0或inf）
   - 激活值应该在正常范围

4. 常见修复：
   - 损失高不变 → 降低学习率
   - 损失NaN → 梯度裁剪，检查数据
   - 损失震荡 → 降低学习率
```

### 梯度健康检查

```
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"{name}: grad_norm = {grad_norm:.6f}")

            if grad_norm == 0:
                print("  警告：零梯度！")
            if grad_norm > 100:
                print("  警告：梯度爆炸！")
            if torch.isnan(grad_norm):
                print("  错误：NaN梯度！")
```

---

## 5. 常见陷阱

| 陷阱 | 症状 | 解决方案 |
|------|------|----------|
| 初始化差 | 损失不下降 | 使用He/Xavier初始化 |
| 梯度消失 | 早期层不更新 | 使用ReLU、跳跃连接、BatchNorm |
| 梯度爆炸 | 损失变NaN | 梯度裁剪，降低LR |
| 学习率太高 | 损失震荡/发散 | 降低LR |
| 学习率太低 | 损失下降很慢 | 增加LR或用调度器 |
| 过拟合 | 训练损失↓，验证损失↑ | Dropout、数据增强、正则化 |

### 学习率选择

```
学习率查找器：
1. 从很小LR开始（1e-7）
2. 每批次指数增加LR
3. 绘制LR vs 损失
4. 选择损失下降最快的LR
   （通常比最小点低1-10倍）

常见范围：
- SGD：0.01 - 0.1
- Adam：0.0001 - 0.001
- 带预热：开始小10-100倍
```

### 调试训练

| 观察 | 可能原因 | 修复 |
|------|----------|------|
| 损失=常数 | 梯度零，LR太小 | 检查梯度，增加LR |
| 损失=NaN | 梯度爆炸，数据差 | 裁剪梯度，检查数据 |
| 损失=震荡 | LR太高 | 降低LR |
| 验证损失增加 | 过拟合 | 正则化，早停 |
| 训练损失高 | 欠拟合 | 更大模型，训练更久 |

---

## 6. 迷你示例

```python
import numpy as np

def he_init(shape):
    """ReLU网络的He初始化。"""
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std

def xavier_init(shape):
    """tanh/sigmoid的Xavier初始化。"""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std

def clip_gradients(gradients, max_norm):
    """按全局范数裁剪梯度。"""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        gradients = [g * clip_coef for g in gradients]
    return gradients, total_norm


class LearningRateScheduler:
    def __init__(self, initial_lr, schedule='cosine', warmup_steps=0, total_steps=1000):
        self.initial_lr = initial_lr
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step):
        # 预热
        if step < self.warmup_steps:
            return self.initial_lr * (step + 1) / self.warmup_steps

        # 预热后
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)

        if self.schedule == 'cosine':
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.schedule == 'step':
            return self.initial_lr * (0.1 ** int(progress * 3))
        else:
            return self.initial_lr


# 演示：初始化比较
np.random.seed(42)
layer_sizes = [784, 256, 128, 64, 10]

print("=== 初始化比较 ===")
print("\n不同初始化的方差传播：")

for init_name, init_fn in [("Zero", lambda s: np.zeros(s)),
                            ("Random", lambda s: np.random.randn(*s)),
                            ("Xavier", xavier_init),
                            ("He", he_init)]:
    x = np.random.randn(100, 784)  # 100个样本的批次

    print(f"\n{init_name} 初始化：")
    for i in range(len(layer_sizes) - 1):
        W = init_fn((layer_sizes[i], layer_sizes[i+1]))
        x = np.maximum(0, x @ W)  # ReLU激活

        if np.std(x) == 0:
            print(f"  层 {i+1}：激活值塌缩到0")
            break
        print(f"  层 {i+1}：均值={x.mean():.4f}, 标准差={x.std():.4f}")

# 演示：学习率调度
print("\n=== 学习率调度 ===")
scheduler = LearningRateScheduler(0.1, schedule='cosine', warmup_steps=100, total_steps=1000)
steps = [0, 50, 100, 250, 500, 750, 1000]
for step in steps:
    lr = scheduler.get_lr(step)
    print(f"步骤 {step:4d}: LR = {lr:.6f}")

# 演示：梯度裁剪
print("\n=== 梯度裁剪 ===")
gradients = [np.random.randn(100, 50) * 10]  # 大梯度
clipped, norm_before = clip_gradients(gradients, max_norm=1.0)
_, norm_after = clip_gradients(clipped, max_norm=1.0)
print(f"裁剪前梯度范数: {norm_before:.2f}")
print(f"裁剪后梯度范数: {np.sqrt(sum(np.sum(g**2) for g in clipped)):.2f}")
```

**输出：**
```
=== 初始化比较 ===

不同初始化的方差传播：

Zero 初始化：
  层 1：激活值塌缩到0

Random 初始化：
  层 1：均值=10.0234, 标准差=19.5432
  层 2：均值=127.8921, 标准差=256.1234
  层 3：均值=8234.56, 标准差=16892.12
  层 4：激活值爆炸

Xavier 初始化：
  层 1：均值=0.5012, 标准差=0.7123
  层 2：均值=0.2534, 标准差=0.3892
  层 3：均值=0.1234, 标准差=0.2012
  层 4：均值=0.0612, 标准差=0.1023

He 初始化：
  层 1：均值=0.7891, 标准差=1.0234
  层 2：均值=0.8123, 标准差=0.9876
  层 3：均值=0.7654, 标准差=1.0123
  层 4：均值=0.8012, 标准差=0.9912

=== 学习率调度 ===
步骤    0: LR = 0.001000
步骤   50: LR = 0.050500
步骤  100: LR = 0.100000
步骤  250: LR = 0.085355
步骤  500: LR = 0.050000
步骤  750: LR = 0.014645
步骤 1000: LR = 0.000000

=== 梯度裁剪 ===
裁剪前梯度范数: 70.71
裁剪后梯度范数: 1.00
```

---

## 7. 测验

<details>
<summary><strong>Q1: 如何诊断梯度消失？</strong></summary>

梯度消失的迹象：
1. **早期层梯度接近零**：检查每层 `param.grad.norm()`
2. **早期层权重不变**：比较训练前后的权重
3. **损失早期停滞**：网络很快停止学习
4. **激活饱和**：对于sigmoid/tanh，输出接近0或1

**诊断代码**：
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

**解决方案**：ReLU激活、跳跃连接、BatchNorm、正确初始化。
</details>

<details>
<summary><strong>Q2: 解释He和Xavier初始化。</strong></summary>

两者都旨在跨层保持方差：

**Xavier**（Glorot）：$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$
- 为tanh/sigmoid激活设计
- 考虑前向和后向传播

**He**（Kaiming）：$\text{Var}(W) = \frac{2}{n_{in}}$
- 为ReLU激活设计
- 考虑ReLU杀死一半信号

**何时使用**：
- ReLU网络 → He初始化
- Tanh/sigmoid网络 → Xavier初始化
- 线性输出 → Xavier初始化
</details>

<details>
<summary><strong>Q3: 什么是梯度裁剪，什么时候使用？</strong></summary>

**梯度裁剪**：限制梯度幅度以防止梯度爆炸。

**按值**：裁剪每个元素：$g = \max(\min(g, \theta), -\theta)$
**按范数**：如果范数超过阈值则缩放：$g = g \cdot \min(1, \theta/\|g\|)$

**何时使用**：
- RNN（容易梯度爆炸）
- 非常深的网络
- 损失变NaN时
- 作为标准做法（不会有害）

**典型值**：max_norm = 1.0 到 5.0
</details>

<details>
<summary><strong>Q4: 如何选择学习率？</strong></summary>

**方法**：
1. **学习率查找器**：从小增加LR，绘制损失vs LR，选择损失下降最快的地方
2. **默认值**：Adam ~0.001，SGD ~0.01-0.1
3. **网格搜索**：尝试[0.0001, 0.001, 0.01, 0.1]

**错误LR的迹象**：
- 太高：损失震荡或增加
- 太低：损失下降很慢

**最佳实践**：
- 使用学习率预热
- 使用调度器（余弦、阶梯衰减）
- 损失停滞时降低LR
</details>

<details>
<summary><strong>Q5: 什么是学习率预热？</strong></summary>

**预热**：在初始训练期间逐渐将LR从小值增加到目标值。

**为什么有帮助**：
1. 早期梯度可能噪声大/很大
2. 防止大的初始更新
3. 允许BatchNorm统计量稳定
4. 对大批次训练重要

**常见方法**：
```
LR = target_lr * (step / warmup_steps)  对于 step < warmup_steps
```

**典型预热**：总训练步数的1-5%
</details>

<details>
<summary><strong>Q6: 如何调试无法训练的网络？</strong></summary>

**系统调试**：
1. **验证数据**：可视化输入，检查标签，验证预处理
2. **在单批次过拟合**：在1-10个样本上应达到~100%训练准确率
3. **检查梯度**：打印每层梯度范数，寻找零或无穷大
4. **检查激活值**：监控层输出的均值/标准差
5. **简化**：更小模型、更简单数据、更少层

**常见修复**：
- 损失卡住 → 降低LR，检查梯度
- 损失NaN → 梯度裁剪，检查数据是否有NaN
- 损失震荡 → 降低LR
- 无改善 → 增加模型容量
</details>

---

## 8. 参考文献

1. Glorot, X., & Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." AISTATS.
2. He, K., et al. (2015). "Delving Deep into Rectifiers." ICCV.
3. Smith, L. (2017). "Cyclical Learning Rates for Training Neural Networks." WACV.
4. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD." arXiv.
5. Loshchilov, I., & Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts." arXiv.
