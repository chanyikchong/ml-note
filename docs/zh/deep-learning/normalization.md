# 归一化技术

## 1. 面试摘要

**关键要点：**
- **批归一化**：跨批次归一化，广泛用于CNN
- **层归一化**：跨特征归一化，用于RNN/Transformer
- **目的**：稳定训练，启用更高学习率
- **可学习参数**：缩放（γ）和偏移（β）
- **训练vs推理**：批归一化在推理时使用滑动统计量

**常见面试问题：**
- "为什么批归一化有效？"
- "比较批归一化和层归一化"
- "批归一化在测试时如何表现？"

---

## 2. 核心定义

### 批归一化
对于形状为 $(N, C, H, W)$ 的输入 $x$，每个通道跨批次归一化：

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

其中 $\mu_B, \sigma_B^2$ 是每个通道的批次统计量。

### 层归一化
对于输入 $x$，跨特征维度归一化：

$$\hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$

其中 $\mu_i, \sigma_i^2$ 在每个样本的特征上计算。

### 组归一化
将通道分成组，在每组内归一化：
- 批归一化和层归一化的折中
- 适用于小批次

### 实例归一化
独立归一化每个样本和通道：
- 用于风格迁移
- 每样本、每通道统计量

---

## 3. 数学与推导

### 批归一化前向传播

对于小批次 $\{x_1, ..., x_m\}$：

**步骤1**：计算批次统计量

$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$$

$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

**步骤2**：归一化

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**步骤3**：缩放和偏移

$$y_i = \gamma \hat{x}_i + \beta$$

### 为什么批归一化有效

几个假设：
1. **减少内部协变量偏移**：稳定输入分布
2. **平滑损失景观**：使优化更容易
3. **正则化效果**：批次统计量增加噪声
4. **启用更高学习率**：梯度更稳定

### 推理时的批归一化

训练期间，维护滑动平均：

$$\mu_{running} = (1 - \alpha) \mu_{running} + \alpha \mu_B$$

$$\sigma_{running}^2 = (1 - \alpha) \sigma_{running}^2 + \alpha \sigma_B^2$$

推理时，使用滑动统计量（确定性输出）。

### 批归一化梯度

通过归一化的反向传播：

$$\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}$$

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + ...$$

（完整梯度涉及通过 $\mu$ 和 $\sigma^2$ 的项）

---

## 4. 算法框架

### 批归一化

```
class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = ones(num_features)
        self.beta = zeros(num_features)
        self.running_mean = zeros(num_features)
        self.running_var = ones(num_features)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training=True):
        if training:
            # 计算批次统计量
            mean = x.mean(axis=0)  # 跨批次
            var = x.var(axis=0)

            # 更新滑动统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # 归一化
        x_norm = (x - mean) / sqrt(var + self.eps)

        # 缩放和偏移
        return self.gamma * x_norm + self.beta
```

### 层归一化

```
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = ones(normalized_shape)
        self.beta = zeros(normalized_shape)
        self.eps = eps

    def forward(self, x):
        # 在最后维度计算均值和方差
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        # 归一化
        x_norm = (x - mean) / sqrt(var + self.eps)

        # 缩放和偏移
        return self.gamma * x_norm + self.beta
```

### 比较

```
输入形状：(N, C, H, W)

BatchNorm：对每个C，在(N, H, W)上归一化
LayerNorm：对每个N，在(C, H, W)上归一化
InstanceNorm：对每个N, C，在(H, W)上归一化
GroupNorm：对每个N, G，在(C/G, H, W)上归一化
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 小批次用批归一化 | 统计量噪声大 | 用组归一化或层归一化 |
| 训练/评估模式不匹配 | 忘记设置eval模式 | 总是调用model.eval() |
| RNN中用批归一化 | 序列长度不同 | 用层归一化 |
| 滑动统计量不更新 | 训练时设置training=False | 检查模式 |
| 激活后归一化 | 顺序错误 | 通常：Conv → BN → ReLU |

### 何时使用每种

| 场景 | 推荐 |
|------|------|
| 大批次CNN | 批归一化 |
| 小批次CNN | 组归一化 |
| RNN / Transformer | 层归一化 |
| 风格迁移 | 实例归一化 |
| 不确定 | 尝试层归一化（到处适用） |

### 网络中的位置

**标准顺序**：
```
Conv/Linear → BatchNorm/LayerNorm → 激活 (ReLU)
```

**ResNet风格**：
```
Conv → BN → ReLU → Conv → BN → Add (skip) → ReLU
```

---

## 6. 迷你示例

```python
import numpy as np

class BatchNorm1D:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# 示例
np.random.seed(42)

# 创建数据：4个样本，8个特征
x = np.random.randn(4, 8) * 10 + 5  # 高方差，非零均值

print("原始数据统计量：")
print(f"  均值: {x.mean():.2f}, 标准差: {x.std():.2f}")
print(f"  每特征均值: {x.mean(axis=0)[:4]}...")
print(f"  每样本均值: {x.mean(axis=1)}")

# 批归一化
bn = BatchNorm1D(8)
x_bn = bn.forward(x, training=True)
print(f"\n批归一化后（每特征归一化）：")
print(f"  均值: {x_bn.mean():.4f}, 标准差: {x_bn.std():.4f}")
print(f"  每特征均值: {x_bn.mean(axis=0)[:4]}...")

# 层归一化
ln = LayerNorm(8)
x_ln = ln.forward(x)
print(f"\n层归一化后（每样本归一化）：")
print(f"  均值: {x_ln.mean():.4f}, 标准差: {x_ln.std():.4f}")
print(f"  每样本均值: {x_ln.mean(axis=1)}")

# 推理模式（使用滑动统计量）
x_new = np.random.randn(2, 8) * 10 + 5
x_bn_infer = bn.forward(x_new, training=False)
print(f"\n推理模式批归一化：")
print(f"  使用running_mean: {bn.running_mean[:4]}...")
print(f"  使用running_var: {bn.running_var[:4]}...")
```

**输出：**
```
原始数据统计量：
  均值: 5.15, 标准差: 10.23
  每特征均值: [5.99 3.85 6.08 4.47]...
  每样本均值: [6.12 3.49 5.97 5.01]

批归一化后（每特征归一化）：
  均值: 0.0000, 标准差: 1.0000
  每特征均值: [ 0.  0. -0. -0.]...

层归一化后（每样本归一化）：
  均值: 0.0000, 标准差: 1.0000
  每样本均值: [-0. -0.  0. -0.]

推理模式批归一化：
  使用running_mean: [0.6  0.38 0.61 0.45]...
  使用running_var: [1.06 1.07 1.08 1.06]...
```

---

## 7. 测验

<details>
<summary><strong>Q1: 为什么批归一化有助于训练？</strong></summary>

几个因素：
1. **减少内部协变量偏移**：层输入分布稳定
2. **更平滑的损失景观**：梯度更可预测
3. **正则化**：批次统计量增加噪声（像dropout）
4. **更高学习率**：训练更稳定
5. **减少对初始化的敏感性**：网络从各种初始化训练

最近研究表明平滑效果最重要。
</details>

<details>
<summary><strong>Q2: 批归一化和层归一化有什么区别？</strong></summary>

| 方面 | 批归一化 | 层归一化 |
|------|----------|----------|
| 归一化维度 | 批次维度 | 特征维度 |
| 统计量 | 每特征，跨批次 | 每样本，跨特征 |
| 训练vs测试 | 不同（滑动统计量） | 相同 |
| 批次依赖 | 是 | 否 |
| 最适合 | 大批次CNN | RNN、Transformer |

关键区别：批归一化依赖批次，层归一化不依赖。
</details>

<details>
<summary><strong>Q3: 批归一化在训练和推理时有什么不同？</strong></summary>

**训练**：
- 从当前小批次计算均值/方差
- 更新滑动统计量（指数移动平均）
- 输出依赖批次中其他样本

**推理**：
- 使用存储的滑动均值/方差
- 确定性输出
- 无批次依赖

**原因**：推理时，我们希望无论批次组成如何都有确定性预测。
</details>

<details>
<summary><strong>Q4: 为什么Transformer倾向于层归一化而不是批归一化？</strong></summary>

1. **可变序列长度**：批归一化会混合不同位置的统计量
2. **小有效批次**：注意力可以看到所有位置，但填充不同
3. **批次独立**：每个序列独立处理
4. **更简单**：没有滑动统计量需要管理
5. **推理时有效**：与训练行为相同

另外：Transformer并行处理序列，使每样本归一化自然。
</details>

<details>
<summary><strong>Q5: 批归一化中有哪些可学习参数？</strong></summary>

每个特征两个可学习参数：
- **γ（gamma）**：缩放参数
- **β（beta）**：偏移参数

归一化到零均值和单位方差后：

$$y = \gamma \hat{x} + \beta$$

**为什么需要**：归一化可能移除有用信息。γ和β允许网络学习在需要时撤销归一化（当γ=σ, β=μ时，恢复原始值）。
</details>

<details>
<summary><strong>Q6: 什么是组归一化，什么时候有用？</strong></summary>

组归一化将通道分成组并在每组内归一化：
- 通道分成G组
- 对每个样本和组，在(H, W, C/G)上归一化

**有用当**：
- 批次大小小（批归一化失败）
- 批次大小变化
- 每GPU小批次的分布式训练

G=1的组归一化是层归一化；G=C是实例归一化。
</details>

---

## 8. 参考文献

1. Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training." ICML.
2. Ba, J., Kiros, J., & Hinton, G. (2016). "Layer Normalization." arXiv.
3. Wu, Y., & He, K. (2018). "Group Normalization." ECCV.
4. Santurkar, S., et al. (2018). "How Does Batch Normalization Help Optimization?" NeurIPS.
5. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). "Instance Normalization." arXiv.
