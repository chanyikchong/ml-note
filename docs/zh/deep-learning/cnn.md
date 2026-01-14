# 卷积神经网络（CNN）

## 1. 面试摘要

**关键要点：**
- **卷积**：滤波器在输入上滑动，计算点积
- **池化**：下采样，提供平移不变性
- **关键超参数**：滤波器大小、步幅、填充、滤波器数量
- **架构模式**：Conv → ReLU → Pool → ... → Flatten → FC
- **参数共享**：同一滤波器应用于整个输入

**常见面试问题：**
- "什么是感受野？"
- "计算卷积层的输出维度"
- "为什么用卷积而不是全连接？"

---

## 2. 核心定义

### 卷积操作
对于输入 $I$ 和核 $K$：

$$(I * K)[i,j] = \sum_m \sum_n I[i+m, j+n] \cdot K[m, n]$$

### 输出维度

$$\text{输出尺寸} = \frac{W - K + 2P}{S} + 1$$
- $W$：输入大小
- $K$：核大小
- $P$：填充
- $S$：步幅

### 感受野
输入中影响特定输出神经元的区域。
- 随深度增长
- 对于L层核K：$RF = 1 + L \cdot (K - 1)$

### 常见层

| 层 | 目的 | 参数 |
|----|------|------|
| Conv2D | 特征提取 | 核权重 |
| MaxPool | 下采样、不变性 | 无 |
| BatchNorm | 归一化激活 | γ, β |
| Dropout | 正则化 | 无（概率） |
| Flatten | 重塑用于FC | 无 |

---

## 3. 数学与推导

### 卷积前向传播

对于形状为 $(H, W, C_{in})$ 的输入 $X$ 和形状为 $(K, K, C_{in}, C_{out})$ 的滤波器 $W$：

$$Y[i, j, k] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X[i \cdot s + m, j \cdot s + n, c] \cdot W[m, n, c, k] + b[k]$$

### 参数计数

对于具有 $C_{in}$ 输入通道、$C_{out}$ 输出通道、核 $K$ 的Conv2D：

$$\text{参数} = K \times K \times C_{in} \times C_{out} + C_{out}$$

### 为什么卷积有效

1. **局部连接**：每个输出依赖局部区域
2. **参数共享**：空间维度上相同权重
3. **平移等变性**：移动输入则输出移动

与FC比较：
- 224×224×3图像到1000输出的FC：1.5亿参数
- Conv：到处相同核，参数少得多

### 通过Conv的反向传播

损失对核的梯度：

$$\frac{\partial L}{\partial W} = X * \frac{\partial L}{\partial Y}$$

输入与上游梯度的完整卷积。

---

## 4. 算法框架

### 基本CNN架构

```
输入：图像 (H × W × C)

# 特征提取
for layer in [conv1, conv2, conv3, ...]：
    x = Conv2D(x, filters, kernel_size)
    x = BatchNorm(x)
    x = ReLU(x)
    x = MaxPool(x, pool_size)

# 分类头
x = Flatten(x)
x = Dense(x, hidden_units)
x = ReLU(x)
x = Dropout(x, rate)
x = Dense(x, num_classes)
output = Softmax(x)
```

### 卷积实现

```
def conv2d(input, kernel, stride=1, padding=0):
    H, W, C_in = input.shape
    K, _, _, C_out = kernel.shape

    # 添加填充
    input_padded = pad(input, padding)

    # 输出维度
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1
    output = zeros(H_out, W_out, C_out)

    # 卷积
    for i in range(H_out):
        for j in range(W_out):
            for k in range(C_out):
                h_start = i * stride
                w_start = j * stride
                region = input_padded[h_start:h_start+K, w_start:w_start+K, :]
                output[i, j, k] = sum(region * kernel[:, :, :, k])

    return output
```

### 池化实现

```
def maxpool2d(input, pool_size=2, stride=2):
    H, W, C = input.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = zeros(H_out, W_out, C)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            region = input[h_start:h_start+pool_size, w_start:w_start+pool_size, :]
            output[i, j, :] = max(region, axis=(0, 1))

    return output
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 维度不匹配 | 填充/步幅错误 | 仔细计算输出大小 |
| 参数太多 | 深FC层 | 使用全局平均池化 |
| 过拟合 | 小数据集 | 数据增强、dropout |
| 梯度消失 | 深网络 | 用跳跃连接、BatchNorm |
| 训练慢 | 大图像 | 渐进式调整大小 |

### 架构指南

| 数据规模 | 建议 |
|----------|------|
| 小（<1万张图） | 用预训练，微调 |
| 中（1万-10万） | 从头训练，带增强 |
| 大（>10万） | 深度自定义架构 |

### 常见架构

| 网络 | 关键创新 | 深度 |
|------|----------|------|
| LeNet | 基本CNN | 5 |
| AlexNet | ReLU、Dropout、GPU | 8 |
| VGG | 只用3×3滤波器 | 16-19 |
| ResNet | 跳跃连接 | 50-152 |
| Inception | 多尺度滤波器 | 22+ |

---

## 6. 迷你示例

```python
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # 初始化权重（He初始化）
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.W = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * scale
        self.b = np.zeros(out_channels)

    def forward(self, x):
        self.x = x
        N, H, W, C = x.shape
        K = self.kernel_size

        # 填充输入
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding),
                          (self.padding, self.padding), (0, 0)), mode='constant')

        # 输出维度
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1
        out = np.zeros((N, H_out, W_out, self.W.shape[3]))

        # 卷积
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                x_slice = x[:, h_start:h_start+K, w_start:w_start+K, :]
                out[:, i, j, :] = np.tensordot(x_slice, self.W, axes=([1,2,3], [0,1,2])) + self.b

        return out


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        N, H, W, C = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        out = np.zeros((N, H_out, W_out, C))

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                out[:, i, j, :] = np.max(
                    x[:, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, :],
                    axis=(1, 2)
                )

        return out


def relu(x):
    return np.maximum(0, x)


# 简单CNN前向传播
np.random.seed(42)

# 创建样本"图像"：批次2，8x8，1通道
x = np.random.randn(2, 8, 8, 1)

# 构建简单CNN
conv1 = Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1)
pool1 = MaxPool2D(pool_size=2, stride=2)
conv2 = Conv2D(in_channels=4, out_channels=8, kernel_size=3, padding=1)
pool2 = MaxPool2D(pool_size=2, stride=2)

# 前向传播
print(f"输入形状: {x.shape}")
x = conv1.forward(x)
print(f"Conv1后: {x.shape}")
x = relu(x)
x = pool1.forward(x)
print(f"Pool1后: {x.shape}")
x = conv2.forward(x)
print(f"Conv2后: {x.shape}")
x = relu(x)
x = pool2.forward(x)
print(f"Pool2后: {x.shape}")

# 展平用于FC
x_flat = x.reshape(x.shape[0], -1)
print(f"展平后: {x_flat.shape}")

# 参数计数
params = (3*3*1*4 + 4) + (3*3*4*8 + 8)
print(f"\n总卷积参数: {params}")
```

**输出：**
```
输入形状: (2, 8, 8, 1)
Conv1后: (2, 8, 8, 4)
Pool1后: (2, 4, 4, 4)
Conv2后: (2, 4, 4, 8)
Pool2后: (2, 2, 2, 8)
展平后: (2, 32)

总卷积参数: 332
```

---

## 7. 测验

<details>
<summary><strong>Q1: 计算卷积层的输出维度。</strong></summary>

对于输入 $(H, W)$，核 $K$，步幅 $S$，填充 $P$：

$$H_{out} = \lfloor\frac{H + 2P - K}{S}\rfloor + 1$$

**例子**：输入32×32，核5×5，步幅1，填充2：

$$H_{out} = \frac{32 + 2(2) - 5}{1} + 1 = \frac{31}{1} + 1 = 32$$

对于"same"填充（输出=输入）：$P = \frac{K-1}{2}$（当S=1时）
</details>

<details>
<summary><strong>Q2: 什么是感受野，为什么重要？</strong></summary>

**感受野**：影响特定输出神经元的输入区域。

**为什么重要**：
- 决定每个神经元"看到"什么上下文
- 深层有更大感受野
- 需要足够大的RF来捕获相关模式

**计算**：对于L层步幅1的核K：

$$RF = 1 + L \times (K - 1)$$

例子：3层3×3卷积：RF = 1 + 3×2 = 7×7
</details>

<details>
<summary><strong>Q3: 为什么对图像用卷积而不是全连接层？</strong></summary>

1. **参数效率**：同一核空间重用
   - 224×224×3 → 1000的FC：1.5亿参数
   - 64个滤波器的3×3卷积：1,728参数

2. **平移等变性**：无论位置都能检测到对象

3. **局部连接**：利用空间结构

4. **层次特征**：随深度从低级到高级

5. **更好泛化**：更少参数 = 更少过拟合
</details>

<details>
<summary><strong>Q4: 池化层的目的是什么？</strong></summary>

池化（通常最大或平均）：

1. **下采样**：减少空间维度
2. **平移不变性**：小位移不改变输出
3. **减少计算**：后续层更少参数
4. **增加感受野**：比堆叠卷积更快

**最大池化**最常见：取每个区域的最大值，保留最强激活。

**趋势**：现代架构常用步幅卷积代替。
</details>

<details>
<summary><strong>Q5: Conv2D层有多少参数？</strong></summary>

对于Conv2D：
- 输入通道：$C_{in}$
- 输出通道：$C_{out}$
- 核大小：$K \times K$

**参数**：$K \times K \times C_{in} \times C_{out} + C_{out}$（偏置）

**例子**：3×3卷积，64输入通道，128输出通道：

$$3 \times 3 \times 64 \times 128 + 128 = 73,856$$

注意：无论输入空间大小参数相同（参数共享）。
</details>

<details>
<summary><strong>Q6: 什么是1×1卷积，什么时候使用？</strong></summary>

1×1卷积：核大小1，作为逐像素全连接层。

**用途**：
1. **通道减少**：在昂贵3×3卷积前减少通道（瓶颈）
2. **添加非线性**：1×1卷积 + ReLU增加容量
3. **跨通道交互**：混合通道间信息
4. **网络中网络**：廉价创建更深网络

在Inception和ResNet架构中大量使用。
</details>

---

## 8. 参考文献

1. LeCun, Y., et al. (1998). "Gradient-Based Learning Applied to Document Recognition." Proc. IEEE.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). "ImageNet Classification with Deep CNNs." NIPS.
3. Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks." arXiv.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
5. Szegedy, C., et al. (2015). "Going Deeper with Convolutions." CVPR.
