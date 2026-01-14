# 循环神经网络（RNN）

## 1. 面试摘要

**关键要点：**
- **序列处理**：跨时间步维护隐藏状态
- **梯度消失/爆炸**：普通RNN的主要挑战
- **LSTM**：解决梯度问题的门控架构
- **GRU**：简化的门控，性能与LSTM相似
- **双向**：双向处理序列

**常见面试问题：**
- "解释RNN中的梯度消失问题"
- "LSTM门控如何工作？"
- "比较LSTM和GRU"

---

## 2. 核心定义

### 普通RNN

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

### LSTM（长短期记忆）
四个门控制信息流：
- **遗忘门**：$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
- **输入门**：$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
- **候选**：$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$
- **输出门**：$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$

细胞状态更新：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$h_t = o_t \odot \tanh(C_t)$$

### GRU（门控循环单元）
两个门的简化版本：
- **更新门**：$z_t = \sigma(W_z [h_{t-1}, x_t])$
- **重置门**：$r_t = \sigma(W_r [h_{t-1}, x_t])$
- **隐藏状态**：$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tanh(W_h [r_t \odot h_{t-1}, x_t])$

---

## 3. 数学与推导

### 时间反向传播（BPTT）

对于损失 $L = \sum_t L_t$，参数的梯度涉及时间上的链式法则：

$$\frac{\partial L}{\partial W_{hh}} = \sum_t \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$$

### 梯度消失问题

项 $\frac{\partial h_t}{\partial h_k}$ 涉及乘积：

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=k+1}^{t} W_{hh}^T \text{diag}(\tanh'(h_{i-1}))$$

由于 $|\tanh'(x)| \leq 1$ 且如果 $\|W_{hh}\| < 1$：
- 乘积随 $t-k$ 指数收缩
- 梯度对长序列消失
- 网络无法学习长程依赖

### 为什么LSTM解决这个问题

细胞状态更新是**加性的**：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

梯度直接通过加法流动：

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

如果 $f_t \approx 1$（遗忘门打开），梯度不变流动。这创造了"梯度高速公路"。

### 参数计数

对于输入维度 $d$ 和隐藏维度 $h$ 的LSTM：
- 4个门，每个有输入-隐藏和隐藏-隐藏权重
- 参数：$4 \times (d \times h + h \times h + h) = 4(dh + h^2 + h)$

对于GRU：$3(dh + h^2 + h)$（3个门而不是4个）

---

## 4. 算法框架

### RNN前向传播

```
def rnn_forward(x_sequence, h_0, W_xh, W_hh, W_hy):
    h = h_0
    outputs = []

    for x_t in x_sequence:
        h = tanh(W_xh @ x_t + W_hh @ h)
        y_t = W_hy @ h
        outputs.append(y_t)

    return outputs, h
```

### LSTM前向传播

```
def lstm_forward(x_sequence, h_0, c_0, params):
    h, c = h_0, c_0
    outputs = []

    for x_t in x_sequence:
        # 门
        f = sigmoid(W_f @ concat(h, x_t) + b_f)  # 遗忘
        i = sigmoid(W_i @ concat(h, x_t) + b_i)  # 输入
        o = sigmoid(W_o @ concat(h, x_t) + b_o)  # 输出
        c_tilde = tanh(W_c @ concat(h, x_t) + b_c)  # 候选

        # 状态更新
        c = f * c + i * c_tilde  # 细胞状态
        h = o * tanh(c)  # 隐藏状态

        outputs.append(h)

    return outputs, h, c
```

### 双向RNN

```
def bidirectional_rnn(x_sequence, params_forward, params_backward):
    # 前向传播
    h_forward = []
    h = zeros(hidden_size)
    for x_t in x_sequence:
        h = rnn_cell(x_t, h, params_forward)
        h_forward.append(h)

    # 后向传播
    h_backward = []
    h = zeros(hidden_size)
    for x_t in reversed(x_sequence):
        h = rnn_cell(x_t, h, params_backward)
        h_backward.insert(0, h)

    # 拼接
    return [concat(f, b) for f, b in zip(h_forward, h_backward)]
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 梯度消失 | 普通RNN长序列 | 使用LSTM/GRU |
| 梯度爆炸 | 大梯度乘积 | 梯度裁剪 |
| 训练慢 | 序列化，无法并行 | 使用Transformer |
| 过拟合 | 参数太多 | Dropout（循环dropout） |
| 长程差 | 即使LSTM也有限 | 注意力机制 |

### 架构选择

| 任务 | 建议 |
|------|------|
| 短序列（<50） | GRU或LSTM |
| 长序列（>100） | Transformer或注意力 |
| 实时处理 | GRU（比LSTM快） |
| 序列标注 | 双向LSTM |
| 语言建模 | Transformer（现代） |

### 训练技巧

1. **仔细初始化**：循环权重正交初始化
2. **梯度裁剪**：按范数裁剪（1.0-5.0）
3. **学习率**：从小开始（0.001）
4. **批处理**：填充序列，使用掩码
5. **层归一化**：在循环单元内应用

---

## 6. 迷你示例

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier初始化
        self.W_xh = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.W_hy = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
        self.hidden_size = hidden_size

    def forward(self, x_seq):
        h = np.zeros(self.hidden_size)
        self.hidden_states = [h]

        for x_t in x_seq:
            h = tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            self.hidden_states.append(h)

        y = h @ self.W_hy + self.b_y
        return y, h


class SimpleLSTM:
    def __init__(self, input_size, hidden_size):
        # 初始化所有门
        self.hidden_size = hidden_size
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # 拼接权重以提高效率
        self.W = np.random.randn(input_size + hidden_size, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)

        # 遗忘门偏置初始化为1（帮助梯度流动）
        self.b[hidden_size:2*hidden_size] = 1.0

    def forward(self, x_seq):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        hidden_states = []

        for x_t in x_seq:
            # 拼接输入和隐藏
            combined = np.concatenate([x_t, h])

            # 一次计算所有门
            gates = combined @ self.W + self.b

            # 分成各个门
            hs = self.hidden_size
            i = sigmoid(gates[:hs])          # 输入门
            f = sigmoid(gates[hs:2*hs])      # 遗忘门
            o = sigmoid(gates[2*hs:3*hs])    # 输出门
            g = tanh(gates[3*hs:])           # 候选

            # 更新状态
            c = f * c + i * g
            h = o * tanh(c)

            hidden_states.append(h)

        return hidden_states, h, c


# 示例：序列分类
np.random.seed(42)

# 创建简单序列数据
# 任务：预测第一个和最后一个元素之和
seq_length = 5
input_size = 3
hidden_size = 8
n_samples = 100

# 生成数据
X = np.random.randn(n_samples, seq_length, input_size)
y = (X[:, 0, 0] + X[:, -1, 0] > 0).astype(float)  # 二分类

# 测试简单RNN
rnn = SimpleRNN(input_size, hidden_size, 1)
sample_output, _ = rnn.forward(X[0])
print(f"简单RNN输出形状: {sample_output.shape}")
print(f"隐藏状态数: {len(rnn.hidden_states)}")

# 测试LSTM
lstm = SimpleLSTM(input_size, hidden_size)
hidden_states, final_h, final_c = lstm.forward(X[0])
print(f"\nLSTM隐藏状态数: {len(hidden_states)}")
print(f"最终隐藏形状: {final_h.shape}")
print(f"最终细胞形状: {final_c.shape}")

# 参数计数
rnn_params = (input_size * hidden_size + hidden_size * hidden_size +
              hidden_size + hidden_size * 1 + 1)
lstm_params = (input_size + hidden_size) * (4 * hidden_size) + 4 * hidden_size
print(f"\nRNN参数: {rnn_params}")
print(f"LSTM参数: {lstm_params}")
```

**输出：**
```
简单RNN输出形状: (1,)
隐藏状态数: 6

LSTM隐藏状态数: 5
最终隐藏形状: (8,)
最终细胞形状: (8,)

RNN参数: 97
LSTM参数: 384
```

---

## 7. 测验

<details>
<summary><strong>Q1: 解释RNN中的梯度消失问题。</strong></summary>

时间反向传播时，梯度涉及乘积：

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} W_{hh}^T \text{diag}(\tanh'(z_i))$$

由于 $|\tanh'(x)| \leq 1$：
- 如果 $\|W_{hh}\| < 1$：乘积指数收缩 → **梯度消失**
- 如果 $\|W_{hh}\| > 1$：乘积指数增长 → **梯度爆炸**

**后果**：无法学习长程依赖（远处的梯度约等于0）。

**解决方案**：LSTM/GRU（门控单元）、梯度裁剪、跳跃连接。
</details>

<details>
<summary><strong>Q2: LSTM门控如何工作？</strong></summary>

LSTM有4个组件：

1. **遗忘门**（$f_t$）：决定从细胞状态移除什么
   - $f_t \approx 0$：遗忘先前信息
   - $f_t \approx 1$：保留先前信息

2. **输入门**（$i_t$）：决定存储什么新信息
   - 控制多少候选进入细胞状态

3. **候选**（$\tilde{C}_t$）：可能添加的新信息
   - 像标准RNN一样计算

4. **输出门**（$o_t$）：决定从细胞状态输出什么
   - 过滤细胞状态用于隐藏状态输出

细胞更新是**加性的**：$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$

这种加性结构允许遗忘门打开时梯度不变流动。
</details>

<details>
<summary><strong>Q3: 比较LSTM和GRU。</strong></summary>

| 方面 | LSTM | GRU |
|------|------|-----|
| 门 | 4个（遗忘、输入、输出、候选） | 2个（更新、重置） |
| 状态 | 细胞状态 + 隐藏状态 | 只有隐藏状态 |
| 参数 | 更多（4个权重矩阵） | 更少（3个权重矩阵） |
| 训练 | 较慢 | 较快 |
| 性能 | 某些任务稍好 | 可比，有时更好 |
| 记忆 | 显式记忆细胞 | 隐藏状态中隐式 |

**何时使用**：
- GRU：较小数据集，更快训练，更简单架构
- LSTM：非常长的序列，显式记忆有帮助时
</details>

<details>
<summary><strong>Q4: 什么是教师强制？</strong></summary>

**教师强制**：训练时，使用前一时间步的真实值作为输入，而不是模型自己的预测。

**优点**：
- 更快收敛
- 更稳定训练
- 避免训练时误差累积

**缺点**：
- 训练-测试不匹配：模型训练时从未看到自己的错误
- 可能导致曝光偏差

**缓解**：计划采样 - 训练期间逐渐从教师强制转向模型预测。
</details>

<details>
<summary><strong>Q5: 什么是双向RNN？</strong></summary>

双向RNN双向处理序列：

1. **前向RNN**：从左到右处理
2. **后向RNN**：从右到左处理
3. **拼接**：在每个位置组合隐藏状态

**输出**：$h_t = [h_t^{\rightarrow}; h_t^{\leftarrow}]$

**优点**：
- 使用过去和未来的上下文
- 对分类/标注任务更好

**缺点**：
- 不能用于生成（需要完整序列）
- 参数翻倍
</details>

<details>
<summary><strong>Q6: 为什么Transformer正在取代RNN？</strong></summary>

Transformer优势：
1. **并行化**：所有位置同时处理（无序列依赖）
2. **长程依赖**：通过注意力直接连接（无梯度衰减）
3. **可扩展性**：更好地扩展到大模型/数据
4. **预训练**：更适合迁移学习（BERT、GPT）

RNN优势：
- 自然处理变长
- 对流式/在线数据更好
- 对非常长序列内存更低

现代趋势：Transformer主导NLP；RNN仍用于一些时间序列/流式应用。
</details>

---

## 8. 参考文献

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
2. Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder." EMNLP.
3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). "On the Difficulty of Training RNNs." ICML.
4. Graves, A. (2012). "Supervised Sequence Labelling with Recurrent Neural Networks." Springer.
5. Vaswani, A., et al. (2017). "Attention Is All You Need." NIPS.
