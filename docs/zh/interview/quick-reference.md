# 机器学习快速参考卡

面试前快速复习的基本公式和概念。

---

## 损失函数

| 损失 | 公式 | 用例 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | 回归 |
| MAE | $\frac{1}{n}\sum|y - \hat{y}|$ | 稳健回归 |
| 交叉熵 | $-\sum y \log \hat{y}$ | 分类 |
| Hinge | $\max(0, 1 - y \cdot \hat{y})$ | SVM |

---

## 正则化

| 类型 | 惩罚 | 效果 |
|------|------|------|
| L1 (Lasso) | $\lambda\sum|w_i|$ | 稀疏权重 |
| L2 (Ridge) | $\lambda\sum w_i^2$ | 小权重 |
| 弹性网络 | $\alpha L1 + (1-\alpha) L2$ | 组合 |

---

## 评估指标

### 分类

$$\text{精确率} = \frac{TP}{TP + FP}$$

$$\text{召回率} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

$$\text{准确率} = \frac{TP + TN}{总数}$$

### ROC vs PR 曲线

| 曲线 | X轴 | Y轴 | 使用场景 |
|------|-----|-----|----------|
| ROC | FPR | TPR | 平衡数据 |
| PR | 召回率 | 精确率 | 不平衡数据 |

---

## 模型公式

### 线性回归

**OLS**: $\hat{w} = (X^TX)^{-1}X^Ty$

**岭回归**: $\hat{w} = (X^TX + \lambda I)^{-1}X^Ty$

### 逻辑回归

$$P(y=1|x) = \sigma(w^Tx) = \frac{1}{1 + e^{-w^Tx}}$$

### SVM

**原问题**: $\min_w \frac{1}{2}||w||^2 + C\sum\xi_i$

约束: $y_i(w^Tx_i + b) \geq 1 - \xi_i$

### 决策树分裂

**Gini**: $1 - \sum_c p_c^2$

**熵**: $-\sum_c p_c \log_2 p_c$

**信息增益**: $IG = H(父) - \sum \frac{n_i}{n} H(子_i)$

### 梯度提升

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

其中$h_m$拟合损失的负梯度。

---

## 神经网络

### 激活函数

| 函数 | 公式 | 导数 |
|------|------|------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(1-\sigma)$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2$ |
| ReLU | $\max(0, x)$ | $\mathbb{1}_{x>0}$ |

### 反向传播

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}$$

链式法则贯穿各层。

### BatchNorm

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

### CNN输出尺寸

$$O = \frac{W - K + 2P}{S} + 1$$

W=输入，K=核，P=填充，S=步幅

---

## 降维

### PCA

1. 中心化数据: $X_c = X - \mu$
2. 协方差: $C = \frac{1}{n}X_c^TX_c$
3. 特征分解: $Cv = \lambda v$
4. 投影: $Z = X_c V_k$

**解释的方差**: $\frac{\lambda_k}{\sum_i \lambda_i}$

---

## 聚类

### K-Means目标

$$\min_{\mu} \sum_{i=1}^n \sum_{k=1}^K r_{ik} ||x_i - \mu_k||^2$$

### 轮廓系数

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

a(i) = 平均簇内距离
b(i) = 平均最近簇距离

---

## 概率模型

### 贝叶斯定理

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

### 朴素贝叶斯

$$P(y|x) \propto P(y) \prod_i P(x_i|y)$$

### 高斯分布

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

---

## 特征工程

### 标准化

$$z = \frac{x - \mu}{\sigma}$$

### Min-Max归一化

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### 目标编码（带平滑）

$$\text{编码} = \frac{n \cdot \bar{y}_{类别} + m \cdot \bar{y}_{全局}}{n + m}$$

---

## 偏差-方差

$$\text{误差} = \text{偏差}^2 + \text{方差} + \text{噪声}$$

| 模型复杂度 | 偏差 | 方差 |
|------------|------|------|
| 低（简单） | 高 | 低 |
| 高（复杂） | 低 | 高 |

---

## 类别不平衡

### SMOTE

$$x_{new} = x_i + \lambda(x_j - x_i), \quad \lambda \in [0,1]$$

### 类别权重

$$w_c = \frac{n_{总}}{n_c \cdot n_{类别数}}$$

---

## 优化

### 梯度下降更新

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

### Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

默认: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

---

## 需要记住的关键数值

| 概念 | 典型值 |
|------|--------|
| 学习率 | 1e-3 到 1e-4 |
| 批量大小 | 32, 64, 128 |
| Dropout率 | 0.2 到 0.5 |
| L2正则化 | 1e-4 到 1e-2 |
| 训练/验证/测试划分 | 70/15/15 或 80/10/10 |
| CV折数 | 5 或 10 |
| VIF阈值 | 5-10 |
| PSI阈值 | 0.1-0.25 |

---

## 常见陷阱检查清单

- [ ] 数据泄露（只在训练集上拟合）
- [ ] 类别不平衡（使用适当指标）
- [ ] 相关特征（检查VIF）
- [ ] 缺失值（正确填充）
- [ ] 特征缩放（许多算法需要）
- [ ] 随机种子（可复现性）
- [ ] 过拟合（检查训练vs验证差距）
- [ ] 训练-服务偏差（相同预处理）
