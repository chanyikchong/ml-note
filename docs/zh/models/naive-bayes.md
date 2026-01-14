# 朴素贝叶斯

## 1. 面试摘要

**关键要点：**
- **贝叶斯定理**：$P(y|x) \propto P(x|y)P(y)$
- **朴素假设**：给定类别，特征条件独立
- **生成模型**：建模 $P(x|y)$ 而非直接建模 $P(y|x)$
- **快速简单**：训练是计数，预测是乘法
- **适用于**：文本分类、垃圾邮件过滤、高维稀疏数据

**常见面试问题：**
- "什么是朴素假设，它何时失效？"
- "为什么朴素贝叶斯尽管有强假设仍然效果好？"
- "比较高斯、多项式和伯努利朴素贝叶斯"

---

## 2. 核心定义

### 贝叶斯定理

$$P(y|x) = \frac{P(x|y)P(y)}{P(x)}$$

对于分类，我们需要：

$$\hat{y} = \arg\max_y P(y|x) = \arg\max_y P(x|y)P(y)$$

### 朴素独立性假设
假设给定类别，特征条件独立：

$$P(x_1, x_2, ..., x_d | y) = \prod_{j=1}^{d} P(x_j | y)$$

### 朴素贝叶斯类型

| 类型 | 似然模型 | 适用场景 |
|------|----------|----------|
| 高斯 | $P(x_j|y) = \mathcal{N}(\mu_{jy}, \sigma_{jy}^2)$ | 连续特征 |
| 多项式 | $P(x_j|y) = \theta_{jy}^{x_j}$ | 词频、计数 |
| 伯努利 | $P(x_j|y) = \theta_{jy}^{x_j}(1-\theta_{jy})^{1-x_j}$ | 二值特征 |

---

## 3. 数学与推导

### 完整推导

给定训练数据 $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$：

**步骤1**：估计类别先验

$$P(y = c) = \frac{\text{count}(y = c)}{n}$$

**步骤2**：估计似然（高斯为例）

$$\mu_{jc} = \frac{1}{n_c}\sum_{i: y^{(i)}=c} x_j^{(i)}$$

$$\sigma_{jc}^2 = \frac{1}{n_c}\sum_{i: y^{(i)}=c} (x_j^{(i)} - \mu_{jc})^2$$

**步骤3**：预测

$$\hat{y} = \arg\max_c \left[ \log P(y=c) + \sum_{j=1}^d \log P(x_j | y=c) \right]$$

### 多项式朴素贝叶斯（用于文本）

对于词频向量 $x = (x_1, ..., x_V)$ 的文档：

$$P(x|y=c) \propto \prod_{j=1}^{V} \theta_{jc}^{x_j}$$

其中 $\theta_{jc} = P(\text{词 } j | \text{类别 } c)$

**最大似然估计**：

$$\theta_{jc} = \frac{\text{count}(j, c)}{\sum_{k=1}^V \text{count}(k, c)}$$

### 拉普拉斯平滑

问题：如果某词从未在某类中出现，$P(x_j|y) = 0$，使整个乘积为0。

解决方案：添加伪计数 $\alpha$（通常为1）：

$$\theta_{jc} = \frac{\text{count}(j, c) + \alpha}{\sum_{k=1}^V \text{count}(k, c) + \alpha V}$$

### 对数空间计算

为避免多个小概率相乘导致下溢：

$$\log P(y|x) = \log P(y) + \sum_j \log P(x_j|y) - \log P(x)$$

由于 $P(x)$ 对所有类别相同，我们比较：

$$\arg\max_y \left[ \log P(y) + \sum_j \log P(x_j|y) \right]$$

---

## 4. 算法框架

### 训练

```
输入：训练数据 (X, y)
输出：模型参数

对于每个类别 c：
    # 先验
    prior[c] = count(y == c) / n

    # 似然（高斯）
    对于每个特征 j：
        mean[j, c] = mean(X[y == c, j])
        var[j, c] = variance(X[y == c, j])
```

### 预测

```
输入：新点 x
输出：预测类别

对于每个类别 c：
    log_prob[c] = log(prior[c])

    对于每个特征 j：
        log_prob[c] += log(P(x[j] | c))
        # 高斯：log(N(x[j]; mean[j,c], var[j,c]))

返回 argmax(log_prob)
```

### 文本分类流程

```
1. 预处理文本：分词、小写、去停用词
2. 构建词表：词到索引的映射
3. 将文档转换为计数向量
4. 用拉普拉斯平滑训练多项式NB
5. 预测：计算每个类别的对数概率
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 零概率 | 未见过的特征值 | 使用拉普拉斯平滑 |
| 数值下溢 | 多个小数相乘 | 使用对数概率 |
| 错误的NB变体 | 对计数用高斯 | 匹配变体和数据类型 |
| 相关特征 | 违反独立假设 | 通常仍有效；即使相关也可尝试 |
| 类别不平衡 | 先验主导预测 | 考虑平衡先验 |

### 朴素假设何时失效

当特征相关时假设失效：
- "热"和"狗"在"热狗"中（词对）
- 图像中的相邻像素
- 有自相关的时间序列

**为什么通常仍然有效：**
- 分类只需正确排序，不需校准的概率
- 概率估计的误差可能相互抵消
- 即使模型错误，决策边界也可能正确

---

## 6. 迷你示例

```python
import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        # 计算先验和似然
        self.priors = {}
        self.means = {}
        self.vars = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0) + 1e-9  # 添加小值保持稳定

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        log_probs = {}
        for c in self.classes:
            # 对数先验
            log_prob = np.log(self.priors[c])
            # 对数似然（高斯）
            log_prob += np.sum(-0.5 * np.log(2 * np.pi * self.vars[c])
                               - 0.5 * (x - self.means[c])**2 / self.vars[c])
            log_probs[c] = log_prob
        return max(log_probs, key=log_probs.get)

# 示例
np.random.seed(42)

# 生成数据：两个高斯类别
X0 = np.random.randn(50, 2) + np.array([0, 0])
X1 = np.random.randn(50, 2) + np.array([3, 3])
X = np.vstack([X0, X1])
y = np.array([0] * 50 + [1] * 50)

# 训练和预测
nb = GaussianNB()
nb.fit(X, y)

# 测试
X_test = np.array([[0, 0], [3, 3], [1.5, 1.5]])
predictions = nb.predict(X_test)
print(f"测试点: {X_test.tolist()}")
print(f"预测: {predictions}")

# 训练准确率
train_pred = nb.predict(X)
accuracy = np.mean(train_pred == y)
print(f"训练准确率: {accuracy:.3f}")
```

**输出：**
```
测试点: [[0, 0], [3, 3], [1.5, 1.5]]
预测: [0 1 1]
训练准确率: 0.970
```

---

## 7. 测验

<details>
<summary><strong>Q1: 朴素贝叶斯中的"朴素"假设是什么？</strong></summary>

朴素假设是所有特征在给定类别标签时**条件独立**：

$$P(x_1, x_2, ..., x_d | y) = \prod_{j=1}^{d} P(x_j | y)$$

这允许我们分别估计每个 $P(x_j|y)$，使训练简单快速。没有这个假设，我们需要估计完整的联合分布，需要指数级更多的数据。
</details>

<details>
<summary><strong>Q2: 为什么朴素贝叶斯尽管独立假设被违反仍然效果好？</strong></summary>

几个原因：
1. **分类只需排序**：我们不需要准确的概率，只需正确的 $\arg\max$
2. **误差抵消**：高估和低估可能相互平衡
3. **决策边界**：即使概率错误，决策边界也可能正确
4. **正则化效果**：强假设起到正则化作用，减少方差
5. **高维成功**：在高维中，假设变得不那么有害
</details>

<details>
<summary><strong>Q3: 什么时候用多项式vs高斯vs伯努利朴素贝叶斯？</strong></summary>

- **多项式NB**：词频、文档分类、任何计数数据
- **高斯NB**：连续实值特征，假设正态分布
- **伯努利NB**：二值特征（存在/不存在）、二值文本分类

例子：
- 垃圾邮件（词频）→ 多项式
- 鸢尾花分类（连续）→ 高斯
- 二值特征（has_link, has_attachment）→ 伯努利
</details>

<details>
<summary><strong>Q4: 什么是拉普拉斯平滑，为什么需要它？</strong></summary>

**问题**：如果某特征值在训练中从未与某类别出现，$P(x_j|y) = 0$，使整个乘积为0。

**解决方案**：添加伪计数 $\alpha$（通常为1）：

$$\theta_{jc} = \frac{\text{count}(j, c) + \alpha}{\text{total count}(c) + \alpha \cdot V}$$

这确保没有概率恰好为0。也叫"加法平滑"或"加一平滑"。
</details>

<details>
<summary><strong>Q5: 朴素贝叶斯是判别模型还是生成模型？</strong></summary>

朴素贝叶斯是**生成**模型：
- 建模联合分布 $P(x, y) = P(x|y)P(y)$
- 学习每个类别的数据如何"生成"
- 可以生成新样本（采样 $y$，然后采样 $x|y$）

与**判别**模型对比（逻辑回归、SVM）：
- 直接建模 $P(y|x)$
- 不建模特征如何分布
- 当假设错误时分类通常更好
</details>

<details>
<summary><strong>Q6: 朴素贝叶斯如何处理连续特征？</strong></summary>

选项：
1. **高斯NB**：假设每个特征在每个类别中服从正态分布
2. **离散化**：将连续值分箱成类别
3. **核密度估计**：非参数密度估计

高斯NB最常见：

$$P(x_j|y=c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)$$
</details>

---

## 8. 参考文献

1. Mitchell, T. (1997). *Machine Learning*. 第6章：贝叶斯学习.
2. Manning, C., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. 第13章.
3. McCallum, A., & Nigam, K. (1998). "A Comparison of Event Models for Naive Bayes Text Classification." AAAI Workshop.
4. Zhang, H. (2004). "The Optimality of Naive Bayes." FLAIRS.
5. Ng, A., & Jordan, M. (2002). "On Discriminative vs. Generative Classifiers." NIPS.
