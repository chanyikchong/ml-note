# 学习范式

## 1. 面试摘要

**关键要点：**
- **监督学习**：从标注数据中学习；预测新输入的输出
- **无监督学习**：在无标注数据中发现模式；没有目标变量
- **自监督学习**：从数据结构本身创建伪标签
- 知道每种范式的例子和使用场景
- 理解每种范式的数据需求

**常见面试问题：**
- "监督学习和无监督学习有什么区别？"
- "给出一个自监督学习的例子"
- "什么时候使用无监督方法而不是监督方法？"

---

## 2. 核心定义

### 监督学习
从标注训练数据 $\{(x_i, y_i)\}_{i=1}^n$ 学习映射 $f: X \rightarrow Y$。

**特点：**
- 需要标注数据（输入-输出对）
- 目标：最小化对未见数据的预测误差
- 类型：分类（离散 $Y$），回归（连续 $Y$）

**例子：**
- 垃圾邮件检测（分类）
- 房价预测（回归）
- 图像分类（分类）

### 无监督学习
从无标注数据 $\{x_i\}_{i=1}^n$ 中学习模式，没有目标变量。

**特点：**
- 不需要标签
- 目标：发现结构、模式或表示
- 类型：聚类、降维、密度估计

**例子：**
- 客户细分（聚类）
- 异常检测（密度估计）
- PCA特征提取（降维）

### 自监督学习
从数据本身创建监督信号，然后学习表示。

**特点：**
- 自动从数据结构生成伪标签
- 监督学习和无监督学习之间的桥梁
- 对表示学习非常有效

**例子：**
- 语言模型预测下一个词（NLP）
- 对比学习（计算机视觉）
- 掩码自编码（BERT, MAE）

---

## 3. 数学与推导

### 监督学习形式化

给定数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$，其中 $x_i \in \mathcal{X}$，$y_i \in \mathcal{Y}$。

**经验风险最小化（ERM）：**
$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(x_i), y_i)$$

其中 $\mathcal{L}$ 是损失函数，$\mathcal{F}$ 是假设类。

**真实风险（泛化误差）：**
$$R(f) = \mathbb{E}_{(x,y) \sim P}[\mathcal{L}(f(x), y)]$$

### 无监督学习形式化

给定数据集 $\mathcal{D} = \{x_i\}_{i=1}^n$，其中 $x_i \in \mathcal{X}$。

**聚类目标（K-Means）：**
$$\min_{C_1,...,C_k} \sum_{j=1}^{k} \sum_{x \in C_j} \|x - \mu_j\|^2$$

**密度估计：**
$$\hat{p}(x) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - x_i)$$

### 自监督学习

**对比损失（InfoNCE）：**
$$\mathcal{L} = -\log \frac{\exp(sim(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(sim(z_i, z_k)/\tau)}$$

其中 $z_i, z_j$ 是同一样本增强视图的嵌入。

---

## 4. 算法框架

### 监督学习流程
```
1. 收集标注数据 {(x_i, y_i)}
2. 划分训练/验证/测试集
3. 选择模型族 F
4. 训练：在训练集上最小化损失
5. 验证：调整超参数
6. 测试：评估最终性能
7. 部署模型
```

### 无监督学习流程
```
1. 收集无标注数据 {x_i}
2. 选择方法（聚类、降维等）
3. 拟合模型发现结构
4. 使用内部指标或下游任务评估
5. 解释和使用发现的模式
```

### 自监督学习流程
```
1. 收集无标注数据 {x_i}
2. 定义前置任务（如预测掩码token）
3. 从数据结构生成伪标签
4. 在前置任务上训练编码器
5. 微调或使用特征进行下游任务
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 标签不足时使用监督方法 | 标注数据不足 | 考虑半监督或自监督 |
| 期望无监督达到监督准确率 | 没有真值指导学习 | 设定现实期望；用于探索 |
| 聚类时忽略领域知识 | 将其视为纯算法问题 | 在设计中融入领域专业知识 |
| 过度依赖自监督特征 | 特征可能不能完美迁移 | 在目标任务上微调 |
| 错误的范式选择 | 没有分析问题需求 | 匹配范式与数据可用性和目标 |

---

## 6. 迷你示例

### Python示例：比较范式

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

# 监督：使用标签学习分类器
clf = LogisticRegression()
clf.fit(X, y_true)
print(f"监督学习准确率: {clf.score(X, y_true):.3f}")

# 无监督：不使用标签进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)
# 注意：聚类标签可能与原始标签不匹配
print(f"K-Means找到 {len(set(y_pred))} 个簇")

# 降维（无监督）
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"解释方差: {sum(pca.explained_variance_ratio_):.3f}")
```

**输出：**
```
监督学习准确率: 0.993
K-Means找到 3 个簇
解释方差: 0.997
```

---

## 7. 测验

<details>
<summary><strong>Q1: 监督学习和无监督学习的区别是什么？</strong></summary>

监督学习使用标注数据（输入-输出对）来学习映射，而无监督学习处理无标注数据，在没有明确目标的情况下发现模式或结构。
</details>

<details>
<summary><strong>Q2: 给出三个自监督学习任务的例子。</strong></summary>

1. **掩码语言建模**（BERT）：预测文本中的掩码token
2. **对比学习**（SimCLR）：通过区分增强视图学习表示
3. **下一句预测**：预测两个句子是否连续
4. **图像旋转预测**：预测应用于图像的旋转角度
5. **掩码自编码**（MAE）：重建被掩码的图像块
</details>

<details>
<summary><strong>Q3: 什么时候优先选择无监督方法而不是监督方法？</strong></summary>

- 当标注数据不可用或获取成本高时
- 用于探索性数据分析以发现未知模式
- 用于异常检测，其中异常是罕见/未定义的
- 用于数据压缩和降维
- 当你想了解底层数据结构时
</details>

<details>
<summary><strong>Q4: 自监督学习的核心思想是什么？</strong></summary>

自监督学习通过定义前置任务从数据本身创建监督信号。模型通过解决这些任务（如预测掩码部分、区分增强视图）学习有用的表示。这些表示可以在有限标注数据的情况下迁移到下游监督任务。
</details>

<details>
<summary><strong>Q5: 什么是经验风险最小化（ERM）？</strong></summary>

ERM是选择在训练数据上最小化平均损失的假设的原则：

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(x_i), y_i)$$

它使用训练样本的经验分布来近似真实风险（数据分布上的期望损失）。
</details>

<details>
<summary><strong>Q6: 为什么无监督聚类可能达不到监督分类的准确率？</strong></summary>

无监督聚类无法获取真值标签，因此：
- 簇边界可能与真实类别边界不一致
- 算法优化几何/统计一致性，而不是分类准确率
- 簇的数量可能与真实类别数量不匹配
- 簇分配是任意的（置换不变）
</details>

---

## 8. 参考文献

1. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. Chen, T., et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML.
4. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
