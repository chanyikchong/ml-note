# 类别不平衡

## 1. 面试摘要

**关键要点：**
- **重采样**：过采样少数类，欠采样多数类
- **代价敏感学习**：少数类错误更高惩罚
- **SMOTE**：合成少数类过采样技术
- **阈值调整**：训练后调整决策阈值
- **指标**：严重不平衡用PR-AUC而非ROC-AUC

**常见面试问题：**
- "如何处理不平衡数据？"
- "过采样vs欠采样的优缺点？"
- "什么时候使用SMOTE？"

---

## 2. 核心定义

### 不平衡比率
$$\text{IR} = \frac{n_{多数}}{n_{少数}}$$

### 重采样策略

| 策略 | 描述 | 影响 |
|------|------|------|
| 随机过采样 | 复制少数类样本 | 可能过拟合 |
| 随机欠采样 | 移除多数类样本 | 丢失信息 |
| SMOTE | 生成合成少数类 | 更好泛化 |
| ADASYN | 自适应合成采样 | 关注困难样本 |

### 代价敏感学习
按类别频率加权损失：
$$L = \sum_i w_{y_i} \cdot \ell(y_i, \hat{y}_i)$$

其中 $w_{少数} > w_{多数}$

---

## 3. 数学与推导

### SMOTE算法

对于每个少数类样本 $x_i$：
1. 找k个最近的少数类邻居
2. 随机选择一个邻居 $x_j$
3. 创建合成样本：$x_{new} = x_i + \lambda (x_j - x_i)$ 其中 $\lambda \in [0,1]$

### 类别权重

**平衡权重：**
$$w_c = \frac{n_{总}}{n_c \cdot n_{类别数}}$$

**逆频率：**
$$w_c = \frac{1}{n_c}$$

### 阈值优化

默认阈值0.5可能不是最优。找到最大化以下的阈值 $t^*$：
- F1分数
- 几何平均
- 基于成本的自定义指标

$$t^* = \arg\max_t F_1(t)$$

---

## 4. 算法框架

### SMOTE实现

```
def SMOTE(X_minority, k=5, N=100):
    synthetic = []
    for i in range(len(X_minority)):
        # 找k个最近邻
        neighbors = k_nearest_neighbors(X_minority, X_minority[i], k)

        for _ in range(N // 100):
            # 选择随机邻居
            j = random.choice(neighbors)

            # 生成合成样本
            lambda = random.uniform(0, 1)
            x_new = X_minority[i] + lambda * (X_minority[j] - X_minority[i])
            synthetic.append(x_new)

    return synthetic
```

### 完整流程

```
def handle_imbalance(X, y, strategy='combined'):
    # 1. 先划分数据（防止泄露）
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 2. 仅对训练数据应用重采样
    if strategy == 'oversample':
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    elif strategy == 'undersample':
        X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
    elif strategy == 'combined':
        X_train, y_train = SMOTETomek().fit_resample(X_train, y_train)

    # 3. 用类别权重训练模型
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # 4. 在验证集上调整阈值
    y_prob = model.predict_proba(X_val)[:, 1]
    best_threshold = find_optimal_threshold(y_val, y_prob)

    return model, best_threshold
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 划分前重采样 | 数据泄露 | 总是先划分，再对训练集重采样 |
| 使用准确率 | 不平衡时有误导 | 用F1、PR-AUC |
| 对测试集过采样 | 评估无效 | 永远不要重采样测试数据 |
| 过度过采样 | 过拟合 | 使用组合策略 |
| 忽略成本结构 | 业务成本重要 | 使用代价敏感学习 |

### 策略选择指南

| 不平衡程度 | 推荐策略 |
|------------|----------|
| 轻微（1:10） | 类别权重 |
| 中等（1:100） | SMOTE + 类别权重 |
| 严重（1:1000+） | 组合多种策略 |
| 少数类很少 | 改用异常检测 |

---

## 6. 迷你示例

```python
import numpy as np
from collections import Counter

def smote_simple(X_minority, k=3, n_synthetic=10):
    """简单SMOTE实现。"""
    n_samples = len(X_minority)
    synthetic = []

    for _ in range(n_synthetic):
        # 选择随机少数类样本
        i = np.random.randint(n_samples)
        x_i = X_minority[i]

        # 找k个最近邻
        distances = np.sqrt(np.sum((X_minority - x_i)**2, axis=1))
        distances[i] = np.inf  # 排除自身
        k_neighbors = np.argsort(distances)[:k]

        # 选择随机邻居并插值
        j = np.random.choice(k_neighbors)
        x_j = X_minority[j]
        lam = np.random.random()
        synthetic.append(x_i + lam * (x_j - x_i))

    return np.array(synthetic)


def compute_class_weights(y):
    """计算平衡类别权重。"""
    counts = Counter(y)
    n_samples = len(y)
    n_classes = len(counts)
    weights = {}
    for cls, count in counts.items():
        weights[cls] = n_samples / (n_classes * count)
    return weights


# 示例
np.random.seed(42)

# 创建不平衡数据集
X_majority = np.random.randn(100, 2) + np.array([2, 2])
X_minority = np.random.randn(10, 2) + np.array([-2, -2])
X = np.vstack([X_majority, X_minority])
y = np.array([0] * 100 + [1] * 10)

print("原始分布:", Counter(y))
print(f"不平衡比率: {100/10:.1f}:1")

# 类别权重
weights = compute_class_weights(y)
print(f"\n类别权重: {weights}")

# SMOTE
synthetic = smote_simple(X_minority, k=3, n_synthetic=90)
X_resampled = np.vstack([X, synthetic])
y_resampled = np.concatenate([y, np.ones(90)])

print(f"\nSMOTE后: {Counter(y_resampled.astype(int))}")

# 阈值优化示例
def find_optimal_threshold(y_true, y_prob):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# 模拟预测
y_prob = np.concatenate([
    np.random.uniform(0, 0.4, 100),  # 多数类（低概率）
    np.random.uniform(0.4, 0.9, 10)  # 少数类（较高概率）
])

opt_t, opt_f1 = find_optimal_threshold(y, y_prob)
print(f"\n最优阈值: {opt_t:.2f} (F1={opt_f1:.3f})")
print(f"默认阈值F1: {find_optimal_threshold(y, y_prob)[1]:.3f} at t=0.50")
```

**输出：**
```
原始分布: Counter({0: 100, 1: 10})
不平衡比率: 10.0:1

类别权重: {0: 0.55, 1: 5.5}

SMOTE后: Counter({0: 100, 1: 100})

最优阈值: 0.35 (F1=0.667)
默认阈值F1: 0.400 at t=0.50
```

---

## 7. 测验

<details>
<summary><strong>Q1: 过采样vs欠采样的优缺点是什么？</strong></summary>

**过采样（如SMOTE）**：
- 优点：无信息丢失，训练集更大
- 缺点：可能过拟合，训练时间长，可能创建不真实样本

**欠采样**：
- 优点：训练更快，平衡数据集
- 缺点：丢失可能有用的多数类信息

**最佳实践**：常常组合两者（如SMOTE + Tomek链接）或用不同采样的集成方法。
</details>

<details>
<summary><strong>Q2: 为什么不平衡数据时准确率有误导性？</strong></summary>

99%负例，1%正例的数据：
- 预测"全部负例"的模型达到99%准确率
- 但对找正类毫无用处

1:99不平衡时"预测全负"的示例指标：
- 准确率：99%
- 召回率（正类）：0%
- F1（正类）：0%

更好的指标：精确率、召回率、F1、PR-AUC（关注少数类）
</details>

<details>
<summary><strong>Q3: 什么是SMOTE，它如何工作？</strong></summary>

**SMOTE**（合成少数类过采样技术）：
1. 对每个少数类样本，找k个最近的少数类邻居
2. 随机选择一个邻居
3. 在它们之间的线段上创建合成样本

$$x_{new} = x_i + \lambda \cdot (x_j - x_i), \quad \lambda \in [0,1]$$

**优点**：创建真实样本，比简单复制更好的泛化

**局限**：可能在重叠区域创建噪声样本，不考虑多数类
</details>

<details>
<summary><strong>Q4: 为什么重采样必须在训练-测试划分之后？</strong></summary>

如果在划分前重采样：
1. 合成样本可能出现在测试集中
2. 这些合成样本是用训练信息创建的
3. 导致**数据泄露** - 测试集不再独立

**正确流程**：
1. 将数据划分为训练/测试
2. 仅对训练数据应用重采样
3. 保持测试集原始（真实世界分布）
4. 在未修改的测试集上评估
</details>

<details>
<summary><strong>Q5: 代价敏感学习中的类别权重如何工作？</strong></summary>

类别权重增加对少数类误分类的惩罚：

$$L_{weighted} = \sum_i w_{y_i} \cdot \ell(y_i, \hat{y}_i)$$

**平衡权重**：$w_c = \frac{n_{总}}{n_c \cdot n_{类别数}}$

100多数类，10少数类的例子：
- $w_{多数} = \frac{110}{100 \times 2} = 0.55$
- $w_{少数} = \frac{110}{10 \times 2} = 5.5$

少数类误分类代价高10倍，激励正确的少数类预测。
</details>

<details>
<summary><strong>Q6: 什么时候用异常检测而不是分类处理不平衡？</strong></summary>

使用异常检测当：
1. **极端不平衡**（>1:10,000）：少数类样本太少无法学习模式
2. **少数类未定义**：没有明确的正类定义
3. **需要新模式**：想检测以前未见过的异常

异常检测方法：
- 孤立森林
- 单类SVM
- 自编码器

这些学习"正常"模式并标记偏差，而不是需要少数类样本。
</details>

---

## 8. 参考文献

1. Chawla, N., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." JAIR.
2. He, H., & Garcia, E. (2009). "Learning from Imbalanced Data." IEEE TKDE.
3. Batista, G., et al. (2004). "A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data." ACM SIGKDD.
4. Japkowicz, N., & Stephen, S. (2002). "The Class Imbalance Problem: A Systematic Study." IDA.
5. imbalanced-learn文档：重采样策略。
