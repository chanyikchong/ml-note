# 数据中心机器学习问题

## 1. 面试摘要

**关键要点：**
- **标签噪声**：错误标签的数据降低模型性能
- **数据集偏移**：训练和测试分布不同
- **异常值**：可能是或可能不是错误的极端值
- **数据质量 > 模型复杂度**：更好的数据往往胜过更好的模型

**常见面试问题：**
- "如何处理数据集中的噪声标签？"
- "什么是数据集偏移，如何检测？"
- "如何决定是删除还是保留异常值？"

---

## 2. 核心定义

### 数据集偏移类型

| 类型 | 描述 | 示例 |
|------|------|------|
| 协变量偏移 | $P(X)$变化，$P(Y|X)$相同 | 不同人群 |
| 标签偏移 | $P(Y)$变化，$P(X|Y)$相同 | 类别比例变化 |
| 概念漂移 | $P(Y|X)$变化 | 欺诈模式演变 |
| 领域偏移 | $P(X)$和$P(Y|X)$都变化 | 新数据源 |

### 标签噪声类型

| 类型 | 描述 | 影响 |
|------|------|------|
| 随机噪声 | 随机错误标签 | 增加方差 |
| 系统噪声 | 一致性错误（标注者偏差） | 有偏模型 |
| 类别依赖 | 某些类别更有噪声 | 不公平预测 |

### 异常值类型

| 类型 | 描述 | 处理 |
|------|------|------|
| 错误异常值 | 数据输入错误 | 删除或更正 |
| 自然异常值 | 真实极端值 | 保留（通常） |
| 有影响力的异常值 | 高杠杆点 | 调查 |

---

## 3. 数学与推导

### 协变量偏移校正

如果$P_{train}(X) \neq P_{test}(X)$但$P(Y|X)$相同：

**重要性加权：**

$$w(x) = \frac{P_{test}(x)}{P_{train}(x)}$$

加权损失：

$$L_{corrected} = \sum_i w(x_i) \cdot \ell(y_i, \hat{y}_i)$$

### 标签噪声模型

噪声率$\eta$（标签翻转概率）：

$$P(\tilde{y}|x) = (1-\eta) P(y|x) + \eta P(y_{wrong}|x)$$

**前向校正：**

$$P(y|x) = \frac{P(\tilde{y}|x) - \eta P(y_{wrong}|x)}{1 - \eta}$$

### 异常值检测（Z分数）

$$z_i = \frac{x_i - \mu}{\sigma}$$

如果$|z_i| > 3$则标记为异常值（假设高斯分布）。

### IQR方法

$$\text{异常值如果 } x < Q_1 - 1.5 \cdot IQR \text{ 或 } x > Q_3 + 1.5 \cdot IQR$$

其中$IQR = Q_3 - Q_1$。

---

## 4. 算法框架

### 置信学习（标签噪声检测）

```
def confident_learning(X, y, model):
    # 步骤1：获取预测概率
    probs = cross_val_predict(model, X, y, method='predict_proba')

    # 步骤2：估计每类的阈值
    thresholds = []
    for c in classes:
        # 标签为c的样本的平均概率
        thresholds.append(np.mean(probs[y == c, c]))

    # 步骤3：创建置信联合矩阵
    C = np.zeros((n_classes, n_classes))
    for i, (yi, pi) in enumerate(zip(y, probs)):
        predicted = np.argmax(pi > thresholds)
        C[yi, predicted] += 1

    # 步骤4：找到标签错误
    errors = []
    for i, (yi, pi) in enumerate(zip(y, probs)):
        if pi[yi] < thresholds[yi] and max(pi) > thresholds[np.argmax(pi)]:
            errors.append(i)

    return errors
```

### 数据集偏移检测

```
def detect_covariate_shift(X_train, X_test):
    # 训练分类器区分训练vs测试
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    X = np.vstack([X_train, X_test])

    model = LogisticRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    # AUC接近0.5 = 无偏移；接近1.0 = 显著偏移
    return np.mean(scores)
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 删除所有异常值 | 有些是有效的 | 先用领域知识 |
| 忽略标签噪声 | 假设标签正确 | 使用置信学习 |
| 在偏移数据上训练 | 分布不匹配 | 检查偏移，重新加权 |
| 过拟合噪声 | 模型记忆错误 | 正则化，噪声鲁棒损失 |
| 单个标注者 | 无质量检查 | 多个标注者，一致性分数 |

### 数据质量检查清单

```
1. 检查重复项
2. 验证数据类型和范围
3. 检查缺失值模式
4. 分析类别平衡
5. 检查标签一致性
6. 检查训练/测试分布相似性
7. 识别潜在泄露特征
8. 验证时间顺序（如适用）
```

---

## 6. 迷你示例

```python
import numpy as np

def detect_outliers_iqr(data):
    """使用IQR方法检测异常值。"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = (data < lower) | (data > upper)
    return outliers, lower, upper


def detect_outliers_zscore(data, threshold=3):
    """使用z分数检测异常值。"""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold


def estimate_label_noise(y_true, y_noisy):
    """从已知标签估计噪声率。"""
    disagreement = np.sum(y_true != y_noisy)
    noise_rate = disagreement / len(y_true)
    return noise_rate


def add_label_noise(y, noise_rate=0.1):
    """添加随机标签噪声。"""
    n_flip = int(len(y) * noise_rate)
    flip_idx = np.random.choice(len(y), n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]  # 二元翻转
    return y_noisy, flip_idx


def detect_shift_simple(X_train, X_test):
    """使用特征统计的简单协变量偏移检测。"""
    train_stats = {'mean': np.mean(X_train, axis=0), 'std': np.std(X_train, axis=0)}
    test_stats = {'mean': np.mean(X_test, axis=0), 'std': np.std(X_test, axis=0)}

    # 比较分布
    mean_diff = np.abs(train_stats['mean'] - test_stats['mean'])
    std_ratio = test_stats['std'] / (train_stats['std'] + 1e-8)

    return mean_diff, std_ratio


# 示例
np.random.seed(42)

# 异常值检测
print("=== 异常值检测 ===")
data = np.concatenate([np.random.randn(100), [10, -8, 15]])  # 添加异常值
outliers_iqr, lower, upper = detect_outliers_iqr(data)
outliers_zscore = detect_outliers_zscore(data)

print(f"数据形状: {data.shape}")
print(f"IQR边界: [{lower:.2f}, {upper:.2f}]")
print(f"IQR异常值: {np.sum(outliers_iqr)} 个样本")
print(f"Z分数异常值: {np.sum(outliers_zscore)} 个样本")

# 标签噪声
print("\n=== 标签噪声 ===")
y_true = np.random.randint(0, 2, 100)
y_noisy, flipped = add_label_noise(y_true, noise_rate=0.15)
estimated_noise = estimate_label_noise(y_true, y_noisy)
print(f"真实噪声率: 0.15")
print(f"估计噪声率: {estimated_noise:.2f}")
print(f"翻转索引（前5个）: {flipped[:5]}")

# 数据集偏移
print("\n=== 数据集偏移检测 ===")
X_train = np.random.randn(100, 3)
X_test_no_shift = np.random.randn(50, 3)  # 相同分布
X_test_shift = np.random.randn(50, 3) + np.array([2, 0, -1])  # 偏移

mean_diff_no, std_ratio_no = detect_shift_simple(X_train, X_test_no_shift)
mean_diff_yes, std_ratio_yes = detect_shift_simple(X_train, X_test_shift)

print("无偏移 - 均值差异:", mean_diff_no.round(2))
print("有偏移 - 均值差异:", mean_diff_yes.round(2))
```

**输出：**
```
=== 异常值检测 ===
数据形状: (103,)
IQR边界: [-2.25, 2.31]
IQR异常值: 3 个样本
Z分数异常值: 3 个样本

=== 标签噪声 ===
真实噪声率: 0.15
估计噪声率: 0.15
翻转索引（前5个）: [51 92 14 71 60]

=== 数据集偏移检测 ===
无偏移 - 均值差异: [0.12 0.08 0.15]
有偏移 - 均值差异: [2.05 0.11 1.12]
```

---

## 7. 测验

<details>
<summary><strong>Q1: 协变量偏移和概念漂移有什么区别？</strong></summary>

**协变量偏移**：
- 输入分布$P(X)$变化
- 关系$P(Y|X)$保持不变
- 示例：在年轻用户上训练，在老年用户上测试

**概念漂移**：
- 关系$P(Y|X)$随时间变化
- 相同输入可能映射到不同输出
- 示例：欺诈模式演变

**关键见解**：协变量偏移可以通过重要性加权校正；概念漂移需要模型更新。
</details>

<details>
<summary><strong>Q2: 如何处理训练数据中的噪声标签？</strong></summary>

**策略**：

1. **噪声鲁棒损失**：使用MAE而不是交叉熵（更不敏感）

2. **置信学习**：检测并删除可能错误标签的样本

3. **协同教学**：训练两个网络，每个在"干净"样本上教另一个

4. **标签平滑**：软标签减少对错误标签的过度自信

5. **多个标注者**：使用多数投票或建模标注者可靠性

**关键**：不要假设标签100%正确；建立噪声容忍度。
</details>

<details>
<summary><strong>Q3: 什么时候应该删除异常值vs保留它们？</strong></summary>

**删除当**：
- 明显的数据输入错误（如年龄=999）
- 不可能的值（负价格）
- 测量设备故障

**保留当**：
- 自然极端值（富裕客户）
- 模型的重要边缘情况
- 异常值是你试图检测的东西（欺诈）

**调查当**：
- 高杠杆点显著影响模型
- 异常值模式暗示系统性问题

**最佳实践**：记录决策；考虑稳健方法（中位数、分位数回归）。
</details>

<details>
<summary><strong>Q4: 如何检测训练和测试之间的数据集偏移？</strong></summary>

**方法**：

1. **分类器双样本检验**：训练模型区分训练vs测试
   - AUC ~ 0.5：无偏移
   - AUC ~ 1.0：显著偏移

2. **特征分布比较**：
   - 比较均值、方差、分位数
   - 每个特征的KS检验、卡方检验

3. **模型预测分布**：
   - 比较训练和测试上的$P(\hat{y})$
   - 如果无偏移应该相似

4. **基于时间的分析**：随时间绘制指标以检测渐进漂移

**行动**：如果检测到偏移，考虑重要性加权或收集新训练数据。
</details>

<details>
<summary><strong>Q5: 什么是协变量偏移的重要性加权？</strong></summary>

当$P_{train}(X) \neq P_{test}(X)$时，加权训练样本：

$$w(x) = \frac{P_{test}(x)}{P_{train}(x)}$$

**估计方法**：
1. 训练分类器区分训练/测试
2. 使用密度比估计（KLIEP、uLSIF）
3. 倾向得分

**加权训练**：

$$\min_\theta \sum_i w(x_i) \cdot L(y_i, f_\theta(x_i))$$

**警告**：当$P_{train}(x)$非常小时不稳定（高方差权重）。
</details>

<details>
<summary><strong>Q6: 标签噪声如何影响不同模型？</strong></summary>

**更鲁棒**（对随机噪声）：
- 基于树的模型（决策边界来自数据）
- K-NN（局部多数投票有帮助）
- 正则化模型（防止记忆）

**不太鲁棒**：
- 神经网络（可以记忆噪声）
- 高容量模型
- 惩罚自信错误预测的损失函数

**缓解**：
- 使用噪声鲁棒损失（MAE、截断损失）
- 早停（防止记忆）
- 置信学习清洗数据
- 集成方法
</details>

---

## 8. 参考文献

1. Northcutt, C., et al. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels." JAIR.
2. Sugiyama, M., & Kawanabe, M. (2012). *Machine Learning in Non-Stationary Environments*. MIT Press.
3. Quinonero-Candela, J., et al. (2009). *Dataset Shift in Machine Learning*. MIT Press.
4. Frenay, B., & Verleysen, M. (2014). "Classification in the Presence of Label Noise." IEEE TNNLS.
5. Nettleton, D., et al. (2010). "A Study of the Effect of Different Types of Noise on the Precision of Supervised Learning Techniques." AIR.
