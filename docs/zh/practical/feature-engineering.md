# 特征工程

## 1. 面试摘要

**关键要点：**
- **数值型**：缩放、归一化、对数变换、分箱
- **类别型**：独热编码、目标编码、嵌入
- **缺失值**：填充策略很重要
- **特征选择**：过滤、包装、嵌入方法
- **特征重要性**：降维、提高可解释性

**常见面试问题：**
- "什么时候用标准化vs归一化？"
- "如何处理高基数类别变量？"
- "特征选择有哪些方法？"

---

## 2. 核心定义

### 数值变换

| 方法 | 公式 | 适用场景 |
|------|------|----------|
| 标准化 | $(x - \mu) / \sigma$ | 大多数算法，假设高斯 |
| Min-Max | $(x - min) / (max - min)$ | 有界[0,1]，神经网络 |
| 对数变换 | $\log(x + 1)$ | 右偏分布 |
| 幂变换 | $x^\lambda$ 或 Box-Cox | 使数据更高斯 |

### 类别编码

| 方法 | 描述 | 基数 |
|------|------|------|
| 独热 | 每类别一个二进制列 | 低（<20） |
| 标签编码 | 每类别一个整数 | 有序数据 |
| 目标编码 | 每类别的目标均值 | 高基数 |
| 嵌入 | 学习的密集向量 | 非常高，深度学习 |

### 缺失值策略

| 策略 | 何时使用 |
|------|----------|
| 删除行 | 少量缺失，随机 |
| 均值/中位数 | 数值型，少量缺失 |
| 众数 | 类别型 |
| 基于模型（KNN，迭代） | 复杂模式 |
| 指示变量 | 缺失有信息量 |

---

## 3. 数学与推导

### 带平滑的目标编码

防止稀有类别过拟合：

$$\text{encoded}_c = \lambda(c) \cdot \bar{y}_c + (1 - \lambda(c)) \cdot \bar{y}_{global}$$

其中：

$$\lambda(c) = \frac{n_c}{n_c + m}$$

$m$ 是平滑参数。样本越多→信任类别均值；越少→信任全局均值。

### 方差膨胀因子（VIF）

检测多重共线性：

$$VIF_j = \frac{1}{1 - R_j^2}$$

$R_j^2$ 是特征$j$对所有其他特征回归的R²。
- VIF > 5-10 表示高多重共线性

### 特征选择的信息增益

$$IG(Y, X) = H(Y) - H(Y|X)$$

更高IG = 更有预测性的特征。

---

## 4. 算法框架

### 特征工程流程

```
def feature_pipeline(df, target, num_cols, cat_cols):
    # 处理缺失值
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # 数值变换
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 类别编码
    for col in cat_cols:
        if df[col].nunique() < 10:
            # 低基数用独热
            df = pd.get_dummies(df, columns=[col])
        else:
            # 高基数用目标编码
            means = df.groupby(col)[target].mean()
            df[col] = df[col].map(means)

    return df
```

### 特征选择方法

```
# 过滤方法：相关性
def correlation_filter(X, y, threshold=0.1):
    correlations = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
    return [i for i, c in enumerate(correlations) if c > threshold]

# 包装方法：递归特征消除
def rfe(model, X, y, n_features):
    while X.shape[1] > n_features:
        model.fit(X, y)
        importances = model.feature_importances_
        worst = np.argmin(importances)
        X = np.delete(X, worst, axis=1)
    return X

# 嵌入方法：L1正则化
def lasso_selection(X, y, alpha=0.01):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return np.where(model.coef_ != 0)[0]
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 编码导致数据泄露 | 在所有数据上拟合编码器 | 仅在训练集拟合 |
| 目标泄露 | 特征中有未来信息 | 仔细特征审计 |
| 高基数独热 | 数千列 | 用目标/嵌入编码 |
| 用训练统计缩放测试 | 不同分布 | 保存训练的scaler |
| 忽略特征交互 | 线性模型遗漏 | 创建显式交互 |

### 预处理顺序

```
1. 训练-测试划分（防止泄露）
2. 处理缺失值（在训练集上拟合填充器）
3. 编码类别（在训练集上拟合编码器）
4. 缩放数值（在训练集上拟合缩放器）
5. 特征选择（仅用训练集）
6. 用拟合的转换器变换测试集
```

---

## 6. 迷你示例

```python
import numpy as np

def standardize(X, fit=True, mean=None, std=None):
    if fit:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std, mean, std

def target_encode(X_cat, y, smoothing=10):
    global_mean = np.mean(y)
    encoded = np.zeros(len(X_cat))
    for cat in np.unique(X_cat):
        mask = X_cat == cat
        n = np.sum(mask)
        cat_mean = np.mean(y[mask])
        lambda_c = n / (n + smoothing)
        encoded[mask] = lambda_c * cat_mean + (1 - lambda_c) * global_mean
    return encoded

def correlation_feature_selection(X, y, k=5):
    correlations = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
    top_k = np.argsort(correlations)[-k:]
    return top_k, correlations[top_k]

# 示例
np.random.seed(42)

# 创建数据集
n_samples = 100
X_num = np.random.randn(n_samples, 3)  # 3个数值特征
X_cat = np.random.choice(['A', 'B', 'C'], n_samples)  # 1个类别
y = X_num[:, 0] + (X_cat == 'A').astype(float) * 2 + np.random.randn(n_samples) * 0.5

# 划分
train_idx = np.arange(80)
test_idx = np.arange(80, 100)

# 标准化数值
X_train_scaled, mean, std = standardize(X_num[train_idx], fit=True)
X_test_scaled, _, _ = standardize(X_num[test_idx], fit=False, mean=mean, std=std)

print("标准化：")
print(f"  训练均值: {np.mean(X_train_scaled, axis=0)}")
print(f"  训练标准差: {np.std(X_train_scaled, axis=0)}")

# 目标编码类别
X_cat_encoded = target_encode(X_cat[train_idx], y[train_idx])
print(f"\n目标编码（类别A均值: {np.mean(y[train_idx][X_cat[train_idx]=='A']):.2f}）")
print(f"  编码值: A={X_cat_encoded[X_cat[train_idx]=='A'][0]:.2f}, B={X_cat_encoded[X_cat[train_idx]=='B'][0]:.2f}")

# 特征选择
all_features = np.column_stack([X_train_scaled, X_cat_encoded])
top_features, correlations = correlation_feature_selection(all_features, y[train_idx], k=2)
print(f"\n按相关性排名前2特征: {top_features}")
print(f"  相关性: {correlations}")
```

---

## 7. 测验

<details>
<summary><strong>Q1: 什么时候用标准化vs min-max归一化？</strong></summary>

**标准化**（z分数）：
- 算法假设高斯分布时
- 对幅度敏感的算法（SVM、逻辑回归）
- 有异常值时（受影响较小）

**Min-Max归一化**：
- 需要有界范围[0, 1]时
- 神经网络（激活函数期望特定范围）
- 分布非高斯时
- 无异常值或异常值有意义时
</details>

<details>
<summary><strong>Q2: 如何处理高基数类别特征？</strong></summary>

选项：
1. **目标编码**：用目标均值替换类别（用平滑）
2. **频率编码**：用计数/频率替换
3. **嵌入**：学习密集向量表示（深度学习）
4. **哈希**：哈希到固定数量的桶
5. **分组**：将稀有类别合并为"其他"

避免独热编码（创建太多稀疏特征）。
</details>

<details>
<summary><strong>Q3: 特征选择有哪三类方法？</strong></summary>

1. **过滤方法**：独立于模型对特征评分
   - 相关性、互信息、卡方
   - 快但忽略特征交互

2. **包装方法**：用模型性能作为标准
   - 前向选择、后向消除、RFE
   - 更好但计算昂贵

3. **嵌入方法**：特征选择内置于训练
   - L1正则化（Lasso）、树重要性
   - 质量和效率的平衡
</details>

<details>
<summary><strong>Q4: 如何防止特征工程中的数据泄露？</strong></summary>

**规则**：
1. 任何变换前先划分数据
2. 仅在训练数据上拟合转换器
3. 用拟合的参数变换测试数据
4. 测试时不要用目标信息编码
5. 注意时间序列中的时间泄露

**示例流程**：
```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 用训练的统计量
```
</details>

<details>
<summary><strong>Q5: 什么是目标编码，如何防止过拟合？</strong></summary>

**目标编码**：用该类别的目标均值替换类别。

**过拟合问题**：稀有类别的均值不可靠。

**解决方案**：
1. **平滑**：将类别均值与全局均值混合
   $$encoded = \frac{n \cdot cat\_mean + m \cdot global\_mean}{n + m}$$

2. **交叉验证编码**：用其他折编码每个折

3. **添加噪声**：通过添加小噪声正则化
</details>

<details>
<summary><strong>Q6: 如何处理缺失值？</strong></summary>

**策略**：
1. **删除**：如果少量缺失，随机模式
2. **统计量填充**：均值（数值型），众数（类别型）
3. **基于模型**：KNN填充器，迭代填充器
4. **创建指示器**：二进制"is_missing"特征（如果有信息量）
5. **领域特定**：用知识（如，"无购买"用0）

**关键**：总是先在训练数据上填充，然后对测试应用相同值。
</details>

---

## 8. 参考文献

1. Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*. CRC Press.
2. Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly.
3. Guyon, I., & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." JMLR.
4. scikit-learn文档：数据预处理。
5. Micci-Barreca, D. (2001). "A Preprocessing Scheme for High-Cardinality Categorical Attributes." ACM SIGKDD.
