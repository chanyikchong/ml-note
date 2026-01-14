# 数据划分与验证

## 1. 面试摘要

**关键要点：**
- **训练集**：用于拟合模型参数
- **验证集**：用于调整超参数和模型选择
- **测试集**：仅用于最终无偏评估一次
- **数据泄露**：测试/验证信息泄露到训练中
- **交叉验证**：K折用于稳健的超参数调整
- 知道何时CV无效（时间序列、分组数据）

**常见面试问题：**
- "为什么需要分开的训练/验证/测试集？"
- "什么是数据泄露？举例说明。"
- "什么时候不应该使用k折交叉验证？"

---

## 2. 核心定义

### 训练/验证/测试划分
- **训练集（~60-80%）**：用于学习模型参数的数据
- **验证集（~10-20%）**：用于超参数调整、早停、模型选择的数据
- **测试集（~10-20%）**：用于最终评估的保留数据；训练期间从不使用

### 数据泄露
训练数据集之外的信息提供了意外的预测信号。

**类型：**
1. **目标泄露**：特征包含关于目标的信息
2. **训练-测试泄露**：测试信息泄露到训练中
3. **时间泄露**：使用未来数据预测过去

### 交叉验证
通过将数据划分为多个训练/验证分割来估计模型性能的技术。

**K折CV：**
- 将数据分成K个相等部分（折）
- 在K-1折上训练，在剩余折上验证
- 重复K次，平均结果

---

## 3. 数学与推导

### 泛化误差分解

真实泛化误差无法直接计算。我们估计它：

$$\text{测试误差} \approx \mathbb{E}[\mathcal{L}(f(x), y)]$$

**偏差-方差分解**（对于平方损失）：
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

### 交叉验证估计器

K折CV泛化误差估计：
$$\hat{R}_{CV} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{|D_k|} \sum_{(x,y) \in D_k} \mathcal{L}(f^{(-k)}(x), y)$$

其中 $f^{(-k)}$ 在除折 $k$ 以外的所有数据上训练。

### CV估计的标准误差

$$SE = \sqrt{\frac{1}{K(K-1)} \sum_{k=1}^{K} (R_k - \hat{R}_{CV})^2}$$

**一个标准误差规则**：选择在最佳模型一个SE范围内最简单的模型。

---

## 4. 算法框架

### 标准训练/验证/测试划分
```
1. 打乱数据（如果i.i.d.假设成立）
2. 划分：60%训练，20%验证，20%测试
3. 在训练集上训练模型
4. 使用验证集调整超参数
5. 基于验证性能选择最佳模型
6. 在测试集上评估一次
7. 报告测试性能作为最终结果
```

### K折交叉验证
```
1. 随机打乱数据
2. 分成K个相等的折
3. 对于k = 1到K：
   a. 使用折k作为验证集
   b. 使用剩余K-1折作为训练集
   c. 训练模型并记录验证分数
4. 平均K个验证分数
5. （可选）用最佳超参数在所有数据上重新训练
```

### 分层K折（用于分类）
```
1. 确保每个折保持类别比例
2. 对于每个类别：
   a. 按比例在各折中分配样本
3. 继续标准K折程序
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 使用测试集调参 | 急于提高分数 | 严格纪律；最终评估前不碰测试集 |
| 划分前特征缩放 | 在所有数据上拟合缩放器 | 只在训练数据上拟合缩放器 |
| 泄露未来信息 | 不尊重时间顺序 | 对时间序列使用时间划分 |
| 对分组数据随机划分 | 忽略组结构 | 使用GroupKFold或组感知划分 |
| CV折数太少 | 计算限制 | 至少使用K=5；K=10常见 |
| 对验证集过拟合 | 过度超参数搜索 | 使用嵌套CV；限制搜索迭代 |

### 数据泄露示例

**示例1：目标泄露**
```python
# 错误：Hospital_discharge_date泄露患者结果
features = ['age', 'admission_date', 'discharge_date']  # 出院意味着存活
```

**示例2：预处理泄露**
```python
# 错误：在完整数据集上缩放
scaler.fit(X)  # 包含测试数据！
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 正确：只在训练数据上缩放
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 6. 迷你示例

### Python示例：正确的验证设置

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 正确的训练/测试划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 创建管道（缩放在CV内部进行）
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# 在训练数据上进行5折交叉验证
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV准确率: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# 在测试集上最终评估
pipeline.fit(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"测试准确率: {test_score:.3f}")
```

**输出：**
```
CV准确率: 0.856 (+/- 0.038)
测试准确率: 0.870
```

### 何时不使用标准K折

```python
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

# 时间序列：使用TimeSeriesSplit
ts_cv = TimeSeriesSplit(n_splits=5)
# 确保只在过去的数据上训练

# 分组数据：使用GroupKFold
groups = [0, 0, 1, 1, 2, 2, 3, 3]  # 患者ID
group_cv = GroupKFold(n_splits=4)
# 确保同一患者不同时在训练和验证集中
```

---

## 7. 测验

<details>
<summary><strong>Q1: 为什么不能使用测试集进行超参数调整？</strong></summary>

使用测试集调参会导致模型选择过程对测试数据过拟合。测试集旨在提供无偏的泛化性能估计。如果我们在上面调参，实际上是在该数据上"训练"我们的模型选择，使最终测试分数成为乐观估计。
</details>

<details>
<summary><strong>Q2: 什么是数据泄露？举两个例子。</strong></summary>

数据泄露发生在训练集之外的信息影响模型训练时，导致过于乐观的性能估计。

**例子：**
1. **预处理泄露**：在划分前在整个数据集上拟合缩放器或编码器
2. **目标泄露**：包含目标信息的特征（如预测患者结果时的"treatment_successful"）
3. **时间泄露**：使用未来数据预测过去事件
</details>

<details>
<summary><strong>Q3: 什么时候k折交叉验证无效？</strong></summary>

当i.i.d.假设被违反时，K折CV无效：
1. **时间序列数据**：未来数据会泄露到训练中；改用TimeSeriesSplit
2. **分组/聚类数据**：同组观测值同时在训练/验证集中；使用GroupKFold
3. **空间数据**：附近的点可能相关；使用空间分块
4. **样本不独立时**：任何依赖结构都需要特殊处理
</details>

<details>
<summary><strong>Q4: 什么是一个标准误差规则？</strong></summary>

一个标准误差规则建议选择性能在最佳模型一个标准误差范围内最简单的模型。这促进了简约性，有助于避免对验证集过拟合，同时接受估计性能的小幅下降以换取更简单的模型。
</details>

<details>
<summary><strong>Q5: 在交叉验证中应该如何处理预处理？</strong></summary>

预处理（缩放、编码、填充）应该只在训练折上拟合，然后应用于验证折。这通过以下方式实现：
1. 使用sklearn `Pipeline`封装预处理和模型
2. 在每个CV折内拟合预处理器
3. 在划分前从不在完整数据集上拟合

这防止了验证数据信息泄露到训练中。
</details>

<details>
<summary><strong>Q6: 什么是嵌套交叉验证？什么时候需要它？</strong></summary>

嵌套CV使用外循环进行性能估计，内循环进行超参数调整。外循环提供无偏的泛化估计，而内循环调整超参数。

需要它的情况：
- 在超参数调整的情况下报告最终性能估计
- 想避免使用相同数据进行调整和评估带来的乐观偏差
- 数据集小，单独的测试集会浪费数据
</details>

---

## 8. 参考文献

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Kohavi, R. (1995). "A Study of Cross-Validation and Bootstrap for Accuracy Estimation." IJCAI.
3. Kaufman, S., et al. (2012). "Leakage in Data Mining: Formulation, Detection, and Avoidance." TKDD.
4. Varma, S., & Simon, R. (2006). "Bias in Error Estimation When Using Cross-Validation for Model Selection." BMC Bioinformatics.
5. Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." JMLR.
