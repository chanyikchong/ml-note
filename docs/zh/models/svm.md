# 支持向量机

## 1. 面试摘要

**关键要点：**
- **最大间隔分类器**：找到间隔最大的超平面
- **支持向量**：离决策边界最近的点
- **软间隔（C参数）**：在间隔和误分类之间权衡
- **核技巧**：在高维空间计算点积而不显式映射
- **常见核**：线性、多项式、RBF（高斯）

**常见面试问题：**
- "SVM的直觉是什么？"
- "解释核技巧"
- "支持向量代表什么？"

---

## 2. 核心定义

### 硬间隔SVM
对于线性可分数据，找到超平面 $w^Tx + b = 0$：
- 正确分类所有点：$y_i(w^Tx_i + b) \geq 1$
- 最大化间隔：$\frac{2}{\|w\|}$

### 软间隔SVM
用松弛变量 $\xi_i$ 允许一些误分类：

$$y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### 支持向量
位于或在间隔边界内的训练点：
- $y_i(w^Tx_i + b) = 1$ 的点（在间隔上）
- 对偶形式中 $0 < \alpha_i < C$ 的点

### 核函数
$K(x, z) = \langle\phi(x), \phi(z)\rangle$ 在特征空间计算内积。

| 核 | 公式 | 用途 |
|----|------|------|
| 线性 | $x^Tz$ | 线性可分 |
| 多项式 | $(x^Tz + c)^d$ | 多项式边界 |
| RBF | $\exp(-\gamma\|x-z\|^2)$ | 复杂边界 |

---

## 3. 数学与推导

### 原始形式（硬间隔）

$$\min_{w,b} \frac{1}{2}\|w\|^2$$

$$\text{s.t. } y_i(w^Tx_i + b) \geq 1, \quad \forall i$$

### 原始形式（软间隔）

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$$

$$\text{s.t. } y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### 对偶形式

使用拉格朗日乘子 $\alpha_i$：

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

$$\text{s.t. } 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

**关键洞察**：只有点积 $x_i^Tx_j$ 出现 → 可以用核！

### 核技巧

用 $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ 替换 $x_i^Tx_j$：

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

**决策函数**：

$$f(x) = \text{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)$$

### RBF核性质

$$K(x, z) = \exp(-\gamma\|x-z\|^2)$$

- 对应无限维特征空间
- $\gamma$ 控制决策边界平滑度
- 高 $\gamma$：复杂边界（过拟合风险）
- 低 $\gamma$：平滑边界（欠拟合风险）

---

## 4. 算法框架

### 训练SVM（SMO算法直觉）

```
序列最小优化：
1. 初始化所有 α = 0
2. 重复直到收敛：
   a. 选择两个违反KKT条件的alpha（α_i, α_j）
   b. 解析优化这两个（2D问题）
   c. 更新b
3. 支持向量：α > 0的点
```

### 预测

```
输入：新点x
输出：类别标签

# 计算决策值
decision = b
对于每个支持向量 (x_i, y_i, α_i)：
    decision += α_i * y_i * K(x_i, x)

返回 sign(decision)
```

### 选择参数

```
核选择：
    从RBF开始（最灵活）
    如果高维/稀疏尝试线性

对于C（正则化）：
    高C：紧密拟合训练数据（过拟合风险）
    低C：更大间隔，允许更多误分类

对于γ（RBF）：
    高γ：每个点只有局部影响
    低γ：点有更广影响

用交叉验证一起调整C和γ
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 不缩放特征 | SVM对尺度敏感 | 总是标准化 |
| 错误核选择 | 默认可能不适合数据 | 尝试多个核 |
| 忽略C参数 | 默认可能过拟合/欠拟合 | 网格搜索C |
| RBF γ太高 | 过拟合，只记住训练 | 交叉验证γ |
| RBF特征太多 | 维度灾难 | 考虑线性核 |

### 何时使用SVM

| 场景 | 建议 |
|------|------|
| 高维，稀疏 | 线性SVM |
| 小中数据集，复杂边界 | RBF SVM |
| 大数据集（>10万样本） | 考虑其他方法（训练慢） |
| 需要概率 | SVM + Platt缩放，或用其他方法 |

---

## 6. 迷你示例

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification, make_circles

# 生成非线性数据
X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

# 划分和缩放
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 线性SVM（在圆形数据上会失败）
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_s, y_train)
print(f"线性SVM准确率: {svm_linear.score(X_test_s, y_test):.3f}")

# RBF SVM（应该有效）
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='auto')
svm_rbf.fit(X_train_s, y_train)
print(f"RBF SVM准确率: {svm_rbf.score(X_test_s, y_test):.3f}")

# 网格搜索最优参数
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train_s, y_train)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳CV分数: {grid.best_score_:.3f}")
print(f"测试准确率: {grid.score(X_test_s, y_test):.3f}")

# 支持向量
print(f"支持向量数: {len(svm_rbf.support_)}")
print(f"每类支持向量: {svm_rbf.n_support_}")
```

**输出：**
```
线性SVM准确率: 0.500
RBF SVM准确率: 0.990
最佳参数: {'C': 10, 'gamma': 1}
最佳CV分数: 0.985
测试准确率: 0.990
支持向量数: 62
每类支持向量: [31 31]
```

---

## 7. 测验

<details>
<summary><strong>Q1: 最大间隔原则的直觉是什么？</strong></summary>

SVM找到到最近训练点（间隔）距离最大的超平面。更大间隔意味着：
- 更有信心的分类
- 对未见数据更好的泛化
- 对噪声更鲁棒

间隔是 $\frac{2}{\|w\|}$，所以最大化间隔 = 最小化 $\|w\|^2$。
</details>

<details>
<summary><strong>Q2: 什么是支持向量？为什么重要？</strong></summary>

支持向量是：
- 正好位于间隔边界上的训练点
- 拉格朗日乘子非零（$\alpha_i > 0$）的点
- 完全决定决策边界

重要性：
- 移除非支持向量不改变模型
- 决策函数只依赖支持向量
- 稀疏表示（通常少数支持向量）
</details>

<details>
<summary><strong>Q3: 解释核技巧。</strong></summary>

核技巧允许在高维特征空间计算点积而不显式计算变换。

不是：$\phi(x)^T\phi(z)$（高维时昂贵）
而是：$K(x, z)$（在原空间计算）

这有效是因为SVM优化和预测只涉及点积，可以用核评估替换。

例子：RBF核对应无限维空间但O(d)时间计算。
</details>

<details>
<summary><strong>Q4: C参数控制什么？</strong></summary>

C控制以下权衡：
- **大间隔**（小C）：优先更宽间隔，允许更多误分类
- **正确分类**（大C）：优先拟合训练数据，更小间隔

等价地，C是误分类的惩罚：
- C → 0：忽略训练误差，最大间隔
- C → ∞：硬间隔（不允许误差）
</details>

<details>
<summary><strong>Q5: 什么时候用线性核vs RBF核？</strong></summary>

**线性核**：
- 高维数据（文本、基因组）
- 稀疏特征
- 样本多（训练快）
- 数据线性可分时

**RBF核**：
- 低到中维数据
- 复杂、非线性决策边界
- 线性效果不好时
- 小数据集（RBF训练慢）
</details>

<details>
<summary><strong>Q6: RBF中的γ参数如何影响模型？</strong></summary>

γ控制每个训练样本的"影响范围"：
- **高γ**：每个点只影响附近点 → 复杂、扭曲边界 → 过拟合
- **低γ**：每个点影响远处点 → 平滑边界 → 欠拟合

经验法则：从 γ = 1/(n_features) 开始，通过交叉验证调整。
</details>

---

## 8. 参考文献

1. Cortes, C., & Vapnik, V. (1995). "Support-Vector Networks." Machine Learning.
2. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
3. Platt, J. (1998). "Sequential Minimal Optimization." Microsoft Research.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 12.
5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 7.
