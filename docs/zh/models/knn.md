# k近邻算法

## 1. 面试摘要

**关键要点：**
- **基于实例的学习**：无显式训练，存储所有数据
- **距离度量**：欧氏距离、曼哈顿距离或自定义距离
- **k参数**：考虑的邻居数量
- **懒惰学习器**：所有计算在预测时进行
- **维度灾难**：在高维空间性能下降

**常见面试问题：**
- "kNN预测的时间复杂度是多少？"
- "如何选择k值？"
- "为什么kNN在高维数据上效果不好？"

---

## 2. 核心定义

### 算法
对于新点 $x$：
1. 计算到所有训练点的距离
2. 找到 $k$ 个最近邻
3. 分类：多数投票
4. 回归：邻居值的平均

### 距离度量

| 度量 | 公式 | 适用场景 |
|------|------|----------|
| 欧氏 | $\sqrt{\sum_i (x_i - y_i)^2}$ | 连续、已缩放的特征 |
| 曼哈顿 | $\sum_i |x_i - y_i|$ | 网格数据、对异常值鲁棒 |
| 闵可夫斯基 | $(\sum_i |x_i - y_i|^p)^{1/p}$ | 泛化（p=1：曼哈顿，p=2：欧氏） |
| 余弦 | $1 - \frac{x \cdot y}{\|x\|\|y\|}$ | 文本、高维稀疏 |

### 加权kNN
按距离倒数加权邻居：

$$\hat{y} = \frac{\sum_{i \in N_k} w_i y_i}{\sum_{i \in N_k} w_i}, \quad w_i = \frac{1}{d(x, x_i)}$$

---

## 3. 数学与推导

### k值的偏差-方差权衡

**小k（如k=1）：**
- 低偏差：捕获局部模式
- 高方差：对噪声敏感
- 决策边界复杂

**大k：**
- 高偏差：过度平滑，遗漏局部模式
- 低方差：预测更稳定
- 决策边界更平滑

### 最优k选择
- 使用交叉验证
- 经验法则：$k \approx \sqrt{n}$
- 二分类用奇数k（避免平局）

### 维度灾难

在高维空间：
- 所有点变得等距
- 最近邻并不比最远邻近多少

对于 $[0,1]^d$ 中的均匀分布：

$$\frac{\text{dist}_{max} - \text{dist}_{min}}{\text{dist}_{min}} \to 0 \text{ 当 } d \to \infty$$

超球体相对于超立方体的体积呈指数衰减。

---

## 4. 算法框架

### 基本kNN

```
输入：训练数据 (X, y)，查询点 x_q，k
输出：预测标签/值

1. 计算距离：
   对于每个训练点 x_i：
       d_i = distance(x_q, x_i)

2. 找k个最近：
   indices = argsort(distances)[:k]

3. 聚合：
   分类：返回 mode(y[indices])
   回归：返回 mean(y[indices])
```

### 使用KD树的高效kNN

```
构建阶段：
1. 选择方差最大的轴
2. 在中位数处分割
3. 递归构建左/右子树

查询阶段：
1. 遍历树到包含查询点的叶子
2. 回溯，剪枝比第k近更远的分支
3. 平均情况：每次查询 O(log n)
```

### 球树（用于高维）

```
构建阶段：
1. 找到包含所有点的质心和半径
2. 分成两个球
3. 递归划分

查询阶段：
- 当 d > 20 时比KD树更好
- 平均情况 O(d * log n)
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 不缩放特征 | 范围大的特征主导 | 标准化所有特征 |
| 错误k值 | 默认k=5可能不是最优 | 交叉验证k |
| 预测慢 | 每次查询 O(nd) | 使用KD树/球树 |
| 高维度 | 距离变得无意义 | 先降维 |
| 类别不平衡 | 多数类主导 | 使用加权投票 |

### 何时使用kNN

| 场景 | 建议 |
|------|------|
| 小数据集，低维度 | 适合 |
| 需要可解释预测 | 适合（显示邻居） |
| 大数据集 | 考虑近似最近邻 |
| 高维度（>50） | 与降维一起使用 |
| 需要快速预测 | 避免（或用树结构） |

---

## 6. 迷你示例

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        # 计算距离
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # 获取k个最近的索引
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        # 多数投票
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# 示例
np.random.seed(42)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

X_test = np.array([[0.5, 0.5], [-0.5, -0.5], [1.0, -1.0]])

knn = KNN(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(f"测试点: {X_test}")
print(f"预测: {predictions}")

# k的影响
for k in [1, 3, 5, 10, 20]:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    train_acc = np.mean(knn.predict(X_train) == y_train)
    print(f"k={k:2d}: 训练准确率 = {train_acc:.3f}")
```

**输出：**
```
测试点: [[ 0.5  0.5] [-0.5 -0.5] [ 1.  -1. ]]
预测: [1 0 0]
k= 1: 训练准确率 = 1.000
k= 3: 训练准确率 = 0.970
k= 5: 训练准确率 = 0.960
k=10: 训练准确率 = 0.950
k=20: 训练准确率 = 0.940
```

---

## 7. 测验

<details>
<summary><strong>Q1: kNN预测的时间复杂度是多少？</strong></summary>

无优化时：
- **训练**：O(1) - 只存储数据
- **预测**：每次查询 O(nd) - 计算到所有n个d维点的距离

使用KD树/球树：
- **构建**：O(n log n)
- **预测**：平均情况 O(log n)（高维时退化到 O(n)）
</details>

<details>
<summary><strong>Q2: k的选择如何影响偏差和方差？</strong></summary>

- **k=1**：训练误差为零，高方差，低偏差。对噪声非常敏感。
- **大k**：边界更平滑，高偏差，低方差。可能欠拟合。
- **k=n**：对所有预测多数类（最大偏差）。

最优k平衡偏差-方差权衡。使用交叉验证找到它。
</details>

<details>
<summary><strong>Q3: 为什么特征缩放对kNN很重要？</strong></summary>

kNN使用距离度量。尺度较大的特征主导距离计算。

例子：如果特征A在[0, 1000]，特征B在[0, 1]：
- 距离被特征A主导
- 特征B实际上被忽略

解决方案：标准化（z分数）或归一化（min-max）所有特征。
</details>

<details>
<summary><strong>Q4: 什么是维度灾难，它如何影响kNN？</strong></summary>

在高维空间：
1. **距离集中**：所有成对距离变得相似
2. **稀疏数据**：数据点分散，最近邻很远
3. **体积**：大部分体积靠近超立方体表面

对kNN的影响：
- "最近"邻并不真正接近
- 随着d增加需要指数级更多数据
- 性能显著下降

解决方案：在kNN之前使用降维（PCA、t-SNE）。
</details>

<details>
<summary><strong>Q5: 如何处理kNN分类中的平局？</strong></summary>

打破平局的方法：
1. **使用奇数k**用于二分类
2. **距离加权**：更近的邻居权重更大
3. **随机选择**：在平局类别中
4. **减小k**：直到平局打破

最佳实践：使用带距离倒数权重的加权kNN。
</details>

<details>
<summary><strong>Q6: 什么时候用曼哈顿距离而不是欧氏距离？</strong></summary>

**曼哈顿（L1）**：
- 网格状移动（出租车距离）
- 高维稀疏数据
- 对异常值更鲁棒
- 特征不相关时

**欧氏（L2）**：
- 连续、密集特征
- 实际几何距离重要时
- 特征尺度相似

曼哈顿对高维数据通常更好，因为它不通过平方放大异常值。
</details>

---

## 8. 参考文献

1. Cover, T., & Hart, P. (1967). "Nearest Neighbor Pattern Classification." IEEE Transactions on Information Theory.
2. Friedman, J., Bentley, J., & Finkel, R. (1977). "An Algorithm for Finding Best Matches in Logarithmic Expected Time." ACM TOMS.
3. Beyer, K., et al. (1999). "When Is 'Nearest Neighbor' Meaningful?" ICDT.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. 第13章.
5. scikit-learn文档：最近邻。
