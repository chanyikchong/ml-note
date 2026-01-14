# 决策树

## 1. 面试摘要

**关键要点：**
- **递归划分**：将特征空间分成矩形区域
- **分裂准则**：基尼不纯度、熵（信息增益）、回归用MSE
- **贪心算法**：每个节点选择局部最优分裂
- **可解释**：易于可视化和解释决策
- **容易过拟合**：深树会记住训练数据

**常见面试问题：**
- "什么是信息增益，如何计算？"
- "比较基尼不纯度和熵"
- "如何防止决策树过拟合？"

---

## 2. 核心定义

### 树结构
- **根节点**：顶部节点，包含所有数据
- **内部节点**：带分裂规则的决策点
- **叶节点**：终端节点，包含预测
- **深度**：从根到任意叶的最大路径长度

### 分裂准则（分类）

**基尼不纯度：**

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

**熵：**

$$H = -\sum_{k=1}^{K} p_k \log_2 p_k$$

**信息增益：**

$$IG = H(\text{父节点}) - \sum_{\text{子节点}} \frac{n_{\text{子}}}{n_{\text{父}}} H(\text{子节点})$$

### 分裂准则（回归）

**均方误差：**

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

分裂以最小化子节点的加权MSE。

---

## 3. 数学与推导

### 基尼不纯度推导

基尼衡量随机选择元素被错误分类的概率：

$$G = \sum_{k=1}^{K} p_k (1 - p_k) = 1 - \sum_{k=1}^{K} p_k^2$$

对于正类概率为 $p$ 的二分类：

$$G = 2p(1-p)$$

- $G = 0$：纯节点（全部同类）
- $G = 0.5$：最大不纯度（50-50分裂）

### 熵 vs 基尼

| 属性 | 熵 | 基尼 |
|------|-----|------|
| 范围 | [0, log₂K] | [0, 1-1/K] |
| 纯节点 | 0 | 0 |
| 最大值 | log₂K（均匀） | 1-1/K |
| 计算 | 较慢（log） | 较快 |
| 敏感性 | 对类别分布更敏感 | 不太敏感 |

实践中，它们通常产生相似的树。

### 最优分裂搜索

对于取值 $\{v_1, ..., v_m\}$ 的特征 $x_j$：
1. 排序唯一值
2. 尝试中点分裂：$(v_i + v_{i+1})/2$
3. 选择信息增益最大的分裂

**每次分裂复杂度：** 排序 $O(n \log n)$ + 评估 $O(n)$

### 为什么贪心是次优的

贪心算法找到局部最优分裂，不是全局最优树。

例子：XOR问题
- 没有单个特征分裂有帮助
- 但分裂的组合可以解决

找到最优树是NP完全问题。

---

## 4. 算法框架

### CART算法（分类与回归树）

```
BuildTree(data, depth):
    # 基本情况
    if depth >= max_depth or |data| < min_samples:
        return LeafNode(多数类 或 均值)

    if 所有标签相同:
        return LeafNode(标签)

    # 找最佳分裂
    best_gain = 0
    for 每个特征 j:
        for 每个阈值 t:
            gain = compute_gain(data, j, t)
            if gain > best_gain:
                best_gain = gain
                best_split = (j, t)

    if best_gain == 0:
        return LeafNode(预测)

    # 分裂数据
    left_data = data where x[j] <= t
    right_data = data where x[j] > t

    # 递归
    left_child = BuildTree(left_data, depth + 1)
    right_child = BuildTree(right_data, depth + 1)

    return InternalNode(best_split, left_child, right_child)
```

### 预测

```
Predict(node, x):
    if node 是 LeafNode:
        return node.prediction

    if x[node.feature] <= node.threshold:
        return Predict(node.left_child, x)
    else:
        return Predict(node.right_child, x)
```

### 剪枝（后剪枝）

```
Prune(node, validation_data):
    if node 是 LeafNode:
        return node

    # 递归剪枝子节点
    node.left = Prune(node.left, validation_data)
    node.right = Prune(node.right, validation_data)

    # 尝试用叶子替换子树
    original_error = evaluate(node, validation_data)
    pruned_node = LeafNode(majority_prediction(node))
    pruned_error = evaluate(pruned_node, validation_data)

    if pruned_error <= original_error + alpha:
        return pruned_node
    return node
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 过拟合 | 树生长太深 | 剪枝，设置max_depth，min_samples |
| 轴对齐分裂 | 不能捕获对角边界 | 使用斜向树或集成 |
| 不稳定 | 小数据变化→不同树 | 使用集成（随机森林） |
| 偏向多值特征 | 更多分裂点可尝试 | 使用信息增益率 |
| 类别不平衡 | 多数类主导 | 类别权重，平衡采样 |

### 正则化参数

| 参数 | 效果 | 典型值 |
|------|------|--------|
| max_depth | 限制树深度 | 3-20 |
| min_samples_split | 分裂最小样本 | 2-20 |
| min_samples_leaf | 叶子最小样本 | 1-10 |
| max_features | 每次分裂考虑的特征 | sqrt(n), log2(n), n |
| min_impurity_decrease | 需要的最小增益 | 0.0-0.1 |

---

## 6. 迷你示例

```python
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def _gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain = 0
        best_feature, best_threshold = None, None
        parent_gini = self._gini(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if sum(left_mask) < self.min_samples or sum(right_mask) < self.min_samples:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])

                n_left, n_right = sum(left_mask), sum(right_mask)
                weighted_gini = (n_left * left_gini + n_right * right_gini) / len(y)
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        # 基本情况
        if depth >= self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1:
            return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}

        feature, threshold, gain = self._best_split(X, y)

        if feature is None:
            return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}

        left_mask = X[:, feature] <= threshold
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# 示例
np.random.seed(42)
X = np.random.randn(200, 2)
y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)  # 第一象限 vs 其他

tree = DecisionTree(max_depth=3)
tree.fit(X, y)
predictions = tree.predict(X)
accuracy = np.mean(predictions == y)
print(f"训练准确率: {accuracy:.3f}")

# 测试新点
X_test = np.array([[1, 1], [-1, -1], [1, -1]])
print(f"测试预测: {tree.predict(X_test)}")
```

**输出：**
```
训练准确率: 0.985
测试预测: [1 0 0]
```

---

## 7. 测验

<details>
<summary><strong>Q1: 基尼不纯度和熵有什么区别？</strong></summary>

两者都衡量节点不纯度：

**基尼**：$G = 1 - \sum p_k^2$
- 随机元素被错误分类的概率
- 计算更快（无对数）
- 范围：[0, 1-1/K]

**熵**：$H = -\sum p_k \log_2 p_k$
- 信息论度量
- 对类别分布变化更敏感
- 范围：[0, log₂K]

实践中，它们产生相似的树。基尼因速度通常更受欢迎。
</details>

<details>
<summary><strong>Q2: 为什么决策树容易过拟合？</strong></summary>

决策树可以：
1. **任意深生长**：完全记住训练数据
2. **创建复杂边界**：每个叶子可以是单个点
3. **拟合噪声**：每个噪声点都可以有自己的路径

max_depth=∞ 的树达到100%训练准确率但泛化差。

**解决方案**：限制max_depth，要求min_samples_split，构建后剪枝，使用集成。
</details>

<details>
<summary><strong>Q3: 什么是信息增益，如何使用？</strong></summary>

信息增益衡量分裂带来的不纯度减少：

$$IG = H(\text{父节点}) - \sum_{\text{子节点}} \frac{n_{\text{子}}}{n_{\text{父}}} H(\text{子节点})$$

更高IG = 更好分裂（更多不纯度减少）

**用法**：在每个节点，尝试所有特征和阈值，选择信息增益最大的。

**注意**：IG偏向多值特征。信息增益率（C4.5）解决这个问题。
</details>

<details>
<summary><strong>Q4: 决策树如何处理连续和类别特征？</strong></summary>

**连续特征**：
- 尝试所有可能的阈值分裂：$x_j \leq t$ vs $x_j > t$
- 通常测试排序唯一值之间的中点
- 仅二元分裂

**类别特征**：
- 可以二元分裂（子集 vs 补集）
- 或多路分裂（每个类别一个分支）
- CART只用二元；ID3/C4.5允许多路

高基数类别特征的多路分裂可能导致过拟合。
</details>

<details>
<summary><strong>Q5: 什么是剪枝，为什么重要？</strong></summary>

剪枝移除不改善泛化的子树：

**预剪枝**（早停）：
- 增益低于阈值时停止生长
- 节点样本太少时停止
- 限制max_depth

**后剪枝**（错误减少剪枝）：
- 先生长完整树
- 移除不改善验证误差的子树
- 更准确但更慢

剪枝防止过拟合同时保留有用结构。
</details>

<details>
<summary><strong>Q6: 为什么决策树不能轻易捕获对角决策边界？</strong></summary>

标准树使用**轴对齐分裂**：$x_j \leq t$

要近似对角边界 $x_1 + x_2 = 0$：
- 需要许多水平和垂直分裂
- 创建阶梯模式
- 需要深树

**解决方案**：
- **斜向树**：允许 $w_1 x_1 + w_2 x_2 \leq t$ 这样的分裂
- **随机森林**：集成平滑边界
- **特征工程**：创建 $x_{new} = x_1 + x_2$
</details>

---

## 8. 参考文献

1. Breiman, L., et al. (1984). *Classification and Regression Trees*. Wadsworth.
2. Quinlan, J. R. (1986). "Induction of Decision Trees." Machine Learning.
3. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. 第9章.
5. scikit-learn文档：决策树。
