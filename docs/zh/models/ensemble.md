# 集成方法

## 1. 面试摘要

**关键要点：**
- **Bagging**：在自助样本上训练，减少方差（随机森林）
- **Boosting**：顺序训练拟合残差，减少偏差（XGBoost、AdaBoost）
- **随机森林**：袋装树 + 每次分裂随机特征子集
- **梯度提升**：对损失的负梯度拟合树
- **关键超参数**：n_estimators、learning_rate、max_depth、subsample

**常见面试问题：**
- "Bagging和Boosting有什么区别？"
- "为什么随机森林效果好？"
- "逐步解释梯度提升"

---

## 2. 核心定义

### Bagging（自助聚合）
1. 创建B个自助样本（有放回抽样）
2. 在每个样本上训练模型
3. 平均预测（回归）或投票（分类）

**方差减少：**

$$\text{Var}(\bar{f}) = \frac{1}{B}\text{Var}(f) + \frac{B-1}{B}\rho \cdot \text{Var}(f)$$

其中 $\rho$ 是模型间的相关性。

### 随机森林
带额外随机性的袋装决策树：
- 每棵树在自助样本上训练
- 每次分裂，考虑随机特征子集
- 减少相关性 $\rho$ → 更好的方差减少

### Boosting
顺序组合弱学习器：

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

其中 $h_m$ 用于纠正 $F_{m-1}$ 的错误。

### AdaBoost
重新加权误分类样本：

$$w_i^{(m)} = w_i^{(m-1)} \cdot \exp(\alpha_m \cdot \mathbb{1}[y_i \neq h_m(x_i)])$$

### 梯度提升
对伪残差拟合新树：

$$r_i^{(m)} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

对于MSE损失：$r_i = y_i - F_{m-1}(x_i)$（实际残差）

---

## 3. 数学与推导

### 随机森林方差减少

对于B棵相关性为 $\rho$ 的树：

$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

- 当 $B \to \infty$：$\text{Var} \to \rho \sigma^2$
- 更低 $\rho$ → 更好（通过特征子采样实现）
- 更多树总是有帮助（添加树不会过拟合）

### 梯度提升推导

**目标**：最小化 $\sum_i L(y_i, F(x_i))$

**函数梯度下降**：

$$F_m = F_{m-1} - \eta \nabla_F L$$

**伪残差**（负梯度方向）：

$$r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

| 损失 | $L(y, F)$ | 伪残差 |
|------|-----------|--------|
| MSE | $\frac{1}{2}(y - F)^2$ | $y - F$ |
| MAE | $\|y - F\|$ | $\text{sign}(y - F)$ |
| 对数损失 | $-y\log p - (1-y)\log(1-p)$ | $y - p$ |

### XGBoost目标函数

$$\text{Obj} = \sum_i L(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

**正则化：**

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $T$：叶子数
- $w_j$：叶子权重
- $\gamma$：每个叶子的复杂度惩罚
- $\lambda$：权重的L2正则化

### 特征重要性

**MDI（平均不纯度减少）**：

$$\text{Importance}(j) = \sum_{\text{使用 } j \text{ 的节点}} p(\text{节点}) \cdot \Delta \text{不纯度}$$

**置换重要性**：

$$\text{Importance}(j) = \text{分数} - \text{置换 } j \text{ 后的分数}$$

---

## 4. 算法框架

### 随机森林

```
训练：
    For b = 1 to B:
        sample = bootstrap(训练数据)
        tree[b] = train_tree(sample, max_features=sqrt(p))

预测（分类）：
    votes = [tree[b].predict(x) for b in 1..B]
    return 多数投票(votes)

预测（回归）：
    predictions = [tree[b].predict(x) for b in 1..B]
    return mean(predictions)
```

### 梯度提升

```
初始化：F_0(x) = argmin_c sum(L(y_i, c))  # 如MSE用均值

For m = 1 to M:
    # 计算伪残差
    r_i = -dL/dF 在 F_{m-1}(x_i) 处求值

    # 对残差拟合树
    h_m = fit_tree(X, r)

    # 线搜索最优步长（可选）
    gamma_m = argmin_gamma sum(L(y_i, F_{m-1}(x_i) + gamma * h_m(x_i)))

    # 更新模型
    F_m = F_{m-1} + learning_rate * gamma_m * h_m

返回 F_M
```

### XGBoost分裂查找

```
对于每个节点：
    对于每个特征 j：
        按特征 j 排序样本
        对于每个分裂点 s：
            # 计算梯度统计量
            G_L = 左子节点的梯度和
            H_L = 左子节点的海森和
            G_R, H_R = 右边同样

            # 分数提升
            gain = 0.5 * (G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - (G_L+G_R)^2/(H_L+H_R+λ)) - γ

        选择增益最大的分裂
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| Boosting过拟合 | 迭代太多，学习率低 | 早停，正则化 |
| RF欠拟合 | 树太浅 | 增加max_depth |
| 训练慢 | 树太多/太深 | 减少n_estimators，子采样 |
| 内存问题 | 存储所有树 | 用更轻的树，剪枝 |
| 特征重要性偏差 | 相关特征分散重要性 | 使用置换重要性 |

### 超参数指南

| 方法 | 关键参数 | 调参策略 |
|------|----------|----------|
| 随机森林 | n_estimators, max_depth, max_features | 更多树很少有害；调深度 |
| 梯度提升 | n_estimators, learning_rate, max_depth | 低LR + 更多树；早停 |
| XGBoost | 同上 + reg_alpha, reg_lambda, subsample | 网格搜索+CV |

### 何时使用哪个

| 场景 | 建议 |
|------|------|
| 表格数据，快速基线 | 随机森林 |
| 需要最佳准确率，有时间 | 调参的XGBoost/LightGBM |
| 可解释性重要 | 随机森林（更简单） |
| 流式/在线学习 | 考虑其他方法 |
| 非常大的数据集 | LightGBM（基于直方图） |

---

## 6. 迷你示例

```python
import numpy as np
from collections import Counter

class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n = len(y)
        indices = np.random.choice(n, size=n, replace=True)
        return X[indices], y[indices]

    def _gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y, feature_indices):
        best_gain, best_feature, best_threshold = 0, None, None
        parent_gini = self._gini(y)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                if sum(left_mask) < 2 or sum(~left_mask) < 2:
                    continue

                weighted_gini = (sum(left_mask) * self._gini(y[left_mask]) +
                                 sum(~left_mask) * self._gini(y[~left_mask])) / len(y)
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth, n_features):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 4:
            return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}

        feature_indices = np.random.choice(X.shape[1], size=n_features, replace=False)
        feature, threshold = self._best_split(X, y, feature_indices)

        if feature is None:
            return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}

        left_mask = X[:, feature] <= threshold
        return {
            'leaf': False, 'feature': feature, 'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1, n_features),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1, n_features)
        }

    def fit(self, X, y):
        n_features = int(np.sqrt(X.shape[1])) if self.max_features == 'sqrt' else X.shape[1]
        self.trees = []
        for _ in range(self.n_trees):
            X_boot, y_boot = self._bootstrap_sample(X, y)
            tree = self._build_tree(X_boot, y_boot, 0, n_features)
            self.trees.append(tree)

    def _predict_tree(self, x, node):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        return self._predict_tree(x, node['right'])

    def predict(self, X):
        predictions = np.array([[self._predict_tree(x, tree) for tree in self.trees] for x in X])
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])


class SimpleGradientBoosting:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        residuals = y - self.initial_pred

        for _ in range(self.n_estimators):
            tree = self._fit_tree(X, residuals, 0)
            self.trees.append(tree)
            predictions = np.array([self._predict_tree(x, tree) for x in X])
            residuals = residuals - self.learning_rate * predictions

    def _fit_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 4:
            return {'leaf': True, 'prediction': np.mean(y)}

        best_mse, best_feature, best_threshold = float('inf'), None, None
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= threshold
                if sum(left_mask) < 2 or sum(~left_mask) < 2:
                    continue
                mse = (np.var(y[left_mask]) * sum(left_mask) +
                       np.var(y[~left_mask]) * sum(~left_mask))
                if mse < best_mse:
                    best_mse, best_feature, best_threshold = mse, feature, threshold

        if best_feature is None:
            return {'leaf': True, 'prediction': np.mean(y)}

        left_mask = X[:, best_feature] <= best_threshold
        return {
            'leaf': False, 'feature': best_feature, 'threshold': best_threshold,
            'left': self._fit_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._fit_tree(X[~left_mask], y[~left_mask], depth + 1)
        }

    def _predict_tree(self, x, node):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        return self._predict_tree(x, node['right'])

    def predict(self, X):
        pred = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            pred += self.learning_rate * np.array([self._predict_tree(x, tree) for x in X])
        return pred


# 示例
np.random.seed(42)
X = np.random.randn(300, 4)
y_class = (X[:, 0] + X[:, 1] > 0).astype(int)
y_reg = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(300) * 0.1

# 随机森林
rf = SimpleRandomForest(n_trees=20, max_depth=5)
rf.fit(X, y_class)
rf_acc = np.mean(rf.predict(X) == y_class)
print(f"随机森林准确率: {rf_acc:.3f}")

# 梯度提升
gb = SimpleGradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
gb.fit(X, y_reg)
gb_mse = np.mean((gb.predict(X) - y_reg) ** 2)
print(f"梯度提升MSE: {gb_mse:.4f}")
```

**输出：**
```
随机森林准确率: 0.987
梯度提升MSE: 0.0098
```

---

## 7. 测验

<details>
<summary><strong>Q1: Bagging和Boosting有什么区别？</strong></summary>

**Bagging**（自助聚合）：
- 在自助样本上**独立**训练模型
- 通过平均/投票组合
- **减少方差**，保持偏差
- 可并行化
- 例子：随机森林

**Boosting**：
- **顺序**训练模型
- 每个模型纠正前一个的错误
- **减少偏差**，可能增加方差
- 不可并行化（顺序依赖）
- 例子：AdaBoost、梯度提升、XGBoost
</details>

<details>
<summary><strong>Q2: 为什么随机森林比单棵树减少方差？</strong></summary>

两个机制：

1. **Bagging**：B个预测的平均，如果独立则方差为 $\text{Var}/B$
2. **特征子采样**：去相关树（减少 $\rho$）

组合方差：$\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$

通过使树更不相关（降低 $\rho$），方差随B增加趋近于 $\frac{\sigma^2}{B}$。

关键洞察：添加更多树永远不会过拟合（方差只会减少）。
</details>

<details>
<summary><strong>Q3: 逐步解释梯度提升。</strong></summary>

1. **初始化**：$F_0(x) = \bar{y}$（均值预测）

2. **对于每次迭代m**：
   - 计算伪残差：$r_i = -\frac{\partial L}{\partial F}|_{F=F_{m-1}}$
   - 对残差拟合树 $h_m$
   - 更新：$F_m = F_{m-1} + \eta \cdot h_m$

3. **最终模型**：$F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)$

对于MSE损失，伪残差就是 $y - F_{m-1}(x)$。

关键：我们在函数空间做梯度下降，每棵树代表一个步进方向。
</details>

<details>
<summary><strong>Q4: 梯度提升中学习率的作用是什么？</strong></summary>

学习率 $\eta$（收缩）控制步长：

$$F_m = F_{m-1} + \eta \cdot h_m$$

**效果**：
- 较低 $\eta$ → 学习更慢 → 需要更多树
- 较低 $\eta$ → 更好的泛化（正则化效果）
- 较高 $\eta$ → 更快收敛 → 过拟合风险

**最佳实践**：使用低学习率（0.01-0.1）配合基于验证性能的早停。
</details>

<details>
<summary><strong>Q5: XGBoost与标准梯度提升有什么不同？</strong></summary>

XGBoost改进：
1. **正则化**：叶子权重的L1/L2，树复杂度惩罚
2. **二阶近似**：使用海森进行更好的分裂
3. **稀疏感知**：原生处理缺失值
4. **并行处理**：并行特征扫描
5. **树剪枝**：剪掉负增益的分裂
6. **硬件优化**：缓存感知，核外计算

这些使XGBoost比sklearn的GradientBoostingClassifier更快、更正则化。
</details>

<details>
<summary><strong>Q6: 什么时候倾向于随机森林而不是梯度提升？</strong></summary>

**倾向于随机森林当**：
- 需要快速、可靠的基线
- 想要不需要太多调参的稳健模型
- 并行化重要
- 需要可解释性（特征重要性更清晰）
- 训练数据有噪声（RF更鲁棒）

**倾向于梯度提升当**：
- 需要最佳可能的准确率
- 有时间进行超参数调优
- 数据集干净且准备充分
- 可以用早停进行正则化

实践中，调参的XGBoost/LightGBM通常胜过随机森林，但需要更多努力。
</details>

---

## 8. 参考文献

1. Breiman, L. (2001). "Random Forests." Machine Learning.
2. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics.
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD.
4. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NIPS.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. 第10、15章.
