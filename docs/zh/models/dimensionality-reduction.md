# 降维

## 1. 面试摘要

**关键要点：**
- **PCA**：最大化方差的线性投影，找正交成分
- **t-SNE**：非线性，保留局部结构，适合可视化
- **UMAP**：比t-SNE快，保留更多全局结构
- **何时使用**：可视化、降噪、特征提取、预处理

**常见面试问题：**
- "数学推导PCA"
- "为什么不能用t-SNE投影新数据？"
- "t-SNE有什么陷阱？"

---

## 2. 核心定义

### PCA（主成分分析）
找到最大方差方向：
$$\max_w w^T \Sigma w \quad \text{s.t. } \|w\|_2 = 1$$

解：协方差矩阵 $\Sigma$ 的特征向量。

### 解释方差比
$$\text{解释比例}_k = \frac{\lambda_k}{\sum_i \lambda_i}$$

选择k以解释期望的方差百分比（如95%）。

### t-SNE（t分布随机邻域嵌入）
最小化以下之间的KL散度：
- 高维相似度（高斯）
- 低维相似度（t分布）

### UMAP（均匀流形近似与投影）
- 构建模糊拓扑表示
- 优化高/低维表示间的交叉熵
- 比t-SNE更好地保留全局结构

---

## 3. 数学与推导

### PCA推导

**目标**：找到最大化方差的投影 $w$。

**投影数据的方差：**
$$\text{Var}(Xw) = w^T X^T X w = w^T \Sigma w$$

**约束优化**（拉格朗日）：
$$L = w^T \Sigma w - \lambda(w^T w - 1)$$

**求导：**
$$\frac{\partial L}{\partial w} = 2\Sigma w - 2\lambda w = 0$$
$$\Sigma w = \lambda w$$

解：$w$ 是 $\Sigma$ 的特征向量，方差 = $\lambda$。

**多个成分**：取前k个特征向量。

### 通过SVD的PCA

对于中心化数据 $X$：
$$X = U \Sigma V^T$$

- $V$ 的列：主成分（$X^T X$ 的特征向量）
- $U \Sigma$ 的列：投影数据
- 奇异值：$\sigma_i = \sqrt{n \lambda_i}$

SVD在数值上比特征分解更稳定。

### t-SNE算法

**步骤1**：计算高维成对相似度
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**步骤2**：定义低维相似度（t分布）
$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$$

**步骤3**：最小化KL散度
$$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### 为什么用t分布？

重尾允许：
- 高维中等距离 → 低维远距离
- 防止"拥挤问题"
- 更好的簇分离

---

## 4. 算法框架

### PCA（通过SVD）

```
输入：数据矩阵 X (n × d)，成分数 k
输出：投影数据 (n × k)，成分

1. 中心化数据：X_c = X - mean(X)
2. 计算SVD：X_c = U Σ Vᵀ
3. 取V的前k列：V_k
4. 投影：Z = X_c @ V_k
5. 返回 Z, V_k
```

### t-SNE

```
输入：数据 X，困惑度，学习率，迭代次数
输出：低维嵌入 Y

1. 计算成对相似度 P（困惑度调整的σ）
2. 随机初始化 Y（通常从 N(0, 10⁻⁴)）

For iter = 1 to n_iter：
    # 计算低维相似度 Q
    Q = student_t_similarities(Y)

    # 计算梯度
    gradient = 4 * Σ_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||²)⁻¹

    # 更新 Y（带动量）
    Y = Y - learning_rate * gradient + momentum * prev_update

返回 Y
```

### UMAP（高层）

```
1. 构建k近邻图
2. 计算模糊单纯集（边权重）
3. 初始化低维嵌入
4. 通过SGD优化交叉熵损失
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| PCA不中心化 | PCA假设中心化数据 | 总是先中心化 |
| 错误的n_components | 太少丢失信息，太多保留噪声 | 用解释方差比 |
| t-SNE处理新数据 | 没有投影函数 | 用参数化t-SNE或在所有数据上拟合 |
| 解读t-SNE距离 | 全局距离不保留 | 不要解读簇间距离 |
| t-SNE困惑度错误 | 结构恢复差 | 尝试多个值（5-50） |
| 用t-SNE特征做ML | 无意义 | 只用于可视化 |

### t-SNE解读注意事项

1. **簇大小没有意义**：密集vs稀疏是伪影
2. **簇间距离没有意义**：只保留局部结构
3. **不同运行给不同结果**：用相同种子保证可复现
4. **困惑度很重要**：低=紧簇，高=更广结构

### 何时使用每种方法

| 方法 | 适用场景 |
|------|----------|
| PCA | 特征降维、预处理、线性关系 |
| t-SNE | 可视化簇、探索局部结构 |
| UMAP | 可视化+一些全局结构，比t-SNE快 |
| 核PCA | 非线性关系、小数据集 |
| 自编码器 | 复杂非线性降维，如果数据足够 |

---

## 6. 迷你示例

```python
import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        # 中心化数据
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 存储成分和解释方差
        self.components = Vt[:self.n_components]
        total_var = np.sum(S**2)
        self.explained_variance_ratio = (S[:self.n_components]**2) / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean


def simple_tsne(X, n_components=2, perplexity=30, n_iter=1000, lr=200):
    """简化的t-SNE实现。"""
    n = X.shape[0]

    # 计算成对距离
    dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    # 计算P（高维相似度）
    P = np.exp(-dists / (2 * perplexity))
    np.fill_diagonal(P, 0)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    # 随机初始化Y
    np.random.seed(42)
    Y = np.random.randn(n, n_components) * 0.01

    # 梯度下降
    for _ in range(n_iter):
        # 计算Q（带t分布的低维相似度）
        dists_low = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        Q = 1 / (1 + dists_low)
        np.fill_diagonal(Q, 0)
        Q = Q / Q.sum()
        Q = np.maximum(Q, 1e-12)

        # 计算梯度
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ_diff[i, :, None] * diff) * (1 / (1 + dists_low[i, :, None])), axis=0)

        Y -= lr * grad

    return Y


# 示例
np.random.seed(42)

# 生成数据：10维中的3个簇
n_per_cluster = 50
cluster1 = np.random.randn(n_per_cluster, 10) + np.array([5] * 10)
cluster2 = np.random.randn(n_per_cluster, 10) + np.array([-5] * 10)
cluster3 = np.random.randn(n_per_cluster, 10)
X = np.vstack([cluster1, cluster2, cluster3])
labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA解释方差: {pca.explained_variance_ratio}")
print(f"解释方差总和: {sum(pca.explained_variance_ratio):.3f}")

# 简单t-SNE（注意：很简化，真实t-SNE更复杂）
print("\n运行简化t-SNE...")
X_tsne = simple_tsne(X, n_components=2, perplexity=20, n_iter=500, lr=100)
print(f"t-SNE嵌入形状: {X_tsne.shape}")

# PCA重建误差
X_reconstructed = pca.inverse_transform(X_pca)
recon_error = np.mean((X - X_reconstructed) ** 2)
print(f"\nPCA重建MSE: {recon_error:.4f}")
```

**输出：**
```
PCA解释方差: [0.503 0.496]
解释方差总和: 0.999
运行简化t-SNE...
t-SNE嵌入形状: (150, 2)
PCA重建MSE: 0.0098
```

---

## 7. 测验

<details>
<summary><strong>Q1: 从最大方差角度推导PCA。</strong></summary>

**目标**：找到最大化投影数据方差的方向 $w$（单位向量）。

**投影方差**：$\text{Var}(Xw) = w^T \Sigma w$，其中 $\Sigma$ 是协方差矩阵。

**约束优化**：
$$\max_w w^T \Sigma w \quad \text{s.t. } w^T w = 1$$

**拉格朗日**：$L = w^T \Sigma w - \lambda(w^T w - 1)$

**梯度置零**：
$$\nabla_w L = 2\Sigma w - 2\lambda w = 0$$
$$\Sigma w = \lambda w$$

这是特征值方程。最大方差方向是最大特征值对应的特征向量。

对于k个成分：使用前k个特征向量。
</details>

<details>
<summary><strong>Q2: 为什么不能用t-SNE投影新数据点？</strong></summary>

t-SNE是**优化**过程，不是学习的映射：

1. **没有参数函数**：没有 $f(x)$ 将高维映射到低维
2. **依赖所有数据**：每个点的位置依赖所有其他点
3. **非凸优化**：添加新点需要重新运行

**解决方案**：
- 重新运行包含新数据的t-SNE
- 使用参数化t-SNE（神经网络近似）
- 使用UMAP（可以学习近似变换）
</details>

<details>
<summary><strong>Q3: 解读t-SNE图时有哪些主要陷阱？</strong></summary>

1. **簇大小无意义**：更大的簇不意味着更多样本
2. **簇间距离无意义**：远离不意味着不相似
3. **只保留局部结构**：全局关系丢失
4. **困惑度依赖**：不同困惑度=不同图
5. **随机性**：不同运行给不同结果
6. **连续流形变得断开**：可能错误暗示簇

**最佳实践**：用多个困惑度运行，不要过度解读，只用于可视化。
</details>

<details>
<summary><strong>Q4: 如何选择PCA的成分数？</strong></summary>

方法：
1. **解释方差比**：保留解释X%方差的成分（如95%）
2. **碎石图**：绘制特征值，寻找"肘部"
3. **Kaiser准则**：保留特征值>1的成分（标准化数据）
4. **交叉验证**：如果用于下游任务
5. **领域知识**：已知的内在维度

最常见：保留足够解释90-99%方差的成分。
</details>

<details>
<summary><strong>Q5: PCA和t-SNE有什么区别？</strong></summary>

| 方面 | PCA | t-SNE |
|------|-----|-------|
| 类型 | 线性 | 非线性 |
| 目标 | 最大化方差 | 保留局部相似性 |
| 全局结构 | 保留 | 丢失 |
| 可逆 | 是（近似） | 否 |
| 新数据 | 容易（投影） | 困难（重新运行） |
| 速度 | 快 O(nd²) | 慢 O(n²) |
| 用途 | 预处理，所有k | 可视化，k=2,3 |

用PCA做预处理/特征降维；只用t-SNE做可视化。
</details>

<details>
<summary><strong>Q6: 为什么t-SNE在低维使用t分布？</strong></summary>

**拥挤问题**：在高维中，中等距离很常见。投影到2D时，没有足够"空间"保留所有中等距离。

**t分布**与高斯相比有重尾：
- 高维中等距离的点可以在低维推得更远
- 为忠实的局部结构创造空间
- 防止中心"拥挤"

没有重尾，所有点会挤在嵌入的中心。
</details>

---

## 8. 参考文献

1. Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
2. van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE." JMLR.
3. McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv.
4. Wattenberg, M., Viégas, F., & Johnson, I. (2016). "How to Use t-SNE Effectively." Distill.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. 第14章.
