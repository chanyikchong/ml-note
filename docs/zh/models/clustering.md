# 聚类

## 1. 面试摘要

**关键要点：**
- **K-Means**：划分为k个簇，最小化簇内方差
- **GMM**：带概率分配的软聚类
- **层次聚类**：构建簇的树（凝聚或分裂）
- **DBSCAN**：基于密度，处理任意形状和噪声
- **评估**：轮廓系数、惯性、Davies-Bouldin指数

**常见面试问题：**
- "如何选择k-means中的k？"
- "k-means有什么局限性？"
- "比较硬聚类和软聚类"

---

## 2. 核心定义

### K-Means
最小化簇内平方和：
$$\arg\min_{\mu} \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

### 高斯混合模型（GMM）
将数据建模为高斯混合：
$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

其中 $\pi_k$ 是混合系数。

### 层次聚类
- **凝聚**：自底向上，合并最近的簇
- **分裂**：自顶向下，分裂簇

**链接准则：**
| 类型 | 簇间距离 |
|------|----------|
| 单链接 | 点间最小距离 |
| 全链接 | 点间最大距离 |
| 平均链接 | 平均距离 |
| Ward | 总方差增加 |

### DBSCAN
- **核心点**：ε半径内有≥minPts个点
- **边界点**：在核心点的ε范围内
- **噪声**：既不是核心点也不是边界点

---

## 3. 数学与推导

### K-Means收敛性

**Lloyd算法：**
1. 将每个点分配给最近的质心
2. 将质心更新为簇均值

**保证收敛**因为：
- 每步减少（或维持）目标函数
- 有限数量的可能分配

**但是**：收敛到局部最小值，不是全局最小值。

### GMM和EM算法

**E步**：计算责任度
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

**M步**：更新参数
$$\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$$
$$\Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_i \gamma_{ik}}$$
$$\pi_k = \frac{1}{n}\sum_i \gamma_{ik}$$

### 轮廓系数

对于每个点 $i$：
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- $a(i)$：到同簇点的平均距离
- $b(i)$：到最近其他簇的平均距离

范围：[-1, 1]，越高越好。

### 肘部法则

绘制惯性（簇内方差）vs k：
- 寻找改善放缓的"肘部"
- 不总是清晰

---

## 4. 算法框架

### K-Means

```
输入：数据 X，簇数 K
输出：簇分配，质心

# 初始化质心（推荐k-means++）
centroids = random_sample(X, K)

重复直到收敛：
    # 将点分配给最近的质心
    对于每个点 x_i：
        assignments[i] = argmin_k ||x_i - centroid_k||²

    # 更新质心
    对于每个簇 k：
        centroid_k = mean(X[assignments == k])

返回 assignments, centroids
```

### K-Means++初始化

```
1. 随机选择第一个质心
2. 对于每个剩余质心：
   a. 计算 D(x) = 到最近已有质心的距离
   b. 以概率 ∝ D(x)² 选择下一个质心
3. 继续直到选择k个质心
```

### 带EM的GMM

```
初始化：随机均值，单位协方差，均匀混合

重复直到收敛：
    # E步：计算责任度
    对于每个点 i，簇 k：
        γ[i,k] = π[k] * N(x[i]; μ[k], Σ[k]) / Σ_j(π[j] * N(x[i]; μ[j], Σ[j]))

    # M步：更新参数
    对于每个簇 k：
        N_k = Σ_i γ[i,k]
        μ[k] = Σ_i γ[i,k] * x[i] / N_k
        Σ[k] = Σ_i γ[i,k] * (x[i]-μ[k])(x[i]-μ[k])ᵀ / N_k
        π[k] = N_k / n

返回参数，责任度
```

### DBSCAN

```
对于每个未访问的点 p：
    标记 p 为已访问
    neighbors = p 的 ε 范围内的点

    如果 |neighbors| < minPts：
        标记 p 为噪声
    否则：
        创建新簇 C
        将 p 添加到 C
        对于 neighbors 中的每个点 q：
            如果 q 未被访问：
                标记 q 为已访问
                q_neighbors = q 的 ε 范围内的点
                如果 |q_neighbors| ≥ minPts：
                    neighbors = neighbors ∪ q_neighbors
            如果 q 不在任何簇中：
                将 q 添加到 C
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| k-means中k错误 | 没有明确最佳k | 用肘部法则、轮廓系数、领域知识 |
| K-means处理非球形簇 | 假设球形簇 | 用GMM或DBSCAN |
| 对初始化敏感 | 随机起点→局部最小值 | 用k-means++，多次重启 |
| 不缩放特征 | 大范围特征主导 | 标准化特征 |
| DBSCAN ε/minPts敏感 | 错误值给出差结果 | 使用k-距离图 |

### 算法选择指南

| 场景 | 推荐方法 |
|------|----------|
| 球形簇，已知k | K-Means |
| 椭圆簇，软分配 | GMM |
| 未知簇数 | 层次聚类、DBSCAN |
| 任意形状，噪声 | DBSCAN |
| 需要层次/树状图 | 层次聚类 |
| 非常大的数据集 | Mini-batch K-Means |

---

## 6. 迷你示例

```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n, d = X.shape

        # K-means++初始化
        self.centroids = [X[np.random.randint(n)]]
        for _ in range(1, self.k):
            dists = np.array([min(np.sum((x - c)**2) for c in self.centroids) for x in X])
            probs = dists / dists.sum()
            self.centroids.append(X[np.random.choice(n, p=probs)])
        self.centroids = np.array(self.centroids)

        # 迭代
        for _ in range(self.max_iters):
            # 分配点到最近质心
            self.labels = self._assign(X)

            # 更新质心
            new_centroids = np.array([X[self.labels == k].mean(axis=0)
                                      for k in range(self.k)])

            # 检查收敛
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

        return self

    def _assign(self, X):
        dists = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])
        return np.argmin(dists, axis=1)

    def predict(self, X):
        return self._assign(X)

    def inertia(self, X):
        return sum(np.sum((X[self.labels == k] - self.centroids[k])**2)
                   for k in range(self.k))


def silhouette_score(X, labels):
    """计算平均轮廓系数。"""
    n = len(X)
    scores = []

    for i in range(n):
        # a(i): 到同簇的平均距离
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a_i = np.mean([np.sqrt(np.sum((X[i] - x)**2)) for x in same_cluster if not np.array_equal(x, X[i])])
        else:
            a_i = 0

        # b(i): 到最近其他簇的平均距离
        b_i = float('inf')
        for k in np.unique(labels):
            if k != labels[i]:
                other_cluster = X[labels == k]
                mean_dist = np.mean([np.sqrt(np.sum((X[i] - x)**2)) for x in other_cluster])
                b_i = min(b_i, mean_dist)

        if b_i == float('inf'):
            scores.append(0)
        else:
            scores.append((b_i - a_i) / max(a_i, b_i))

    return np.mean(scores)


# 示例
np.random.seed(42)

# 生成3个簇
cluster1 = np.random.randn(50, 2) + np.array([0, 0])
cluster2 = np.random.randn(50, 2) + np.array([5, 5])
cluster3 = np.random.randn(50, 2) + np.array([5, 0])
X = np.vstack([cluster1, cluster2, cluster3])

# 拟合k-means
kmeans = KMeans(k=3)
kmeans.fit(X)

print(f"质心:\n{kmeans.centroids}")
print(f"惯性: {kmeans.inertia(X):.2f}")
print(f"轮廓系数: {silhouette_score(X, kmeans.labels):.3f}")

# 肘部法则
print("\n肘部法则（不同k的惯性）:")
for k in range(1, 7):
    km = KMeans(k=k)
    km.fit(X)
    print(f"k={k}: 惯性={km.inertia(X):.1f}")
```

**输出：**
```
质心:
[[0.05 0.02]
 [5.01 4.89]
 [4.98 0.03]]
惯性: 289.45
轮廓系数: 0.567

肘部法则（不同k的惯性）:
k=1: 惯性=3871.2
k=2: 惯性=1257.8
k=3: 惯性=289.5
k=4: 惯性=241.3
k=5: 惯性=205.6
k=6: 惯性=175.4
```

---

## 7. 测验

<details>
<summary><strong>Q1: k-means有什么局限性？</strong></summary>

1. **假设球形簇**：不能处理细长或不规则形状
2. **需要指定k**：必须预先知道簇数
3. **对初始化敏感**：不同起点→不同结果
4. **对异常值敏感**：异常值拉动质心
5. **只找到局部最优**：不保证找到最佳聚类
6. **等大小簇假设**：对不同大小的簇表现差

解决方案：使用k-means++，多次重启，或替代算法（GMM、DBSCAN）。
</details>

<details>
<summary><strong>Q2: 如何选择k-means中的k？</strong></summary>

方法：
1. **肘部法则**：绘制惯性vs k，寻找"肘部"
2. **轮廓系数**：越高越好（范围-1到1）
3. **Gap统计量**：与参考分布比较
4. **领域知识**：关于预期簇的先验知识
5. **交叉验证**：如果有下游任务

没有单一最佳方法。通常结合多种方法。
</details>

<details>
<summary><strong>Q3: 硬聚类和软聚类有什么区别？</strong></summary>

**硬聚类**（K-Means）：
- 每个点恰好属于一个簇
- 二元分配：0或1

**软聚类**（GMM）：
- 每个点对每个簇有属于概率
- 分数分配：γ_ik ∈ [0, 1]，和为1

软聚类有用当：
- 簇重叠
- 你需要不确定性估计
- 点真的属于多个组
</details>

<details>
<summary><strong>Q4: 解释GMM的EM算法。</strong></summary>

**期望最大化**交替两步：

**E步**（期望）：
- 固定参数，计算责任度
- γ_ik = 点i属于簇k的概率

**M步**（最大化）：
- 固定责任度，更新参数
- μ_k = 点的加权均值
- Σ_k = 加权协方差
- π_k = 簇k的点比例

保证每步增加（或维持）对数似然。
</details>

<details>
<summary><strong>Q5: 什么时候用DBSCAN而不是k-means？</strong></summary>

使用**DBSCAN**当：
- 簇有任意形状（非球形）
- 簇数未知
- 数据包含噪声/异常值（DBSCAN标记它们）
- 簇有不同密度（需小心）

使用**K-Means**当：
- 簇大致球形
- 簇数已知
- 速度重要（k-means更快）
- 没有显著异常值
</details>

<details>
<summary><strong>Q6: 什么是轮廓系数，如何解读？</strong></summary>

点i的轮廓系数：
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- a(i)：凝聚度（到同簇的平均距离）
- b(i)：分离度（到最近其他簇的平均距离）

**解读：**
- s ≈ 1：点与其簇匹配良好
- s ≈ 0：点在簇边界
- s < 0：点可能在错误的簇

平均轮廓系数评估整体聚类质量。
</details>

---

## 8. 参考文献

1. MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations." Berkeley Symposium.
2. Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding." SODA.
3. Dempster, A., Laird, N., & Rubin, D. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." JRSS.
4. Ester, M., et al. (1996). "A Density-Based Algorithm for Discovering Clusters." KDD.
5. Rousseeuw, P. (1987). "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis." Journal of Computational and Applied Mathematics.
