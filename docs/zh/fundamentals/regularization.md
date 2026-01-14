# 正则化

## 1. 面试摘要

**关键要点：**
- **L2（岭回归）**：惩罚大权重，鼓励小权重
- **L1（Lasso）**：鼓励稀疏性，特征选择
- **弹性网络**：结合L1和L2
- **早停**：当验证误差增加时停止训练
- 正则化以偏差为代价减少方差（过拟合）

**常见面试问题：**
- "L1和L2正则化有什么区别？"
- "为什么L1产生稀疏解？"
- "早停如何起到正则化作用？"

---

## 2. 核心定义

### L2正则化（岭回归/权重衰减）
将权重的平方幅度添加到损失中：

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda \sum_i w_i^2 = \mathcal{L}_{data} + \lambda \|w\|_2^2$$

**效果：**
- 将所有权重收缩向零
- 保留所有特征，没有精确为零
- 等价于权重的高斯先验

### L1正则化（Lasso）
将权重的绝对幅度添加到损失中：

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda \sum_i |w_i| = \mathcal{L}_{data} + \lambda \|w\|_1$$

**效果：**
- 产生稀疏解（许多权重精确为零）
- 内置特征选择
- 等价于权重的拉普拉斯先验

### 弹性网络
结合L1和L2：

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$$

### 早停
当验证误差停止改善时停止训练。

---

## 3. 数学与推导

### 为什么L1产生稀疏性

考虑最小化：$\min_w (w - c)^2 + \lambda|w|$

**解：**

$$w^* = \begin{cases}
c - \lambda/2 & \text{如果 } c > \lambda/2 \\
0 & \text{如果 } |c| \leq \lambda/2 \\
c + \lambda/2 & \text{如果 } c < -\lambda/2
\end{cases}$$

当 $|c| \leq \lambda/2$ 时解**精确为零**（软阈值）。

对于L2：$w^* = c/(1 + \lambda)$ — 永远不会精确为零！

### 贝叶斯解释

**L2正则化：**
先验：$p(w) \propto \exp(-\lambda\|w\|_2^2)$ — 方差为 $1/(2\lambda)$ 的高斯

**L1正则化：**
先验：$p(w) \propto \exp(-\lambda\|w\|_1)$ — 拉普拉斯分布

带正则化的MAP估计 = 带先验的MLE。

### 岭回归闭式解

对于带L2的线性回归：

$$\hat{w}_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$$

对比OLS：$\hat{w}_{OLS} = (X^TX)^{-1}X^Ty$

$\lambda I$ 项使得即使 $X^TX$ 奇异时求逆也数值稳定。

---

## 4. 算法框架

### 选择正则化强度

```
1. 定义λ值网格（对数尺度）：[1e-4, 1e-3, ..., 1e1]
2. 对于每个λ：
   a. 用正则化训练模型
   b. 在验证集上评估
3. 选择验证性能最佳的λ
4. （可选）用选定的λ在训练+验证集上重新训练
5. 在测试集上评估
```

### 早停

```
初始化：best_val_loss = ∞，patience_counter = 0
对于每个epoch：
    在训练数据上训练
    计算验证损失

    如果 val_loss < best_val_loss：
        best_val_loss = val_loss
        保存模型检查点
        patience_counter = 0
    否则：
        patience_counter += 1

    如果 patience_counter >= patience：
        停止训练
        加载最佳检查点
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 正则化偏置项 | 偏置不贡献过拟合 | 从惩罚中排除偏置 |
| 所有特征相同λ | 特征可能有不同尺度 | 先标准化特征 |
| 正则化太强 | 欠拟合 | 用验证调整λ |
| 不用早停 | 训练太久浪费时间 | 监控验证损失 |
| 相关特征用L1 | 任意选择一个 | 改用弹性网络 |

### L1与L2比较

| 属性 | L1（Lasso） | L2（Ridge） |
|------|------------|------------|
| 稀疏性 | 是（精确零） | 否 |
| 特征选择 | 内置 | 否 |
| 相关特征 | 任意选一个 | 同等收缩所有 |
| 解唯一性 | 可能不唯一 | 总是唯一 |
| 优化 | 次梯度方法 | 线性有闭式解 |

---

## 6. 迷你示例

### Python示例：L1与L2正则化

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# 生成带一些无关特征的数据
np.random.seed(42)
X, y, true_coef = make_regression(
    n_samples=100, n_features=20, n_informative=5,
    noise=10, coef=True, random_state=42
)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拟合模型
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

ridge.fit(X_scaled, y)
lasso.fit(X_scaled, y)
elastic.fit(X_scaled, y)

# 统计非零系数
print(f"Ridge非零系数：{np.sum(ridge.coef_ != 0)}")  # 全部20
print(f"Lasso非零系数：{np.sum(np.abs(lasso.coef_) > 1e-6)}")  # ~5-7
print(f"弹性网络非零系数：{np.sum(np.abs(elastic.coef_) > 1e-6)}")  # ~7-10

print(f"\n真实信息特征：5")
print(f"Ridge收缩全部，Lasso选择子集")

# 输出：
# Ridge非零系数：20
# Lasso非零系数：6
# 弹性网络非零系数：8
```

### 早停示例

```python
import numpy as np

def train_with_early_stopping(X_train, y_train, X_val, y_val,
                              patience=5, max_epochs=100):
    """模拟带早停的训练。"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    # 模拟训练（损失来自实际训练）
    train_losses = np.exp(-np.arange(max_epochs) / 20) + 0.1
    val_losses = np.exp(-np.arange(max_epochs) / 30) + 0.15
    val_losses[40:] += np.arange(max_epochs - 40) * 0.005  # 过拟合

    for epoch in range(max_epochs):
        val_loss = val_losses[epoch]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # 这里保存检查点
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"在epoch {epoch}早停")
            print(f"最佳epoch：{best_epoch}，最佳验证损失：{best_val_loss:.4f}")
            return best_epoch

    return best_epoch

# 模拟
best = train_with_early_stopping(None, None, None, None, patience=10)
# 输出：在epoch ~50早停，最佳epoch：~40
```

---

## 7. 测验

<details>
<summary><strong>Q1: 为什么L1正则化产生稀疏解？</strong></summary>

L1的惩罚 $|w|$ 在零点创建一个不可微点，具有恒定的次梯度。优化解涉及软阈值：

$$w^* = \text{sign}(c) \cdot \max(|c| - \lambda/2, 0)$$

当无正则化解 $c$ 小于 $\lambda/2$ 时，最优 $w^*$ 精确为零。L2的平滑惩罚 $w^2$ 只将权重收缩向零但永远不会达到。
</details>

<details>
<summary><strong>Q2: 正则化的贝叶斯解释是什么？</strong></summary>

正则化对应于在权重上放置先验：
- **L2** = 高斯先验：$p(w) \propto \exp(-\lambda\|w\|^2)$
- **L1** = 拉普拉斯先验：$p(w) \propto \exp(-\lambda\|w\|_1)$

正则化损失是负对数后验：

$$\mathcal{L}_{reg} = -\log p(y|X,w) - \log p(w) = \mathcal{L}_{data} + \text{正则化}$$

最小化正则化损失 = 找到MAP估计。
</details>

<details>
<summary><strong>Q3: 什么时候选择弹性网络而不是Lasso？</strong></summary>

在以下情况使用弹性网络：
- 特征**相关**：Lasso任意选择一个，弹性网络保留组
- **特征比样本多**：Lasso最多选择n个特征，弹性网络可以选择更多
- 你想要**一些稀疏性**但也要L2的稳定性
- 有**相关特征组**你想一起选择
</details>

<details>
<summary><strong>Q4: 早停如何起到正则化作用？</strong></summary>

早停通过以下方式限制有效模型容量：
- 限制参数从初始化移动多远
- 防止模型拟合噪声（这需要更长时间）
- 在线性模型中类似于L2正则化

训练步数像一个逆正则化强度：更少步数 = 更强正则化。
</details>

<details>
<summary><strong>Q5: 应该正则化偏置项吗？</strong></summary>

通常**不**。原因：
- 偏置项移动预测而不影响模型复杂度
- 正则化偏置在真实均值非零时可能有害
- 偏置不像权重那样贡献过拟合
- sklearn、PyTorch等的标准做法是排除偏置
</details>

<details>
<summary><strong>Q6: 如何选择正则化强度λ？</strong></summary>

1. **交叉验证**：尝试多个λ值，在验证上选择最佳
2. **网格搜索**：对数尺度网格（如[1e-4, 1e-3, ..., 10]）
3. **正则化路径**：高效算法计算所有λ的解
4. **信息准则**：AIC、BIC用于模型选择
5. **贝叶斯方法**：将λ视为带先验的超参数

常见方法：在对数间隔λ值上进行5折CV。
</details>

---

## 8. 参考文献

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapters 3, 7.
2. Zou, H., & Hastie, T. (2005). "Regularization and Variable Selection via the Elastic Net." JRSS-B.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 3.3.
4. Prechelt, L. (1998). "Early Stopping - But When?" Neural Networks: Tricks of the Trade.
5. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Chapter 11.
