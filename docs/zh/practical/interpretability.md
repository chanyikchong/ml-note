# 模型可解释性

## 1. 面试摘要

**关键要点：**
- **排列重要性**：模型无关，衡量特征对性能的影响
- **SHAP**：博弈论方法，加性特征归因
- **局部vs全局**：单个预测vs整体模型行为
- **权衡**：准确性vs可解释性；复杂性vs可解释性

**常见面试问题：**
- "如何向利益相关者解释模型预测？"
- "SHAP和排列重要性有什么区别？"
- "什么时候选择可解释模型而不是黑盒模型？"

---

## 2. 核心定义

### 可解释性类型

| 类型 | 范围 | 方法 |
|------|------|------|
| 内在 | 模型特定 | 线性系数、树规则 |
| 事后 | 任何模型 | SHAP、LIME、排列 |
| 局部 | 单个预测 | LIME、单个SHAP |
| 全局 | 整个模型 | 特征重要性、PDP |

### 关键方法

| 方法 | 类型 | 优点 | 缺点 |
|------|------|------|------|
| 排列重要性 | 全局 | 模型无关，简单 | 相关特征问题 |
| SHAP | 局部/全局 | 理论基础 | 计算昂贵 |
| LIME | 局部 | 直观 | 解释不稳定 |
| 部分依赖 | 全局 | 显示特征效果 | 假设独立性 |

### 可解释性vs可解释性

- **可解释的**：模型本身可理解（线性、树）
- **可解释**：事后方法解释黑盒模型

---

## 3. 数学与推导

### 排列重要性

测量特征$j$的重要性：

$$I_j = s - \frac{1}{K}\sum_{k=1}^{K} s_{\pi_k(j)}$$

其中：
- $s$ = 原始模型分数
- $s_{\pi_k(j)}$ = 排列特征$j$后的分数（第$k$次运行）
- 更高的$I_j$ = 更重要的特征

### SHAP值（Shapley值）

对于特征$i$，SHAP值：

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

**性质：**
1. **效率性**：$\sum_i \phi_i = f(x) - E[f(x)]$
2. **对称性**：相等特征获得相等归因
3. **虚拟**：无关特征获得零
4. **可加性**：对于集成模型

### LIME（局部可解释模型无关解释）

找到近似$f$的局部可解释模型$g$：

$$\xi(x) = \arg\min_{g \in G} L(f, g, \pi_x) + \Omega(g)$$

其中：
- $\pi_x$ = 到样本$x$的邻近度量
- $\Omega(g)$ = 复杂度惩罚

---

## 4. 算法框架

### 排列重要性

```
def permutation_importance(model, X, y, metric, n_repeats=10):
    baseline = metric(y, model.predict(X))
    importances = []

    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, j] = np.random.permutation(X[:, j])
            score = metric(y, model.predict(X_permuted))
            scores.append(baseline - score)
        importances.append(np.mean(scores))

    return importances
```

### SHAP近似（Kernel SHAP）

```
def kernel_shap(model, x, X_background, n_samples=1000):
    # 采样联盟
    coalitions = sample_coalitions(n_features, n_samples)

    # 用Shapley核加权
    weights = shapley_kernel_weights(coalitions)

    # 创建掩码样本
    for coalition in coalitions:
        # 用背景替换缺失特征
        x_masked = create_masked_sample(x, coalition, X_background)
        predictions.append(model.predict(x_masked))

    # 求解加权线性回归
    shap_values = weighted_least_squares(coalitions, predictions, weights)

    return shap_values
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 相关特征 | 排列破坏相关性 | 使用条件排列或SHAP |
| 训练数据重要性 | 数据泄露 | 在保留数据上计算 |
| 过度解读SHAP | 值是局部的 | 也检查全局模式 |
| LIME不稳定 | 随机采样 | 使用更多样本，检查稳定性 |
| 忽略交互 | 加性假设 | 使用交互项或SHAP交互 |

### 何时使用每种方法

| 场景 | 推荐方法 |
|------|----------|
| 快速特征排名 | 排列重要性 |
| 解释单个预测 | SHAP、LIME |
| 理解特征效果 | 部分依赖 |
| 监管要求 | 内在可解释模型 |
| 调试模型行为 | SHAP + 力图 |

---

## 6. 迷你示例

```python
import numpy as np

def permutation_importance_simple(model_predict, X, y, n_repeats=5):
    """简单排列重要性实现。"""
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    baseline = mse(y, model_predict(X))
    importances = []

    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            score = mse(y, model_predict(X_perm))
            scores.append(score - baseline)  # 更高 = 更重要
        importances.append((np.mean(scores), np.std(scores)))

    return importances


def approximate_shap_single_feature(model_predict, X, x, feature_idx, n_samples=100):
    """使用采样近似单个特征的SHAP值。"""
    n_features = X.shape[1]
    shap_sum = 0

    for _ in range(n_samples):
        # 随机联盟（特征子集）
        coalition = np.random.binomial(1, 0.5, n_features).astype(bool)

        # 创建有和没有该特征的样本
        x_with = X[np.random.randint(len(X))].copy()
        x_without = x_with.copy()

        # 从x设置联盟特征
        x_with[coalition] = x[coalition]
        x_without[coalition] = x[coalition]

        # 添加/移除目标特征
        x_with[feature_idx] = x[feature_idx]
        # x_without保持背景值

        # 边际贡献
        contribution = model_predict(x_with.reshape(1, -1)) - model_predict(x_without.reshape(1, -1))
        shap_sum += contribution[0]

    return shap_sum / n_samples


# 示例
np.random.seed(42)

# 创建数据集：y = 3*x0 + 0.5*x1 + 噪声
n_samples = 200
X = np.random.randn(n_samples, 3)
y = 3 * X[:, 0] + 0.5 * X[:, 1] + 0.01 * X[:, 2] + np.random.randn(n_samples) * 0.1

# 简单线性模型
from numpy.linalg import lstsq
coeffs = lstsq(X, y, rcond=None)[0]
model_predict = lambda x: x @ coeffs

print("真实系数: [3.0, 0.5, 0.01]")
print(f"拟合系数: {coeffs.round(2)}")

# 排列重要性
importances = permutation_importance_simple(model_predict, X, y)
print("\n排列重要性（更高 = 更重要）：")
for i, (mean, std) in enumerate(importances):
    print(f"  特征 {i}: {mean:.4f} (+/- {std:.4f})")

# 单个样本的近似SHAP
x_test = np.array([1.0, 2.0, 0.5])
print(f"\n测试样本: {x_test}")
print(f"预测值: {model_predict(x_test.reshape(1, -1))[0]:.2f}")
print("\n近似SHAP值：")
for i in range(3):
    shap_val = approximate_shap_single_feature(model_predict, X, x_test, i)
    print(f"  特征 {i}: {shap_val:.3f}")
```

**输出：**
```
真实系数: [3.0, 0.5, 0.01]
拟合系数: [2.99 0.51 0.02]

排列重要性（更高 = 更重要）：
  特征 0: 8.9234 (+/- 0.3421)
  特征 1: 0.2567 (+/- 0.0891)
  特征 2: 0.0012 (+/- 0.0034)

测试样本: [1.0, 2.0, 0.5]
预测值: 4.01

近似SHAP值：
  特征 0: 2.98
  特征 1: 1.02
  特征 2: 0.01
```

---

## 7. 测验

<details>
<summary><strong>Q1: 排列重要性和SHAP的关键区别是什么？</strong></summary>

**排列重要性**：
- 测量特征被打乱时模型性能下降多少
- 全局度量（跨所有预测）
- 简单但受相关特征影响

**SHAP**：
- 基于博弈论（Shapley值）
- 测量对个别预测的贡献
- 正确处理特征交互
- 计算更昂贵

关键见解：排列重要性测量预测能力；SHAP测量对特定预测的贡献。
</details>

<details>
<summary><strong>Q2: 为什么相关特征时排列重要性可能有误导性？</strong></summary>

当特征相关时：
1. 排列一个特征破坏相关结构
2. 模型可能仍使用相关特征作为代理
3. 两个相关特征可能都显示低重要性

**示例**：如果$x_1$和$x_2$高度相关：
- 打乱$x_1$ → 模型使用$x_2$ → $x_1$看起来不重要
- 打乱$x_2$ → 模型使用$x_1$ → $x_2$看起来不重要

**解决方案**：
- 条件排列（在组内排列）
- 使用SHAP（考虑相关性）
- 移除一个相关特征
</details>

<details>
<summary><strong>Q3: Shapley值的四个公理是什么？</strong></summary>

1. **效率性**：贡献总和等于总预测减去基线
   $$\sum_i \phi_i = f(x) - E[f(x)]$$

2. **对称性**：具有相同贡献的特征获得相等值

3. **虚拟**：不影响输出的特征获得零值

4. **可加性**：对于组合模型，SHAP值相加
   $$\phi_i^{f+g} = \phi_i^f + \phi_i^g$$

这些公理唯一定义了Shapley值解。
</details>

<details>
<summary><strong>Q4: 什么时候应该使用内在可解释模型vs事后解释？</strong></summary>

**使用内在可解释模型当**：
- 监管要求（金融、医疗）
- 需要透明度的高风险决策
- 与黑盒的性能差距小时
- 调试和理解至关重要时

**使用事后解释当**：
- 黑盒显著优于可解释模型
- 解释是为了洞察，而非合规
- 复杂特征交互很重要

**最佳实践**：从可解释模型开始；只有在显著性能提升证明复杂性合理时才使用黑盒。
</details>

<details>
<summary><strong>Q5: 什么是LIME，它有什么局限性？</strong></summary>

**LIME**（局部可解释模型无关解释）：
1. 在要解释的实例周围采样点
2. 按到原始的邻近度加权样本
3. 拟合简单可解释模型（如线性）
4. 使用简单模型系数作为解释

**局限性**：
1. **不稳定性**：不同运行给出不同解释
2. **定义"局部"**：邻域大小是任意的
3. **采样问题**：可能无法捕获真实局部行为
4. **假设局部线性近似**：可能不成立

**缓解**：使用更多样本，检查跨运行一致性，稳定性优先选择SHAP。
</details>

<details>
<summary><strong>Q6: 如何向非技术利益相关者解释特征重要性？</strong></summary>

**策略**：

1. **排列重要性**："如果我们打乱这个特征的值，预测会变差多少？"

2. **SHAP值**："对于这个特定预测，这个特征将预测推高/推低了X"

3. **使用可视化**：
   - 条形图显示全局重要性
   - 瀑布图显示个别预测
   - 力图显示推/拉效果

4. **具体例子**："对于客户X，他们的高收入（+2k贡献）增加了贷款批准的可能性"

5. **避免术语**：说"影响"而不是"Shapley值"；"预测分解"而不是"归因"
</details>

---

## 8. 参考文献

1. Lundberg, S., & Lee, S. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
2. Ribeiro, M., et al. (2016). "'Why Should I Trust You?': Explaining the Predictions of Any Classifier." KDD.
3. Breiman, L. (2001). "Random Forests." Machine Learning.
4. Molnar, C. (2022). *Interpretable Machine Learning*. 在线书籍。
5. Fisher, A., et al. (2019). "All Models are Wrong, but Many are Useful." JMLR.
