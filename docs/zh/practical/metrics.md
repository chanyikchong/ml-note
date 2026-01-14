# 评估指标

## 1. 面试摘要

**关键要点：**
- **准确率**：整体正确性，类别不平衡时有误导
- **精确率**：预测为正的中有多少是正确的
- **召回率**：实际为正的中有多少被找到
- **F1**：精确率和召回率的调和平均
- **ROC-AUC**：排序质量，与阈值无关
- **PR-AUC**：对不平衡数据更好

**常见面试问题：**
- "什么时候准确率不是好指标？"
- "解释精确率和召回率的权衡"
- "什么时候用PR-AUC而不是ROC-AUC？"

---

## 2. 核心定义

### 混淆矩阵
```
                预测
                负    正
实际  负       TN    FP
      正       FN    TP
```

### 分类指标

| 指标 | 公式 | 解释 |
|------|------|------|
| 准确率 | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确性 |
| 精确率 | $\frac{TP}{TP + FP}$ | 阳性预测值 |
| 召回率 | $\frac{TP}{TP + FN}$ | 真阳性率，敏感性 |
| 特异性 | $\frac{TN}{TN + FP}$ | 真阴性率 |
| F1分数 | $\frac{2 \cdot Prec \cdot Rec}{Prec + Rec}$ | 调和平均 |

### 回归指标

| 指标 | 公式 | 注意 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 惩罚大误差 |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 对异常值鲁棒 |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解释方差 |

---

## 3. 数学与推导

### ROC曲线

在不同阈值下绘制真阳性率vs假阳性率：
- **TPR**（y轴）= $\frac{TP}{TP + FN}$ = 召回率
- **FPR**（x轴）= $\frac{FP}{FP + TN}$ = 1 - 特异性

**AUC解释：**
- AUC = 1.0：完美分类器
- AUC = 0.5：随机分类器
- AUC < 0.5：比随机差

**概率解释：**
$$\text{AUC} = P(\text{score}_{正例} > \text{score}_{负例})$$

### PR曲线

绘制精确率vs召回率：
- 对不平衡数据更有信息量
- 关注正类性能

### F-beta分数

广义F1允许精确率-召回率加权：
$$F_\beta = \frac{(1 + \beta^2) \cdot Prec \cdot Rec}{\beta^2 \cdot Prec + Rec}$$

- $\beta = 1$：等权重（F1）
- $\beta = 2$：召回率重要性是精确率的两倍
- $\beta = 0.5$：精确率重要性是召回率的两倍

---

## 4. 算法框架

### 选择指标

```
如果类别平衡：
    → 准确率是合理的
    → F1作为单一数字
    → ROC-AUC用于排序

如果类别不平衡：
    → 避免准确率！
    → 使用精确率-召回率
    → 使用PR-AUC

如果成本敏感：
    → 自定义加权指标
    → 考虑业务成本矩阵

对于回归：
    如果异常值重要：→ MSE
    如果应忽略异常值：→ MAE
    如果相对拟合重要：→ R²
```

### 阈值选择

```
1. 训练带概率输出的分类器
2. 在不同阈值计算精确率和召回率
3. 绘制精确率-召回率曲线
4. 基于以下选择阈值：
   - 业务需求
   - FP vs FN的成本
   - 期望的精确率/召回率权衡
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 不平衡时使用准确率 | 1%正类也能99%准确率 | 使用F1、PR-AUC |
| 忽略阈值选择 | 默认0.5可能不是最优 | 基于需求调整 |
| 严重不平衡时用ROC-AUC | ROC可能有误导 | 改用PR-AUC |
| 错误平均指标 | Micro vs macro vs weighted | 理解每种类型 |
| 优化错误指标 | 用准确率训练，用AUC评估 | 训练和评估用相同指标 |

### ROC-AUC何时失效

严重类别不平衡时（如99.9%负例）：
- ROC可能因大量真负例而看起来好
- PR曲线更好地显示稀有正例的性能
- 预测"全部为负"的模型FPR为0，在ROC上看起来好

---

## 6. 迷你示例

```python
import numpy as np

def compute_metrics(y_true, y_pred):
    """计算分类指标。"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_roc_auc(y_true, y_scores):
    """使用梯形法则计算ROC-AUC。"""
    # 按分数降序排序
    order = np.argsort(y_scores)[::-1]
    y_true = y_true[order]

    # 在每个阈值计算TPR和FPR
    tpr = np.cumsum(y_true) / np.sum(y_true)
    fpr = np.cumsum(1 - y_true) / np.sum(1 - y_true)

    # 用梯形法则计算AUC
    auc = np.trapz(tpr, fpr)
    return auc

# 示例
np.random.seed(42)
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.7, 0.3, 0.4, 0.85, 0.15])

metrics = compute_metrics(y_true, y_pred)
print(f"准确率: {metrics['accuracy']:.3f}")
print(f"精确率: {metrics['precision']:.3f}")
print(f"召回率: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"ROC-AUC: {compute_roc_auc(y_true, y_scores):.3f}")
```

---

## 7. 测验

<details>
<summary><strong>Q1: 什么时候准确率是差的指标？</strong></summary>

类别不平衡时准确率失效。例如：
- 数据集：99%负例，1%正例
- 预测"全部为负"的模型：99%准确率！
- 但对找正例没用

改用精确率、召回率、F1或PR-AUC。
</details>

<details>
<summary><strong>Q2: 解释精确率-召回率权衡。</strong></summary>

随着阈值增加：
- **精确率增加**：更少预测，但更有信心
- **召回率减少**：遗漏更多实际正例

权衡取决于成本：
- 需要高精确率：垃圾邮件检测（不要丢失真邮件）
- 需要高召回率：医学诊断（不要漏诊）
</details>

<details>
<summary><strong>Q3: 什么时候应该用PR-AUC而不是ROC-AUC？</strong></summary>

使用**PR-AUC**当：
- 类别不平衡严重（如<10%正例）
- 你主要关心正类
- 真负例不太重要

使用**ROC-AUC**当：
- 类别大致平衡
- 正负预测都重要
- 你想要与阈值无关的比较
</details>

<details>
<summary><strong>Q4: ROC-AUC从概率上衡量什么？</strong></summary>

ROC-AUC = 随机选择的正例比随机选择的负例得分更高的概率。

$$\text{AUC} = P(\text{score}_{正例} > \text{score}_{负例})$$

AUC = 1意味着完美排序；AUC = 0.5意味着随机排序。
</details>

<details>
<summary><strong>Q5: 解释micro、macro和weighted平均。</strong></summary>

对于多分类：
- **Micro**：全局计算指标（求和所有TP、FP、FN）
- **Macro**：每类计算，然后平均（平等对待所有类别）
- **Weighted**：每类计算，按支持度加权平均

当所有类别同等重要时用macro，无论大小。
当你想反映类别分布时用weighted。
</details>

<details>
<summary><strong>Q6: 什么是F-beta分数？什么时候用？</strong></summary>

$$F_\beta = \frac{(1 + \beta^2) \cdot Prec \cdot Rec}{\beta^2 \cdot Prec + Rec}$$

- F1（$\beta=1$）：等重要性
- F2（$\beta=2$）：召回率重要性是精确率的两倍
- F0.5（$\beta=0.5$）：精确率重要性是召回率的两倍

当你想基于业务需求明确加权精确率vs召回率时使用。
</details>

---

## 8. 参考文献

1. Davis, J., & Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves." ICML.
2. Saito, T., & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." PLOS ONE.
3. Powers, D. (2011). "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation." JMLT.
4. Fawcett, T. (2006). "An Introduction to ROC Analysis." Pattern Recognition Letters.
5. scikit-learn文档：指标和评分。
