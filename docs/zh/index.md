# 机器学习学习笔记

欢迎使用面向面试准备和深度理解的综合机器学习学习笔记。

## 导航

### 1. 机器学习基础
- [学习范式](fundamentals/learning-paradigms.md) - 监督学习、非监督学习、自监督学习
- [数据划分与验证](fundamentals/data-splits.md) - 训练/验证/测试集、交叉验证、数据泄露
- [偏差-方差权衡](fundamentals/bias-variance.md) - 欠拟合、过拟合
- [损失函数](fundamentals/loss-functions.md) - MSE、MAE、交叉熵、校准
- [优化算法](fundamentals/optimization.md) - GD、SGD、动量、Adam、学习率调度
- [正则化](fundamentals/regularization.md) - L1、L2、早停
- [泛化与容量](fundamentals/generalization.md) - VC维直觉

### 2. 核心模型
- [线性回归](models/linear-regression.md) - OLS、岭回归、Lasso
- [逻辑回归](models/logistic-regression.md) - 决策边界、正则化
- [支持向量机](models/svm.md) - 原始/对偶问题、核技巧
- [k近邻](models/knn.md) - 距离度量、维度灾难
- [朴素贝叶斯](models/naive-bayes.md) - 概率分类
- [决策树](models/decision-trees.md) - 分裂标准、剪枝
- [集成方法](models/ensemble.md) - 随机森林、梯度提升、XGBoost
- [聚类](models/clustering.md) - K-Means、GMM、层次聚类
- [降维](models/dimensionality-reduction.md) - PCA、t-SNE、UMAP

### 3. 深度学习基础
- [神经网络基础](deep-learning/nn-fundamentals.md) - MLP、反向传播
- [卷积神经网络](deep-learning/cnn.md) - 卷积、池化
- [循环神经网络](deep-learning/rnn.md) - 序列、LSTM、GRU
- [归一化](deep-learning/normalization.md) - BatchNorm、LayerNorm、Dropout
- [深度网络训练](deep-learning/training.md) - 初始化、调试、梯度问题

### 4. 实用机器学习工程
- [评估指标](practical/metrics.md) - 精确率、召回率、F1、ROC-AUC、PR-AUC
- [类别不平衡](practical/class-imbalance.md) - 策略与技术
- [特征工程](practical/feature-engineering.md) - 预处理、特征选择
- [可解释性](practical/interpretability.md) - SHAP、排列重要性
- [数据中心问题](practical/data-centric.md) - 标签噪声、数据集偏移、异常值
- [MLOps概述](practical/mlops.md) - 可复现性、版本控制、部署

### 5. 面试准备
- [常见面试问题](interview/common-questions.md) - 问答格式
- [快速参考卡](interview/quick-reference.md) - 关键公式和概念

---

## 如何使用这些笔记

1. **面试准备**: 从"面试摘要"部分开始快速复习
2. **深入学习**: 阅读"数学与推导"获得严谨理解
3. **练习**: 使用"测验"部分测试你的知识
4. **代码**: 运行示例 `python -m ml_examples.run --demo <name>`

## 代码示例

```bash
# 运行特定示例
python -m ml_examples.run --demo linear_regression
python -m ml_examples.run --demo pca
python -m ml_examples.run --demo neural_network

# 列出所有可用示例
python -m ml_examples.run --list
```
