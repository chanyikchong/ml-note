# Machine Learning Study Notes

Welcome to the comprehensive Machine Learning study notes designed for interview preparation and deep understanding.

## Navigation

### 1. ML Fundamentals
- [Learning Paradigms](fundamentals/learning-paradigms.md) - Supervised, Unsupervised, Self-supervised
- [Data Splits & Validation](fundamentals/data-splits.md) - Train/Val/Test, Cross-validation, Leakage
- [Bias-Variance Tradeoff](fundamentals/bias-variance.md) - Underfitting, Overfitting
- [Loss Functions](fundamentals/loss-functions.md) - MSE, MAE, Cross-entropy, Calibration
- [Optimization](fundamentals/optimization.md) - GD, SGD, Momentum, Adam, Learning Rate Schedules
- [Regularization](fundamentals/regularization.md) - L1, L2, Early Stopping
- [Generalization & Capacity](fundamentals/generalization.md) - VC Dimension Intuition

### 2. Core Models
- [Linear Regression](models/linear-regression.md) - OLS, Ridge, Lasso
- [Logistic Regression](models/logistic-regression.md) - Decision Boundary, Regularization
- [Support Vector Machines](models/svm.md) - Primal/Dual, Kernel Trick
- [k-Nearest Neighbors](models/knn.md) - Distance Metrics, Curse of Dimensionality
- [Naive Bayes](models/naive-bayes.md) - Probabilistic Classification
- [Decision Trees](models/decision-trees.md) - Splitting Criteria, Pruning
- [Ensemble Methods](models/ensemble.md) - Random Forests, Gradient Boosting, XGBoost
- [Clustering](models/clustering.md) - K-Means, GMM, Hierarchical
- [Dimensionality Reduction](models/dimensionality-reduction.md) - PCA, t-SNE, UMAP

### 3. Deep Learning Basics
- [Neural Network Fundamentals](deep-learning/nn-fundamentals.md) - MLP, Backprop
- [Convolutional Neural Networks](deep-learning/cnn.md) - Convolutions, Pooling
- [Recurrent Neural Networks](deep-learning/rnn.md) - Sequences, LSTM, GRU
- [Normalization](deep-learning/normalization.md) - BatchNorm, LayerNorm, Dropout
- [Training Deep Networks](deep-learning/training.md) - Initialization, Debugging, Gradient Issues

### 4. Practical ML Engineering
- [Evaluation Metrics](practical/metrics.md) - Precision, Recall, F1, ROC-AUC, PR-AUC
- [Class Imbalance](practical/class-imbalance.md) - Strategies and Techniques
- [Feature Engineering](practical/feature-engineering.md) - Preprocessing, Feature Selection
- [Interpretability](practical/interpretability.md) - SHAP, Permutation Importance
- [Data-Centric Issues](practical/data-centric.md) - Label Noise, Dataset Shift, Outliers
- [MLOps Overview](practical/mlops.md) - Reproducibility, Versioning, Deployment

### 5. Interview Preparation
- [Common Interview Questions](interview/common-questions.md) - Q&A Format
- [Quick Reference Card](interview/quick-reference.md) - Key Formulas and Concepts

---

## How to Use These Notes

1. **Interview Prep**: Start with "Interview Summary" sections for quick review
2. **Deep Dive**: Read "Math and Derivations" for rigorous understanding
3. **Practice**: Use "Quiz" sections to test your knowledge
4. **Code**: Run examples with `python -m ml_examples.run --demo <name>`

## Code Examples

```bash
# Run specific demo
python -m ml_examples.run --demo linear_regression
python -m ml_examples.run --demo pca
python -m ml_examples.run --demo neural_network

# List all available demos
python -m ml_examples.run --list
```
