# Machine Learning Study Notes

A comprehensive, bilingual (English/中文) Machine Learning study note system that is interview-ready, mathematically rigorous, easy to navigate, and continuously extensible.

## Features

- **Bilingual Documentation**: Full coverage in English and Chinese
- **Interview-Ready**: Each topic includes interview summaries and common questions
- **Mathematically Rigorous**: Complete derivations and proofs where appropriate
- **Runnable Code Examples**: Python implementations for all key algorithms
- **Interactive Quizzes**: 5+ quiz questions per topic with hidden answers
- **Static Site**: Generate a navigable website with math rendering (KaTeX)

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run a code demo
python -m ml_examples.run --demo linear_regression
python -m ml_examples.run --demo pca
python -m ml_examples.run --list  # See all demos

# Validate documentation
python -m validators.main

# Build static site
python -m site_generator.build
python -m site_generator.build --serve  # Build and serve locally

# Extract skills
python -m skills.extract
```

## Project Structure

```
ML/
├── docs/
│   ├── en/                          # English documentation
│   │   ├── index.md                 # Main index
│   │   ├── fundamentals/            # ML fundamentals
│   │   │   ├── learning-paradigms.md
│   │   │   ├── data-splits.md
│   │   │   ├── bias-variance.md
│   │   │   ├── loss-functions.md
│   │   │   ├── optimization.md
│   │   │   ├── regularization.md
│   │   │   └── generalization.md
│   │   ├── models/                  # Core ML models
│   │   │   ├── linear-regression.md
│   │   │   ├── logistic-regression.md
│   │   │   └── ...
│   │   ├── deep-learning/           # Deep learning basics
│   │   ├── practical/               # Practical ML engineering
│   │   └── interview/               # Interview preparation
│   └── zh/                          # Chinese documentation (mirrors en/)
│
├── ml_examples/                     # Python code examples
│   ├── __init__.py
│   ├── run.py                       # Main entry point
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── svm.py
│   ├── decision_tree.py
│   ├── ensemble.py
│   ├── clustering.py
│   ├── pca.py
│   ├── neural_network.py
│   └── calibration.py
│
├── site_generator/                  # Static site generator
│   ├── __init__.py
│   └── build.py
│
├── validators/                      # Documentation validators
│   ├── __init__.py
│   └── main.py
│
├── skills/                          # Skills extraction
│   ├── __init__.py
│   ├── template.yaml
│   └── extract.py
│
├── qa_workflow/                     # Q&A and patch workflow
│   └── README.md
│
├── _site/                           # Generated static site (gitignored)
├── config.yaml                      # Project configuration
├── pyproject.toml                   # Python package config
├── CLAUDE.md                        # AI assistant instructions
└── README.md                        # This file
```

## Documentation Structure

Each topic page follows this structure:

1. **Interview Summary** - Key points and common questions
2. **Core Definitions** - Precise definitions and terminology
3. **Math and Derivations** - Mathematical foundations
4. **Algorithm Sketch** - Pseudocode and implementation details
5. **Common Pitfalls** - Mistakes to avoid
6. **Mini Example** - Runnable code demonstration
7. **Quiz** - 5+ questions with hidden answers
8. **References** - Academic and practical sources

## Code Examples

All code examples can be run from the command line:

```bash
# Linear Regression: OLS, Ridge, Lasso, Gradient Descent
python -m ml_examples.run --demo linear_regression

# Logistic Regression: Binary classification, metrics
python -m ml_examples.run --demo logistic_regression

# SVM: Margin maximization, kernel trick
python -m ml_examples.run --demo svm

# Decision Trees: Gini/Entropy, tree building
python -m ml_examples.run --demo decision_tree

# Random Forest: Bagging, feature importance
python -m ml_examples.run --demo random_forest

# Gradient Boosting: Sequential learning
python -m ml_examples.run --demo gradient_boosting

# K-Means: Clustering, elbow method
python -m ml_examples.run --demo kmeans

# PCA: From scratch, variance explained
python -m ml_examples.run --demo pca

# Neural Network: MLP, backprop (NumPy)
python -m ml_examples.run --demo neural_network

# Calibration: Platt scaling, ECE
python -m ml_examples.run --demo calibration
```

## Validation

Run validators to ensure documentation quality:

```bash
# Run all validators
python -m validators.main

# Strict mode (warnings as errors)
python -m validators.main --strict
```

Validators check:
- Required sections are present
- Minimum 5 quiz questions per topic
- Bilingual consistency (EN/ZH structure matches)
- Code blocks in example sections

## Building the Site

Generate a static HTML site:

```bash
# Build site
python -m site_generator.build

# Build and serve locally (port 8000)
python -m site_generator.build --serve
```

Features:
- Bilingual toggle (EN/ZH)
- KaTeX math rendering
- Responsive design
- Navigation sidebar
- Collapsible quiz sections

## Skills Extraction

Extract and track ML skills:

```bash
# Extract skills from documentation
python -m skills.extract

# Custom output path
python -m skills.extract --output skills/my_report.yaml
```

Output includes:
- Skills coverage by category
- Skills by difficulty level
- Interview relevance ratings
- Missing skills identification

## Q&A Workflow

For proposing updates:

1. Create proposal in `qa_workflow/proposals/`
2. Follow template in `qa_workflow/README.md`
3. Run validators before submission
4. Await review and merge

## Topics Covered

### ML Fundamentals
- Learning Paradigms (Supervised, Unsupervised, Self-supervised)
- Data Splits & Validation (Train/Val/Test, Cross-validation, Leakage)
- Bias-Variance Tradeoff
- Loss Functions (MSE, MAE, Cross-entropy, Calibration)
- Optimization (GD, SGD, Momentum, Adam)
- Regularization (L1, L2, Early Stopping)
- Generalization & Capacity (VC Dimension)

### Core Models
- Linear Regression (OLS, Ridge, Lasso)
- Logistic Regression
- Support Vector Machines
- k-Nearest Neighbors
- Naive Bayes
- Decision Trees
- Random Forests
- Gradient Boosting
- Clustering (K-Means, GMM, Hierarchical)
- Dimensionality Reduction (PCA, t-SNE, UMAP)

### Deep Learning Basics
- Neural Network Fundamentals (MLP, Backprop)
- CNNs (Convolutions, Pooling)
- RNNs (LSTM, GRU)
- Normalization (BatchNorm, LayerNorm, Dropout)
- Training Pipeline (Initialization, Debugging, Gradient Issues)

### Practical ML Engineering
- Evaluation Metrics (Precision, Recall, F1, ROC-AUC)
- Class Imbalance Strategies
- Feature Engineering
- Model Interpretability (SHAP, Permutation Importance)
- Data-Centric Issues (Label Noise, Dataset Shift)
- MLOps Overview

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes (update both EN and ZH)
4. Run validators
5. Submit pull request

## License

MIT License
