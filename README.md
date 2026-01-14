# Machine Learning Study Notes

A comprehensive, bilingual (English/中文) Machine Learning study note system that is interview-ready, mathematically rigorous, easy to navigate, and continuously extensible.

## Features

- **Bilingual Documentation**: Full coverage in English and Chinese with automatic language toggle
- **Interview-Ready**: Each topic includes interview summaries and common questions
- **Mathematically Rigorous**: Complete derivations with MathJax rendering
- **Runnable Code Examples**: Python implementations for all key algorithms
- **Interactive Quizzes**: 5+ quiz questions per topic with click-to-reveal answers
- **Q&A System**: Offline BM25 search with patch proposal workflow

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Serve locally with live reload
mkdocs serve
# Open http://localhost:8000

# Build static site
mkdocs build
```

## Project Structure

```
ML/
├── docs/                        # Documentation (markdown)
│   ├── en/                      # English content
│   │   ├── fundamentals/        # ML basics
│   │   ├── models/              # Core ML models
│   │   ├── deep-learning/       # Deep learning
│   │   ├── practical/           # Practical ML engineering
│   │   └── interview/           # Interview prep
│   └── zh/                      # Chinese content (mirrored)
│
├── ml_examples/                 # Runnable algorithm demos
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── pca.py
│   ├── kmeans.py
│   ├── neural_network.py
│   └── run.py                   # CLI entry point
│
├── qa/                          # Q&A and validation system
│   ├── ask.py                   # Query interface
│   ├── index.py                 # BM25 search index
│   ├── validate.py              # Content validation
│   └── patch.py                 # Patch management
│
├── skills/                      # Reusable skill definitions
├── proposals/                   # Patch proposals
├── site/                        # Generated static site (output)
│
├── mkdocs.yml                   # Site configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Usage

### Run Code Examples

```bash
# List available demos
python -m ml_examples.run --list

# Run specific demo
python -m ml_examples.run --demo linear_regression
python -m ml_examples.run --demo pca
python -m ml_examples.run --demo neural_network
```

### Q&A System

```bash
# Ask a question
python -m qa.ask "What is gradient descent?"
python -m qa.ask "How does PCA work?" --language en

# Ask with patch proposal
python -m qa.ask "I don't understand regularization" --propose-patch

# Rebuild search index
python -m qa.index --rebuild
```

### Content Validation

```bash
# Validate all documentation
python -m qa.validate

# Show only errors
python -m qa.validate --errors-only

# Output as JSON
python -m qa.validate --json
```

### Patch Management

```bash
# List all patches
python -m qa.patch list

# Show patch details
python -m qa.patch show patch_20260114_120000

# Apply patch (dry run)
python -m qa.patch apply patch_20260114_120000

# Apply patch (actual)
python -m qa.patch apply patch_20260114_120000 --force

# Reject patch
python -m qa.patch reject patch_20260114_120000 --reason "Not applicable"
```

## Documentation Structure

Each topic page follows this standardized structure:

1. **Interview Summary** - Key points and what to memorize
2. **Core Definitions** - Formal definitions with notation
3. **Math and Derivations** - Step-by-step mathematical details
4. **Algorithm Sketch** - Pseudocode and complexity
5. **Common Pitfalls** - Mistakes to avoid
6. **Mini Example** - Worked example with numbers
7. **Quiz** - 5+ self-assessment questions (click to reveal)
8. **References** - Books, papers, resources

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
- Ensemble Methods (Random Forest, Gradient Boosting, XGBoost)
- Clustering (K-Means, GMM, DBSCAN)
- Dimensionality Reduction (PCA, t-SNE, UMAP)

### Deep Learning Basics
- Neural Network Fundamentals (MLP, Backprop)
- CNNs (Convolutions, Pooling)
- RNNs (LSTM, GRU)
- Normalization (BatchNorm, LayerNorm, Dropout)
- Training Pipeline (Initialization, Debugging, Gradient Issues)

### Practical ML Engineering
- Evaluation Metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Class Imbalance Strategies
- Feature Engineering
- Model Interpretability (SHAP, Permutation Importance)
- Data-Centric Issues (Label Noise, Dataset Shift)
- MLOps Overview

### Interview Preparation
- Common Questions Q&A
- Quick Reference Card (formulas, key numbers)

## Deployment

### GitHub Pages

1. Push to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. Enable GitHub Pages:
   - Go to repository **Settings** → **Pages**
   - Source: **GitHub Actions**

3. The site will be automatically built and deployed on push to main.

Your site will be available at: `https://YOUR_USERNAME.github.io/ml-study-notes/`

### Manual Deployment

```bash
# Build site
mkdocs build

# Or deploy directly to GitHub Pages
mkdocs gh-deploy
```

## Contributing

1. Ensure content follows the 8-section structure
2. Run validation before committing:
   ```bash
   python -m qa.validate
   ```
3. Maintain bilingual consistency (EN and ZH must match)
4. Include at least 5 quiz questions per topic

## License

MIT License
