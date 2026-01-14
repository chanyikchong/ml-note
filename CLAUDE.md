You are acting as a senior software engineer and technical writer. Build a complete, working project in the current repository that implements a bilingual (English/中文) Machine Learning study note system that is interview-ready, mathematically rigorous, easy to navigate, and continuously extensible. You must implement the full system (docs + code examples + site/app + Q&A/update workflow + skills extraction + quizzes).

Use the same structure, rules, repository layout, Q&A workflow, validation requirements, and acceptance-criteria philosophy as the RL prompt. Modify ONLY the domain content to Machine Learning.

============================================================
DOMAIN-SPECIFIC REQUIREMENTS (ML)
============================================================
B) ML Notes must include at minimum:
- ML fundamentals:
  - supervised vs unsupervised vs self-supervised (high-level)
  - train/val/test, leakage, cross-validation (including when CV is invalid)
  - bias-variance tradeoff, under/overfitting
  - loss functions (MSE, MAE, cross-entropy), calibration basics
  - optimization: GD/SGD, momentum, Adam, learning rate schedules, regularization (L1/L2), early stopping
  - generalization and capacity; VC-style intuition (no excessive theory, but correct)
- Core models:
  - linear regression (OLS, ridge, lasso) + assumptions
  - logistic regression + decision boundary + regularization
  - SVM (primal/dual intuition, kernel trick high-level)
  - kNN, Naive Bayes
  - decision trees, random forests, gradient boosting (XGBoost-style concepts)
  - clustering: k-means, GMM (high-level), hierarchical clustering (overview)
  - dimensionality reduction: PCA (derivation), t-SNE/UMAP (intuition + pitfalls)
- Deep learning basics:
  - MLP, CNN, RNN basics; backprop intuition
  - normalization (batch/layer), dropout
  - training pipeline: initialization, monitoring, debugging (vanishing/exploding gradients), gradient clipping
- Practical ML engineering:
  - metrics (accuracy, precision/recall/F1, ROC-AUC/PR-AUC), class imbalance strategies
  - feature engineering and preprocessing
  - interpretability: permutation importance, SHAP (overview + caveats)
  - data-centric issues: label noise, dataset shift, outliers
  - MLOps overview: reproducibility, model/versioning, deployment patterns (offline notes only)
  - common interview questions with concise answers
- Each topic page must follow the fixed section order:
  1) Interview summary
  2) Core definitions
  3) Math and derivations
  4) Algorithm sketch
  5) Common pitfalls
  6) Mini example
  7) Quiz (>=5 hidden-answer questions)
  8) References
- Mirrored bilingual structure: docs/en/... and docs/zh/...; identical tree and ordering; quizzes aligned.

C) Example Code Library (Python):
- Provide runnable minimal demos for:
  - linear regression (closed-form + gradient descent)
  - logistic regression (GD + evaluation metrics)
  - SVM (if dependency allowed; else simplified margin demo)
  - decision tree (simple implementation or sklearn-based if allowed)
  - random forest / gradient boosting (sklearn-based acceptable, offline)
  - k-means clustering
  - PCA (from scratch)
  - a small neural network classifier (PyTorch or NumPy; justify)
  - calibration / thresholding demo (optional but recommended)
- Single entrypoint: `python -m ml_examples.run --demo pca`.

Keep all other sections identical to the RL prompt:
- static site + bilingual toggle + math rendering
- Q&A + patch proposal workflow
- validators (including quiz count checks)
- skills extraction (template + at least two ML-relevant skills)
- strict ".claude/" rule
- full README with exact commands

Now implement the project completely. Create files and write all necessary content/code/config. Ensure everything is internally consistent and runnable.
If and only if you hit a blocking ambiguity, ask up to 3 concise clarifying questions as specified in the CLARIFICATION POLICY, then continue.

