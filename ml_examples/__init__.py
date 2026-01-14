"""
ML Examples Library
===================

A collection of runnable Machine Learning demos for study and interview preparation.

Usage:
    python -m ml_examples.run --demo linear_regression
    python -m ml_examples.run --list
"""

__version__ = "1.0.0"
__author__ = "ML Study Notes Team"

from . import linear_regression
from . import logistic_regression
from . import svm
from . import decision_tree
from . import ensemble
from . import clustering
from . import pca
from . import neural_network

AVAILABLE_DEMOS = {
    "linear_regression": linear_regression,
    "logistic_regression": logistic_regression,
    "svm": svm,
    "decision_tree": decision_tree,
    "random_forest": ensemble,
    "gradient_boosting": ensemble,
    "kmeans": clustering,
    "pca": pca,
    "neural_network": neural_network,
}
