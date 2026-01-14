#!/usr/bin/env python3
"""
ML Examples Runner
==================

Single entrypoint for running all ML demos.

Usage:
    python -m ml_examples.run --demo linear_regression
    python -m ml_examples.run --demo pca
    python -m ml_examples.run --list
"""

import argparse
import sys

DEMOS = {
    "linear_regression": "Linear Regression (OLS, Ridge, Lasso + GD)",
    "logistic_regression": "Logistic Regression with Gradient Descent",
    "svm": "Support Vector Machine (simplified margin demo)",
    "decision_tree": "Decision Tree Classifier",
    "random_forest": "Random Forest Classifier",
    "gradient_boosting": "Gradient Boosting (sklearn-based)",
    "kmeans": "K-Means Clustering",
    "pca": "Principal Component Analysis (from scratch)",
    "neural_network": "Simple Neural Network Classifier (NumPy)",
    "calibration": "Model Calibration and Thresholding",
}


def list_demos():
    """List all available demos."""
    print("\nAvailable ML Demos:")
    print("=" * 50)
    for name, description in DEMOS.items():
        print(f"  {name:20s} - {description}")
    print("\nUsage: python -m ml_examples.run --demo <name>")
    print("=" * 50)


def run_demo(demo_name: str):
    """Run a specific demo."""
    if demo_name not in DEMOS:
        print(f"Error: Unknown demo '{demo_name}'")
        print("Use --list to see available demos")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Running: {DEMOS[demo_name]}")
    print(f"{'='*60}\n")

    if demo_name == "linear_regression":
        from . import linear_regression
        linear_regression.run_demo()
    elif demo_name == "logistic_regression":
        from . import logistic_regression
        logistic_regression.run_demo()
    elif demo_name == "svm":
        from . import svm
        svm.run_demo()
    elif demo_name == "decision_tree":
        from . import decision_tree
        decision_tree.run_demo()
    elif demo_name == "random_forest":
        from . import ensemble
        ensemble.run_random_forest_demo()
    elif demo_name == "gradient_boosting":
        from . import ensemble
        ensemble.run_gradient_boosting_demo()
    elif demo_name == "kmeans":
        from . import clustering
        clustering.run_demo()
    elif demo_name == "pca":
        from . import pca
        pca.run_demo()
    elif demo_name == "neural_network":
        from . import neural_network
        neural_network.run_demo()
    elif demo_name == "calibration":
        from . import calibration
        calibration.run_demo()

    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ML Examples Runner - Interview-ready ML demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ml_examples.run --demo linear_regression
    python -m ml_examples.run --demo pca
    python -m ml_examples.run --list
        """,
    )
    parser.add_argument(
        "--demo",
        type=str,
        help="Name of the demo to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available demos",
    )

    args = parser.parse_args()

    if args.list:
        list_demos()
    elif args.demo:
        run_demo(args.demo)
    else:
        parser.print_help()
        print("\nUse --list to see available demos")


if __name__ == "__main__":
    main()
