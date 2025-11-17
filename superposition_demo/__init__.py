"""
Toy Models of Superposition

A demonstration of how neural networks can represent more features than dimensions
through superposition when features are sparse.

Based on Anthropic's research:
https://transformer-circuits.pub/2022/toy_model/index.html
"""

from .model import ToyModel, train_toy_model, generate_sparse_features, compute_feature_interference
from .visualize import (
    plot_feature_embeddings_2d,
    plot_interference_matrix,
    plot_feature_norms,
    compare_models
)

__version__ = "0.1.0"
__all__ = [
    "ToyModel",
    "train_toy_model",
    "generate_sparse_features",
    "compute_feature_interference",
    "plot_feature_embeddings_2d",
    "plot_interference_matrix",
    "plot_feature_norms",
    "compare_models",
]
