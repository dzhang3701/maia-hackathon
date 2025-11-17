"""
Toy Model of Superposition

Based on Anthropic's paper: "Toy Models of Superposition"
https://transformer-circuits.pub/2022/toy_model/index.html

This implements a simple model that demonstrates how neural networks can
represent more features than they have dimensions through superposition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional


class ToyModel(nn.Module):
    """
    A simple linear model: x -> Wx -> ReLU -> W^T x'

    Where:
    - x is the input features (n_features dim)
    - W is the embedding matrix (n_hidden x n_features)
    - x' is the hidden representation (n_hidden dim)
    - Output is reconstruction of x (n_features dim)

    The key insight: if n_features > n_hidden, the model must use
    superposition to represent all features.
    """

    def __init__(self, n_features: int, n_hidden: int):
        """
        Args:
            n_features: Number of input features
            n_hidden: Hidden dimension (bottleneck)
        """
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden

        # Single weight matrix (W)
        # We'll use W and W^T to embed and unembed
        self.W = nn.Parameter(torch.randn(n_hidden, n_features))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input features (batch_size, n_features)

        Returns:
            hidden: Hidden representation (batch_size, n_hidden)
            output: Reconstructed features (batch_size, n_features)
        """
        # Embed: hidden = ReLU(W @ x)
        hidden = torch.relu(self.W @ x.T)  # (n_hidden, batch_size)
        hidden = hidden.T  # (batch_size, n_hidden)

        # Unembed: output = W^T @ hidden
        output = hidden @ self.W  # (batch_size, n_features)

        return hidden, output

    def get_feature_embeddings(self) -> torch.Tensor:
        """
        Get the embedding vectors for each feature.
        Each column of W^T is the embedding of a feature.

        Returns:
            Feature embeddings (n_hidden, n_features)
        """
        return self.W.data


def generate_sparse_features(
    batch_size: int,
    n_features: int,
    feature_probability: float,
    feature_importance: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Generate sparse feature vectors where each feature is present
    with some probability.

    Args:
        batch_size: Number of samples
        n_features: Number of features
        feature_probability: Probability each feature is active
        feature_importance: Optional importance weights for each feature

    Returns:
        Sparse feature vectors (batch_size, n_features)
    """
    # Sample which features are active
    active = torch.rand(batch_size, n_features) < feature_probability

    # Generate feature values (uniform [0, 1])
    values = torch.rand(batch_size, n_features)

    # Apply sparsity mask
    features = active.float() * values

    # Apply importance weighting if provided
    if feature_importance is not None:
        features = features * feature_importance.unsqueeze(0)

    return features


def train_toy_model(
    n_features: int,
    n_hidden: int,
    feature_probability: float = 0.05,
    feature_importance: Optional[torch.Tensor] = None,
    n_steps: int = 10000,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = True
) -> ToyModel:
    """
    Train the toy model on sparse features.

    Args:
        n_features: Number of input features
        n_hidden: Hidden dimension
        feature_probability: Sparsity level (probability feature is active)
        feature_importance: Optional importance weights
        n_steps: Number of training steps
        batch_size: Batch size
        lr: Learning rate
        verbose: Print training progress

    Returns:
        Trained model
    """
    model = ToyModel(n_features, n_hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        # Generate batch of sparse features
        x = generate_sparse_features(
            batch_size, n_features, feature_probability, feature_importance
        )

        # Forward pass
        hidden, output = model(x)

        # Loss: mean squared error weighted by importance
        if feature_importance is not None:
            loss = ((output - x) ** 2 * feature_importance.unsqueeze(0)).mean()
        else:
            loss = ((output - x) ** 2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if verbose and (step % 1000 == 0 or step == n_steps - 1):
            print(f"Step {step:5d} | Loss: {loss.item():.6f}")

    return model


def compute_feature_interference(model: ToyModel) -> torch.Tensor:
    """
    Compute interference between features.

    When features are in superposition, they interfere with each other.
    This computes the dot product between feature embeddings.

    Args:
        model: Trained toy model

    Returns:
        Interference matrix (n_features, n_features)
    """
    W = model.get_feature_embeddings()  # (n_hidden, n_features)

    # Normalize embeddings
    W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-8)

    # Compute dot products (interference)
    interference = W_norm.T @ W_norm  # (n_features, n_features)

    return interference
