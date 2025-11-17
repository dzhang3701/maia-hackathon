"""
Visualization utilities for superposition demonstrations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from model import ToyModel


def plot_feature_embeddings_2d(
    model: ToyModel,
    title: str = "Feature Embeddings",
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None
):
    """
    Plot feature embeddings in 2D space.
    Only works when n_hidden = 2.

    Args:
        model: Trained toy model
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if model.n_hidden != 2:
        raise ValueError("This visualization only works for 2D hidden space (n_hidden=2)")

    W = model.get_feature_embeddings()  # (2, n_features)

    plt.figure(figsize=figsize)

    # Plot each feature as a vector from origin
    for i in range(model.n_features):
        vec = W[:, i].cpu().numpy()
        plt.arrow(
            0, 0, vec[0], vec[1],
            head_width=0.1,
            head_length=0.1,
            fc=f'C{i % 10}',
            ec=f'C{i % 10}',
            alpha=0.7,
            width=0.02
        )
        # Label the feature
        plt.text(
            vec[0] * 1.1, vec[1] * 1.1,
            f'F{i}',
            fontsize=10,
            ha='center',
            va='center'
        )

    # Draw unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Hidden Dimension 1')
    plt.ylabel('Hidden Dimension 2')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_interference_matrix(
    interference: torch.Tensor,
    title: str = "Feature Interference Matrix",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot the interference matrix between features.

    Args:
        interference: Interference matrix (n_features, n_features)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    interference_np = interference.cpu().numpy()

    # Plot heatmap
    im = plt.imshow(
        interference_np,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        aspect='auto'
    )

    plt.colorbar(im, label='Dot Product (Interference)')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.title(title)

    # Add grid
    n_features = interference.shape[0]
    plt.xticks(range(0, n_features, max(1, n_features // 10)))
    plt.yticks(range(0, n_features, max(1, n_features // 10)))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_norms(
    model: ToyModel,
    feature_importance: Optional[torch.Tensor] = None,
    title: str = "Feature Embedding Norms",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None
):
    """
    Plot the norms of feature embeddings.

    Features with higher importance should have larger norms.

    Args:
        model: Trained toy model
        feature_importance: Optional ground truth importance
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    W = model.get_feature_embeddings()  # (n_hidden, n_features)
    norms = W.norm(dim=0).cpu().numpy()

    plt.figure(figsize=figsize)

    # Plot norms
    plt.subplot(1, 2, 1)
    plt.bar(range(len(norms)), norms, alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Embedding Norm')
    plt.title('Feature Embedding Norms')
    plt.grid(True, alpha=0.3)

    # Plot importance vs norm if available
    if feature_importance is not None:
        plt.subplot(1, 2, 2)
        importance_np = feature_importance.cpu().numpy()
        plt.scatter(importance_np, norms, alpha=0.6)
        plt.xlabel('Feature Importance')
        plt.ylabel('Embedding Norm')
        plt.title('Importance vs Embedding Norm')
        plt.grid(True, alpha=0.3)

        # Add correlation
        corr = np.corrcoef(importance_np, norms)[0, 1]
        plt.text(
            0.05, 0.95,
            f'Correlation: {corr:.3f}',
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top'
        )

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_phase_diagram(
    sparsity_levels: list,
    compression_ratios: list,
    superposition_detected: np.ndarray,
    title: str = "Superposition Phase Diagram",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot phase diagram showing when superposition occurs.

    Args:
        sparsity_levels: List of sparsity levels (feature probability)
        compression_ratios: List of compression ratios (n_features / n_hidden)
        superposition_detected: 2D array indicating superposition
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    plt.imshow(
        superposition_detected,
        aspect='auto',
        cmap='RdYlGn_r',
        extent=[
            min(compression_ratios),
            max(compression_ratios),
            min(sparsity_levels),
            max(sparsity_levels)
        ],
        origin='lower'
    )

    plt.colorbar(label='Superposition Score')
    plt.xlabel('Compression Ratio (n_features / n_hidden)')
    plt.ylabel('Sparsity Level (feature probability)')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_models(
    model_sparse: ToyModel,
    model_dense: ToyModel,
    save_dir: Optional[str] = None
):
    """
    Compare sparse vs dense trained models.

    Args:
        model_sparse: Model trained on sparse features
        model_dense: Model trained on dense features
        save_dir: Optional directory to save figures
    """
    if model_sparse.n_hidden != 2 or model_dense.n_hidden != 2:
        print("Comparison visualization requires 2D hidden space")
        return

    plt.figure(figsize=(16, 6))

    # Sparse model
    plt.subplot(1, 2, 1)
    W_sparse = model_sparse.get_feature_embeddings().cpu().numpy()
    for i in range(model_sparse.n_features):
        vec = W_sparse[:, i]
        plt.arrow(
            0, 0, vec[0], vec[1],
            head_width=0.1,
            head_length=0.1,
            fc=f'C{i % 10}',
            ec=f'C{i % 10}',
            alpha=0.7,
            width=0.02
        )
        plt.text(vec[0] * 1.1, vec[1] * 1.1, f'F{i}', fontsize=10)

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title('Sparse Features (Superposition)')
    plt.xlabel('Hidden Dim 1')
    plt.ylabel('Hidden Dim 2')

    # Dense model
    plt.subplot(1, 2, 2)
    W_dense = model_dense.get_feature_embeddings().cpu().numpy()
    for i in range(model_dense.n_features):
        vec = W_dense[:, i]
        plt.arrow(
            0, 0, vec[0], vec[1],
            head_width=0.1,
            head_length=0.1,
            fc=f'C{i % 10}',
            ec=f'C{i % 10}',
            alpha=0.7,
            width=0.02
        )
        plt.text(vec[0] * 1.1, vec[1] * 1.1, f'F{i}', fontsize=10)

    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title('Dense Features (No Superposition)')
    plt.xlabel('Hidden Dim 1')
    plt.ylabel('Hidden Dim 2')

    plt.tight_layout()

    if save_dir:
        plt.savefig(f'{save_dir}/comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
