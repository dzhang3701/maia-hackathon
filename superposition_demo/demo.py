"""
Superposition Demonstration Script

This script runs several experiments to demonstrate the superposition phenomenon:
1. Basic superposition: More features than dimensions
2. Sparse vs dense comparison
3. Feature importance effects
4. Phase transitions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import (
    ToyModel,
    train_toy_model,
    compute_feature_interference
)
from visualize import (
    plot_feature_embeddings_2d,
    plot_interference_matrix,
    plot_feature_norms,
    compare_models
)


def experiment_1_basic_superposition():
    """
    Experiment 1: Basic Superposition Demonstration

    Train a model with more features than hidden dimensions on sparse data.
    This should show features arranging themselves in superposition.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Basic Superposition")
    print("=" * 70)
    print("\nSetup:")
    print("  - 5 features, 2 hidden dimensions (overcomplete)")
    print("  - Sparse features (5% probability)")
    print("  - Training for 10,000 steps")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Train model
    n_features = 5
    n_hidden = 2
    feature_probability = 0.05

    print("Training model...")
    model = train_toy_model(
        n_features=n_features,
        n_hidden=n_hidden,
        feature_probability=feature_probability,
        n_steps=10000,
        lr=1e-3,
        verbose=True
    )

    print("\nVisualization:")
    print("  - Feature embeddings in 2D space")
    print("  - Notice how 5 features are packed into 2 dimensions")
    print("  - Features form a quasi-orthogonal arrangement")
    print()

    # Visualize
    plot_feature_embeddings_2d(
        model,
        title=f"Superposition: {n_features} features in {n_hidden}D space"
    )

    # Compute and show interference
    interference = compute_feature_interference(model)
    print(f"\nFeature Interference Matrix:")
    print(f"  - Diagonal: {interference.diag().mean():.3f} (self-interference, should be 1.0)")
    print(f"  - Off-diagonal: {(interference.sum() - interference.diag().sum()) / (n_features * (n_features - 1)):.3f}")
    print(f"    (cross-interference, lower is better)")
    print()

    plot_interference_matrix(
        interference,
        title="Feature Interference (superposition)"
    )

    return model


def experiment_2_sparse_vs_dense():
    """
    Experiment 2: Sparse vs Dense Features

    Compare models trained on sparse vs dense features.
    Sparse features enable superposition, dense features don't.
    """
    print("=" * 70)
    print("EXPERIMENT 2: Sparse vs Dense Comparison")
    print("=" * 70)
    print("\nSetup:")
    print("  - 5 features, 2 hidden dimensions")
    print("  - Sparse: 5% feature probability")
    print("  - Dense: 99% feature probability")
    print()

    torch.manual_seed(42)

    n_features = 5
    n_hidden = 2

    # Train sparse model
    print("Training SPARSE model...")
    model_sparse = train_toy_model(
        n_features=n_features,
        n_hidden=n_hidden,
        feature_probability=0.05,
        n_steps=10000,
        lr=1e-3,
        verbose=False
    )
    print("Done!")

    # Train dense model
    print("Training DENSE model...")
    model_dense = train_toy_model(
        n_features=n_features,
        n_hidden=n_hidden,
        feature_probability=0.99,
        n_steps=10000,
        lr=1e-3,
        verbose=False
    )
    print("Done!")

    print("\nKey Observations:")
    print("  - Sparse model: Features spread out in superposition")
    print("  - Dense model: Only 2 features dominate (no room for superposition)")
    print()

    # Compare
    compare_models(model_sparse, model_dense)

    return model_sparse, model_dense


def experiment_3_feature_importance():
    """
    Experiment 3: Feature Importance

    Show how feature importance affects the geometry.
    Important features get larger embeddings.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Feature Importance")
    print("=" * 70)
    print("\nSetup:")
    print("  - 5 features, 2 hidden dimensions")
    print("  - Varying importance: [1.0, 0.9, 0.7, 0.4, 0.2]")
    print("  - Sparse features (5% probability)")
    print()

    torch.manual_seed(42)

    n_features = 5
    n_hidden = 2
    feature_importance = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.2])

    print("Training model with importance weighting...")
    model = train_toy_model(
        n_features=n_features,
        n_hidden=n_hidden,
        feature_probability=0.05,
        feature_importance=feature_importance,
        n_steps=10000,
        lr=1e-3,
        verbose=True
    )

    print("\nVisualization:")
    print("  - More important features should have larger embeddings")
    print("  - This allows the model to represent them more accurately")
    print()

    # Visualize
    plot_feature_embeddings_2d(
        model,
        title="Superposition with Feature Importance"
    )

    plot_feature_norms(
        model,
        feature_importance=feature_importance,
        title="Feature Importance vs Embedding Norm"
    )

    return model


def experiment_4_dimensionality_sweep():
    """
    Experiment 4: Dimensionality Sweep

    Show what happens as we vary the hidden dimension.
    """
    print("=" * 70)
    print("EXPERIMENT 4: Dimensionality Sweep")
    print("=" * 70)
    print("\nSetup:")
    print("  - 8 features, varying hidden dimensions")
    print("  - Sparse features (10% probability)")
    print()

    torch.manual_seed(42)

    n_features = 8
    hidden_dims = [2, 3, 4, 6, 8]
    feature_probability = 0.10

    results = []

    for n_hidden in hidden_dims:
        print(f"\nTraining with n_hidden={n_hidden}...")
        model = train_toy_model(
            n_features=n_features,
            n_hidden=n_hidden,
            feature_probability=feature_probability,
            n_steps=5000,
            lr=1e-3,
            verbose=False
        )

        # Compute interference
        interference = compute_feature_interference(model)
        off_diag_interference = (
            (interference.sum() - interference.diag().sum()) /
            (n_features * (n_features - 1))
        ).item()

        results.append({
            'n_hidden': n_hidden,
            'compression': n_features / n_hidden,
            'interference': off_diag_interference
        })

        print(f"  Compression ratio: {n_features / n_hidden:.2f}x")
        print(f"  Interference: {off_diag_interference:.3f}")

    # Plot results
    print("\nVisualization:")
    print("  - As hidden dim increases, interference decreases")
    print("  - At n_hidden = n_features, no superposition needed")
    print()

    plt.figure(figsize=(10, 6))

    compressions = [r['compression'] for r in results]
    interferences = [r['interference'] for r in results]

    plt.plot(compressions, interferences, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Compression Ratio (n_features / n_hidden)')
    plt.ylabel('Average Off-Diagonal Interference')
    plt.title('Superposition vs Compression')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Annotate points
    for r in results:
        plt.annotate(
            f"n_h={r['n_hidden']}",
            xy=(r['compression'], r['interference']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    return results


def run_all_experiments():
    """Run all experiments in sequence."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TOY MODELS OF SUPERPOSITION" + " " * 25 + "║")
    print("║" + " " * 20 + "Demonstration Suite" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    # Run experiments
    exp1_model = experiment_1_basic_superposition()
    input("\nPress Enter to continue to Experiment 2...")

    exp2_sparse, exp2_dense = experiment_2_sparse_vs_dense()
    input("\nPress Enter to continue to Experiment 3...")

    exp3_model = experiment_3_feature_importance()
    input("\nPress Enter to continue to Experiment 4...")

    exp4_results = experiment_4_dimensionality_sweep()

    print("\n")
    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Neural networks can represent more features than dimensions")
    print("  2. This works when features are SPARSE (rarely co-occur)")
    print("  3. Features arrange in quasi-orthogonal configurations")
    print("  4. Important features get larger embeddings (less interference)")
    print("  5. More dimensions = less interference = less superposition")
    print("\nImplications for AI Safety:")
    print("  - Models may have many more 'concepts' than neurons")
    print("  - Interpretability is harder: features are distributed")
    print("  - Understanding what models represent requires new tools")
    print("=" * 70)
    print()


if __name__ == "__main__":
    # For non-interactive use, just run all experiments
    run_all_experiments()
