"""
Quick Start: Minimal Superposition Example

Run this for a fast demonstration of the core concept.
"""

import torch
from model import train_toy_model, compute_feature_interference
from visualize import plot_feature_embeddings_2d, plot_interference_matrix


def main():
    """Run a minimal superposition demonstration."""
    print("\n" + "="*60)
    print("  QUICK START: Superposition in 5 Features → 2 Dimensions")
    print("="*60 + "\n")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    n_features = 5      # We have 5 features to represent
    n_hidden = 2        # But only 2 dimensions in the hidden layer
    sparsity = 0.05     # Features are 5% likely to be active

    print(f"Configuration:")
    print(f"  • Input features: {n_features}")
    print(f"  • Hidden dimensions: {n_hidden}")
    print(f"  • Sparsity: {sparsity*100:.1f}% (features rarely co-occur)")
    print(f"  • Compression ratio: {n_features/n_hidden:.1f}x overcomplete")
    print()

    # Train the model
    print("Training model on sparse features...")
    print("-" * 60)
    model = train_toy_model(
        n_features=n_features,
        n_hidden=n_hidden,
        feature_probability=sparsity,
        n_steps=10000,
        lr=1e-3,
        verbose=True
    )

    print("\n" + "="*60)
    print("  RESULTS: Feature Geometry")
    print("="*60 + "\n")

    # Analyze results
    W = model.get_feature_embeddings()
    norms = W.norm(dim=0)

    print("Feature embedding norms:")
    for i in range(n_features):
        print(f"  Feature {i}: {norms[i].item():.3f}")

    interference = compute_feature_interference(model)

    print(f"\nInterference analysis:")
    print(f"  • Diagonal (self): {interference.diag().mean():.3f} (should be 1.0)")
    off_diag = (interference.sum() - interference.diag().sum()) / (n_features * (n_features-1))
    print(f"  • Off-diagonal (cross): {off_diag:.3f} (lower is better)")
    print()

    # Interpretation
    print("="*60)
    print("  INTERPRETATION")
    print("="*60 + "\n")

    print("🎯 The model successfully packed 5 features into 2D space!")
    print()
    print("How? By arranging features at different angles:")
    print("  • If features were orthogonal, we'd need 5 dimensions")
    print("  • With superposition, features can overlap slightly")
    print("  • Because features are sparse, interference is rare")
    print()
    print("Check the visualizations to see the geometry!")
    print()

    # Visualize
    print("Generating visualizations...")
    print("-" * 60)

    plot_feature_embeddings_2d(
        model,
        title=f"Superposition: {n_features} Features in {n_hidden}D Space\n(Sparse: {sparsity*100:.0f}%)"
    )

    plot_interference_matrix(
        interference,
        title="Feature Interference Matrix\n(Low values = less interference)"
    )

    print("\n" + "="*60)
    print("  NEXT STEPS")
    print("="*60 + "\n")
    print("Try experimenting:")
    print("  1. Change sparsity to 0.99 → dense features prevent superposition")
    print("  2. Increase n_hidden to 5 → no need for superposition")
    print("  3. Add more features (n_features=10) → tighter packing")
    print()
    print("Run 'python demo.py' for comprehensive experiments!")
    print()


if __name__ == "__main__":
    main()
