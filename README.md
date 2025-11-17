# Toy Models of Superposition

A simple implementation demonstrating **superposition** in neural networks, based on Anthropic's research paper ["Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html).

## What is Superposition?

**Superposition** is a phenomenon where neural networks represent **more features than they have dimensions**. This happens when:

1. **Features are sparse** - they rarely occur together in the same input
2. **Limited capacity** - the network has fewer neurons than features to represent
3. **Optimization pressure** - the network is incentivized to represent all features

### Why Does This Matter for AI Safety?

Understanding superposition is crucial for AI interpretability and safety:

- **Hidden Complexity**: Models may represent far more "concepts" than their neuron count suggests
- **Distributed Representations**: Features aren't cleanly separated into individual neurons
- **Interpretability Challenges**: Traditional neuroscience-inspired approaches may miss distributed features
- **Mechanistic Understanding**: We need new tools to understand what models actually learn

## The Toy Model

Our toy model is deliberately simple:

```
Input (n features) → W → ReLU → W^T → Output (n features reconstruction)
```

Where:
- Input has `n_features` dimensions
- Hidden layer has `n_hidden` dimensions (with `n_hidden < n_features`)
- The model must compress and reconstruct the input

**Key Insight**: When features are sparse, the model can "pack" multiple features into the same hidden dimensions by arranging them at different angles (superposition).

## Installation

```bash
cd superposition_demo
pip install -r requirements.txt
```

## Running the Demo

```bash
python demo.py
```

This runs 4 experiments:

### Experiment 1: Basic Superposition
- **Setup**: 5 features, 2 hidden dimensions, 5% sparsity
- **Observation**: Features arrange in a quasi-orthogonal pattern, packing 5 features into 2D space

### Experiment 2: Sparse vs Dense Comparison
- **Setup**: Compare 5% sparsity vs 99% sparsity
- **Observation**: Sparse features enable superposition; dense features force the model to pick only 2 dominant features

### Experiment 3: Feature Importance
- **Setup**: 5 features with varying importance weights
- **Observation**: More important features get larger embeddings (better reconstruction, less interference)

### Experiment 4: Dimensionality Sweep
- **Setup**: 8 features with varying hidden dimensions
- **Observation**: As hidden dimension approaches feature count, interference decreases and superposition becomes less necessary

## Example Results

When you run the demo, you'll see visualizations like:

**2D Feature Embeddings** (Experiment 1):
```
    F1↗
       ╲
F0←─────┼─────→F2
       ╱
    F3↙  ↘F4
```
5 features arranged in 2D space, forming approximate orthogonal directions.

**Interference Matrix**:
Shows how much features interfere with each other (off-diagonal values). Lower interference = better superposition.

**Phase Transitions**:
As you vary sparsity and compression, the model transitions between:
- **No superposition**: Features are orthogonal (enough space)
- **Superposition**: Features are packed densely
- **Feature suppression**: Some features are dropped (too dense, not enough space)

## Code Structure

```
superposition_demo/
├── model.py          # Core toy model implementation
├── visualize.py      # Visualization utilities
├── demo.py           # Main demonstration script
└── requirements.txt  # Python dependencies
```

## Key Functions

### `model.py`
- `ToyModel`: The basic linear autoencoder model
- `generate_sparse_features()`: Creates sparse random features
- `train_toy_model()`: Training loop
- `compute_feature_interference()`: Measures feature overlap

### `visualize.py`
- `plot_feature_embeddings_2d()`: Visualize features in 2D space
- `plot_interference_matrix()`: Show feature interference heatmap
- `plot_feature_norms()`: Display embedding magnitudes
- `compare_models()`: Side-by-side sparse vs dense comparison

## Extending This Demo

Ideas for further exploration:

1. **3D Visualization**: Extend to `n_hidden=3` with 3D plots
2. **Phase Diagrams**: Map the full sparsity × compression space
3. **Nonlinear Models**: Add more layers or nonlinearities
4. **Real Features**: Use real sparse data (e.g., word frequencies)
5. **Privileged Basis**: Explore whether the standard basis is special
6. **Neuron Interpretability**: Study individual hidden neurons' responses

## Mathematical Details

### The Model

Given input $x \in \mathbb{R}^{n}$ where most entries are zero:

$$h = \text{ReLU}(Wx)$$
$$\hat{x} = W^T h$$

Loss: $L = ||x - \hat{x}||^2$

### Why Superposition Works

When features are sparse:
- Features rarely co-occur in the same input
- The model can use overlapping directions for different features
- The interference (when features do co-occur) is rare enough to be acceptable

### Feature Geometry

Each feature $i$ has an embedding $w_i$ (column of $W^T$).

**Interference** between features $i$ and $j$:
$$\text{interference}_{ij} = \frac{w_i \cdot w_j}{||w_i|| \cdot ||w_j||}$$

Ideally $\approx 0$ for $i \neq j$, but with superposition $> 0$.

## References

- **Original Paper**: [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic, 2022)
- **Related**: [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) (OpenAI Clarity team)
- **Related**: [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html) (Anthropic, 2022)

## AI Safety Context

This work is part of **mechanistic interpretability** research, which aims to:

1. Understand what neural networks actually learn
2. Build tools to inspect model internals
3. Detect dangerous capabilities or deceptive behavior
4. Create more aligned and controllable AI systems

Superposition is a fundamental challenge: if we can't cleanly identify what features a model represents, we can't fully understand or control its behavior.

## License

MIT License - Feel free to use for research and education.

## Acknowledgments

This implementation is inspired by Anthropic's excellent research on mechanistic interpretability. All credit for the core ideas goes to the Anthropic team.
