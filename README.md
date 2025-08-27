# RBF Network for Function Approximation

A Python implementation of a Radial Basis Function (RBF) neural network using nonlinear optimization techniques to approximate the sine function.

## Overview

This project implements a 2-layer RBF network that learns to approximate the target function `f(p) = sin(p)` for `p ∈ [0, π]`. The network uses Gaussian activation functions in the hidden layer and linear activation in the output layer, trained using backpropagation with nonlinear optimization.

## Network Architecture

- **Input Layer**: 1 neuron (input value p)
- **Hidden Layer**: 2 neurons with Gaussian (RBF) activation functions
- **Output Layer**: 1 neuron with linear activation function
- **Activation Functions**:
  - Hidden Layer: Gaussian function `φ(n) = exp(-n²)`
  - Output Layer: Linear function `φ(n) = n`

## Features

- **Custom RBF Implementation**: Built from scratch using NumPy
- **Nonlinear Optimization**: Uses gradient descent with backpropagation
- **Multiple Learning Rates**: Support for testing different learning rates (α = 0.01, 0.02, 0.04, 0.06, 0.08, 0.1)
- **Convergence Monitoring**: Tracks Sum of Squared Errors (SSE) with configurable threshold
- **Visualization**: 
  - Network response vs target function plot
  - SSE convergence plot (log-scale for both axes)
- **Parameter Tracking**: Displays initial and final weights and biases

## Requirements

```
numpy
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rbf-network-function-approximation.git
cd rbf-network-function-approximation
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

Run the main script:

```bash
python Question5.py
```

### Training Parameters

The default configuration uses:
- **Number of observations**: 100 data points
- **Maximum iterations**: 2000
- **Learning rate (α)**: 0.01
- **SSE threshold**: 0.001
- **Initial conditions**: `N(μ = 0, σ = 1)` for weights and biases

### Testing Different Learning Rates

Uncomment the respective lines in the code to test different learning rates:

```python
# train_rbf(rbf, inputs, targets, learning_rate=0.02, max_iterations=2000, sse_cutoff=0.001)
# train_rbf(rbf, inputs, targets, learning_rate=0.04, max_iterations=2000, sse_cutoff=0.001)
# train_rbf(rbf, inputs, targets, learning_rate=0.06, max_iterations=2000, sse_cutoff=0.001)
# train_rbf(rbf, inputs, targets, learning_rate=0.08, max_iterations=2000, sse_cutoff=0.001)
# train_rbf(rbf, inputs, targets, learning_rate=0.1, max_iterations=2000, sse_cutoff=0.001)
```

## Results

The network successfully learns to approximate the sine function with the following outputs:

1. **Training Progress**: Real-time SSE tracking and convergence monitoring
2. **Function Approximation Plot**: Comparison between target sine function and network output
3. **Error Analysis**: Log-scale plot showing SSE reduction over iterations
4. **Parameter Evolution**: Display of initial and final network parameters

### Effect of Learning Rate (α)

The learning rate significantly affects training behavior:
- **Lower α (0.01-0.04)**: Slower but more stable convergence
- **Higher α (0.06-0.1)**: Faster convergence but potential instability

Recommended starting value: **α = 0.01** for stable training.

## Implementation Details

### Forward Propagation
- Hidden layer uses Gaussian RBF with absolute distance metric
- Output layer applies linear transformation

### Backward Propagation
- Calculates gradients using chain rule
- Updates weights and biases using gradient descent
- Special handling for RBF derivative computation

### Key Functions
- `gaussian(n)`: RBF activation function
- `gaussian_derivative(n)`: Derivative of RBF function
- `forward()`: Forward pass computation
- `backward()`: Backpropagation algorithm
- `train_rbf()`: Main training loop with visualization

## File Structure

```
.
├── Question5.py          # Main implementation
├── README.md            # This file
└── requirements.txt     # Dependencies (if added)
```

## Mathematical Foundation

The network minimizes the objective function:
```
E = Σ(target - output)²
```

Using the update rules derived from gradient descent on the RBF network parameters.

## Contributing

Feel free to fork this project and submit pull requests for improvements such as:
- Additional activation functions
- Different optimization algorithms
- Enhanced visualization features
- Performance optimizations

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

This implementation is based on neural network theory from Hagan, Demuth, and Beale's "Neural Network Design" and follows standard RBF network architectures for function approximation tasks.
