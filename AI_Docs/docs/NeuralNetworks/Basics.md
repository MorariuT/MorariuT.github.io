# Neural Networks - Basics

## What is a Neural Network?

A **Neural Network** (NN) is a computational model inspired by biological neural networks in animal brains. It consists of interconnected nodes (neurons) organized in layers that work together to process information.

## Key Concepts

### Neurons (Perceptrons)

A neuron is the basic building block of neural networks. Each neuron performs a simple mathematical operation:

$$
a = \sigma(w^T x + b)
$$

Where:
- $x$ = input vector
- $w$ = weight vector
- $b$ = bias term
- $\sigma$ = activation function
- $a$ = activation (output)

### Layers

Neural networks are organized in layers:

1. **Input Layer**: Receives the raw data
2. **Hidden Layers**: Perform computations and feature extraction
3. **Output Layer**: Produces the final prediction

```
Input Layer    Hidden Layers          Output Layer
    x₁ ────┐
    x₂ ────┤──→ [h₁] ──→ [h₃]
    x₃ ────┤    [h₂] ──→ [h₄] ──→ ŷ
    x₄ ────┘
```

### Activation Functions

Activation functions introduce non-linearity to the network, enabling it to learn complex patterns.

**Common activation functions:**

#### ReLU (Rectified Linear Unit)
$$
f(x) = \max(0, x)
$$

#### Sigmoid
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

#### Tanh
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

#### Softmax (for multi-class classification)
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

## Architecture Types

### Feed-Forward Networks

The simplest type where information flows in one direction from input to output.

### Recurrent Neural Networks (RNNs)

Include feedback connections, allowing them to process sequences.

### Convolutional Neural Networks (CNNs)

Specialized for image processing using convolutional layers.

## Forward Pass

The forward pass is the process of sending input data through the network to get predictions.

For each layer $l$:
$$
a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})
$$

Where:
- $a^{(l)}$ = activations at layer $l$
- $W^{(l)}$ = weights at layer $l$
- $b^{(l)}$ = biases at layer $l$

## Backpropagation

Backpropagation is the algorithm used to train neural networks by computing gradients and updating weights.

See [Back Propagation](BackPropagation.md) for detailed explanation.

## Training Process

1. **Initialize weights** randomly
2. **Forward pass**: Compute predictions
3. **Compute loss**: Measure prediction error
4. **Backward pass**: Compute gradients
5. **Update weights**: Adjust using gradients
6. **Repeat** until convergence

## Loss Functions

Common loss functions for different tasks:

- **Regression**: Mean Squared Error (MSE)
- **Binary Classification**: Binary Cross-Entropy
- **Multi-class Classification**: Categorical Cross-Entropy

## Key Advantages

✓ Can learn complex non-linear relationships
✓ Universal function approximators
✓ Work well with large amounts of data
✓ Flexible architecture design

## Key Challenges

✗ Computational cost for training
✗ Risk of overfitting
✗ Difficult to interpret ("black box")
✗ Sensitive to hyperparameter choices

## Further Reading

- [Forward Propagation](ForwardPropagation.md)
- [Back Propagation](BackPropagation.md)
- [Simple Neural Network Implementation](SimpleNeuralNetwork.md)
