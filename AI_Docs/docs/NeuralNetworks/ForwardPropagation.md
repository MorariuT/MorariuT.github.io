# Forward Propagation

## What is Forward Propagation?

Forward propagation (also called forward pass) is the process of sending input data through a neural network to compute the output predictions. It's the first step in training a neural network.

## The Forward Pass Process

For a neural network with $L$ layers, the forward pass computes activations layer by layer:

### Layer Computation

For layer $l$ (where $l = 1, 2, \ldots, L$):

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = \sigma(z^{(l)})
$$

Where:
- $z^{(l)}$ = pre-activation (linear combination)
- $a^{(l)}$ = activation output of layer $l$
- $W^{(l)}$ = weight matrix of layer $l$
- $b^{(l)}$ = bias vector of layer $l$
- $\sigma$ = activation function
- $a^{(0)}$ = $x$ (input data)

## Example: 2-Layer Network

Consider a simple neural network with:
- Input layer: 2 features
- Hidden layer: 3 neurons
- Output layer: 1 neuron

### Input
$$
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

### Hidden Layer
$$
z^{(1)} = W^{(1)} x + b^{(1)} = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}
$$

$$
a^{(1)} = \text{ReLU}(z^{(1)}) = \begin{bmatrix} \max(0, z_1^{(1)}) \\ \max(0, z_2^{(1)}) \\ \max(0, z_3^{(1)}) \end{bmatrix}
$$

### Output Layer
$$
z^{(2)} = W^{(2)} a^{(1)} + b^{(2)} = \begin{bmatrix} w_1 & w_2 & w_3 \end{bmatrix} a^{(1)} + b
$$

$$
\hat{y} = a^{(2)} = \sigma(z^{(2)})
$$

## Algorithm

```
Forward_Propagation(X, W, b):
    m = number of samples
    n_layers = number of layers
    
    A = X  // A stores activations, initialize with input
    
    for l = 1 to n_layers:
        Z = matmul(W[l], A) + b[l]
        A = activation_function(Z)
        
        // Store Z[l] and A[l] for backpropagation
        cache[l] = (A_prev, Z, W[l], b[l])
    
    return A, cache
```

## Computational Cost

For a network with:
- $m$ samples in a batch
- Layer $l$ with $n^{(l-1)}$ inputs and $n^{(l)}$ neurons

Computational complexity: $O(m \cdot n^{(l-1)} \cdot n^{(l)})$ per layer

## Key Points

1. **Sequential**: Each layer depends on the output of the previous layer
2. **Deterministic**: Given same inputs and weights, always produces same output
3. **Caching**: Store intermediate values (Z, A) for use in backpropagation
4. **Vectorization**: Use matrix operations for efficiency

## Vectorized Implementation

For a batch of $m$ samples:

$$
Z^{(l)} = W^{(l)} A^{(l-1)} + b^{(l)}
$$

$$
A^{(l)} = \sigma(Z^{(l)})
$$

Where dimensions are:
- $A^{(l-1)}$: $(n^{(l-1)}, m)$
- $W^{(l)}$: $(n^{(l)}, n^{(l-1)})$
- $Z^{(l)}$: $(n^{(l)}, m)$
- $A^{(l)}$: $(n^{(l)}, m)$

## Practical Considerations

### Numerical Stability
- Be careful with activation functions that can cause vanishing gradients
- Use appropriate initialization schemes (Xavier, He initialization)

### Memory Usage
- Storing all intermediate activations increases memory requirements
- For inference (prediction), only keep the final output
- For training, keep intermediate values for backpropagation

### Optimization
- Use batch processing for better GPU utilization
- Vectorize operations using libraries like NumPy, TensorFlow, PyTorch

## Further Reading

- [Neural Networks Basics](Basics.md)
- [Back Propagation](BackPropagation.md)
- [Simple Neural Network Implementation](SimpleNeuralNetwork.md)
