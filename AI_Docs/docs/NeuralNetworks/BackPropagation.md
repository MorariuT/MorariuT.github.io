# Back Propagation

## What is Back Propagation?

Backpropagation is the fundamental algorithm for training neural networks. It computes gradients of the loss function with respect to all parameters (weights and biases) by propagating errors backward through the network.

## The Chain Rule

Backpropagation is built on the chain rule of calculus. For composite functions:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

## Backpropagation Algorithm

### Step 1: Compute Output Layer Gradient

For the output layer with loss function $L$:

$$
\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \odot \sigma'(z^{(L)})
$$

Where:
- $\delta^{(L)}$ = error/gradient at layer $L$
- $\odot$ = element-wise multiplication (Hadamard product)
- $\sigma'$ = derivative of activation function

### Step 2: Backpropagate Through Layers

For layer $l$ (going backward from $L-1$ to $1$):

$$
\delta^{(l)} = (\delta^{(l+1)} \cdot (W^{(l+1)})^T) \odot \sigma'(z^{(l)})
$$

### Step 3: Compute Parameter Gradients

For each layer $l$:

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

$$
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
$$

For a batch of $m$ samples, average the gradients:

$$
\frac{\partial L}{\partial W^{(l)}} = \frac{1}{m} \delta^{(l)} (A^{(l-1)})^T
$$

### Step 4: Update Parameters

Using gradient descent:

$$
W^{(l)} := W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}
$$

$$
b^{(l)} := b^{(l)} - \alpha \frac{\partial L}{\partial b^{(l)}}
$$

Where $\alpha$ is the learning rate.

## Backpropagation Algorithm Pseudocode

```
Backpropagation(y_true, y_pred, cache, L):
    m = batch size
    
    // Step 1: Compute output layer error
    delta = (y_pred - y_true) * sigmoid_derivative(z[L])
    
    // Step 2: Backpropagate through layers
    for l = L down to 1:
        // Compute gradients
        dW[l] = (1/m) * delta @ A[l-1].T
        db[l] = (1/m) * sum(delta, axis=1)
        
        // Propagate error to previous layer
        if l > 1:
            delta = (W[l].T @ delta) * activation_derivative(z[l-1])
    
    // Step 3: Update parameters
    for l = 1 to L:
        W[l] = W[l] - learning_rate * dW[l]
        b[l] = b[l] - learning_rate * db[l]
```

## Example: Simple 2-Layer Network

### Forward Pass
Given input $x$, we compute:
- $z^{(1)} = W^{(1)}x + b^{(1)}$, $a^{(1)} = \text{ReLU}(z^{(1)})$
- $z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$, $\hat{y} = \text{sigmoid}(z^{(2)})$

### Backward Pass

Loss (Binary Cross-Entropy):
$$
L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

Output layer gradient:
$$
\delta^{(2)} = \hat{y} - y
$$

Hidden layer gradient:
$$
\delta^{(1)} = (W^{(2)})^T \delta^{(2)} \odot \text{ReLU}'(z^{(1)})
$$

Parameter gradients:
$$
\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} (a^{(1)})^T, \quad \frac{\partial L}{\partial b^{(2)}} = \delta^{(2)}
$$

$$
\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} x^T, \quad \frac{\partial L}{\partial b^{(1)}} = \delta^{(1)}
$$

## Common Activation Function Derivatives

| Function | $f(z)$ | $f'(z)$ |
|----------|--------|---------|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $f(z)(1-f(z))$ |
| ReLU | $\max(0,z)$ | $\begin{cases} 1 & \text{if } z>0 \\ 0 & \text{otherwise} \end{cases}$ |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ |
| Linear | $z$ | $1$ |

## Challenges in Backpropagation

### Vanishing Gradient Problem
- Gradients become very small in deep networks
- Causes slow learning in early layers
- **Solution**: ReLU activation, careful initialization

### Exploding Gradient Problem
- Gradients become very large
- Can cause numerical instability
- **Solution**: Gradient clipping, normalization

### Computational Cost
- $O(n^2)$ memory for storing intermediate values
- Scales poorly with very deep networks

## Variations and Improvements

### Mini-Batch Gradient Descent
- Use batches of samples instead of single samples
- More efficient GPU computation
- Better gradient estimates

### Momentum
- Keep a moving average of gradients
- Helps escape local minima
- Accelerates convergence

### Adam Optimizer
- Combines momentum with per-parameter adaptive learning rates
- Often works well in practice

## Further Reading

- [Neural Networks Basics](Basics.md)
- [Forward Propagation](ForwardPropagation.md)
- [Simple Neural Network Implementation](SimpleNeuralNetwork.md)
- [Gradient Descent](../Basics/GradientDescent.md)
