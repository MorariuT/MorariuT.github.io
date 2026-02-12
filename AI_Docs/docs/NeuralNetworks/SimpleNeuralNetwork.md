# Simple Neural Network Implementation

## Building a Neural Network from Scratch

This guide walks through implementing a simple feedforward neural network using NumPy, without deep learning frameworks.

## Architecture

We'll build a network with:
- Input layer: configurable size
- Hidden layers: configurable number and sizes
- Output layer: configurable size
- Activation function: ReLU for hidden layers, Sigmoid for output

## Implementation

### 1. Network Class

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize neural network
        
        Args:
            layer_sizes: list of layer sizes, e.g., [784, 128, 64, 10]
            learning_rate: learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights using He initialization, biases to zero"""
        for i in range(self.num_layers - 1):
            # He initialization for ReLU
            w = np.random.randn(self.layer_sizes[i], 
                               self.layer_sizes[i+1]) * np.sqrt(2/self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
```

### 2. Activation Functions

```python
def relu(self, z):
    """ReLU activation function"""
    return np.maximum(0, z)

def relu_derivative(self, z):
    """Derivative of ReLU"""
    return (z > 0).astype(float)

def sigmoid(self, z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(self, z):
    """Derivative of sigmoid"""
    s = self.sigmoid(z)
    return s * (1 - s)
```

### 3. Forward Propagation

```python
def forward_propagation(self, X):
    """
    Forward pass through network
    
    Args:
        X: input data (m, n_features)
    
    Returns:
        cache: intermediate values for backpropagation
        output: final predictions
    """
    cache = []
    A = X
    
    # Forward through all layers except last
    for i in range(self.num_layers - 2):
        Z = np.dot(A, self.weights[i]) + self.biases[i]
        A = self.relu(Z)
        cache.append({'A_prev': A.copy(), 'Z': Z})
    
    # Output layer with sigmoid
    Z_out = np.dot(A, self.weights[-1]) + self.biases[-1]
    A_out = self.sigmoid(Z_out)
    cache.append({'A_prev': A_out.copy(), 'Z': Z_out})
    
    return cache, A_out
```

### 4. Loss Function

```python
def compute_loss(self, y_true, y_pred):
    """
    Compute binary cross-entropy loss
    
    Args:
        y_true: true labels (m, 1)
        y_pred: predicted labels (m, 1)
    
    Returns:
        loss: average loss over batch
    """
    m = y_true.shape[0]
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    loss = -np.mean(y_true * np.log(y_pred) + 
                   (1 - y_true) * np.log(1 - y_pred))
    
    return loss
```

### 5. Backpropagation

```python
def backward_propagation(self, X, y_true, y_pred, cache):
    """
    Backward pass through network
    
    Args:
        X: input data
        y_true: true labels
        y_pred: predicted labels
        cache: values from forward pass
    
    Returns:
        dW: weight gradients
        db: bias gradients
    """
    m = X.shape[0]
    dW = []
    db = []
    
    # Output layer error
    delta = y_pred - y_true  # (m, 1)
    
    # Backpropagate through layers
    for i in range(self.num_layers - 2, -1, -1):
        # Get previous activation
        if i == 0:
            A_prev = X
        else:
            A_prev = cache[i-1]['A_prev']
        
        # Compute gradients
        dW_i = np.dot(A_prev.T, delta) / m
        db_i = np.sum(delta, axis=0, keepdims=True) / m
        
        dW.insert(0, dW_i)
        db.insert(0, db_i)
        
        # Propagate error to previous layer
        if i > 0:
            delta = np.dot(delta, self.weights[i].T) * \
                   self.relu_derivative(cache[i-1]['Z'])
    
    return dW, db
```

### 6. Update Parameters

```python
def update_parameters(self, dW, db):
    """
    Update weights and biases using gradient descent
    
    Args:
        dW: weight gradients
        db: bias gradients
    """
    for i in range(self.num_layers - 1):
        self.weights[i] -= self.learning_rate * dW[i]
        self.biases[i] -= self.learning_rate * db[i]
```

### 7. Training Loop

```python
def train(self, X_train, y_train, epochs=100, batch_size=32):
    """
    Train the network
    
    Args:
        X_train: training data (m, n_features)
        y_train: training labels (m, 1)
        epochs: number of training epochs
        batch_size: size of mini-batches
    """
    m = X_train.shape[0]
    losses = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            cache, y_pred = self.forward_propagation(X_batch)
            
            # Backward pass
            dW, db = self.backward_propagation(X_batch, y_batch, 
                                             y_pred, cache)
            
            # Update parameters
            self.update_parameters(dW, db)
        
        # Compute epoch loss
        _, y_pred_full = self.forward_propagation(X_train)
        loss = self.compute_loss(y_train, y_pred_full)
        losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    return losses
```

### 8. Prediction

```python
def predict(self, X):
    """
    Make predictions
    
    Args:
        X: input data
    
    Returns:
        predictions: binary predictions (0 or 1)
    """
    _, y_pred = self.forward_propagation(X)
    return (y_pred > 0.5).astype(int)
```

## Complete Example

```python
# Generate synthetic data
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, random_state=42)
y = y.reshape(-1, 1)

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create and train network
nn = NeuralNetwork([20, 64, 32, 1], learning_rate=0.01)
losses = nn.train(X, y, epochs=100, batch_size=32)

# Make predictions
predictions = nn.predict(X[:5])
print(predictions)
```

## Key Concepts Implemented

1. **Initialization**: He initialization for weights
2. **Forward Pass**: Layer-by-layer computation with activations
3. **Loss Function**: Binary cross-entropy
4. **Backward Pass**: Gradient computation using chain rule
5. **Optimization**: Mini-batch gradient descent
6. **Training**: Iterative weight updates

## Advantages of This Implementation

✓ Educational - understand every component
✓ No external dependencies (except NumPy)
✓ Flexible architecture
✓ Easy to debug and modify

## Limitations

✗ Much slower than optimized frameworks
✗ Limited to CPU (no GPU support)
✗ No advanced optimizers (momentum, Adam, etc.)
✗ No regularization techniques

## Further Reading

- [Neural Networks Basics](Basics.md)
- [Forward Propagation](ForwardPropagation.md)
- [Back Propagation](BackPropagation.md)

## Next Steps

To extend this implementation:
- Add regularization (L1/L2)
- Implement optimizers (momentum, Adam)
- Add batch normalization
- Support multiple layers flexibility
- Add validation and early stopping
