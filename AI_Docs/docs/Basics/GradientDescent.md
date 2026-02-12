# Gradient Descent

## What is Gradient Descent?

Gradient Descent (GD) is one of the simplest and most widely used optimization algorithms.  
It is used in models ranging from:

- Linear Regression  
- Logistic Regression  
- Support Vector Machines  
- Neural Networks  
- Deep Learning  

Its goal is simple:

\[
\min_{\theta} J(\theta)
\]

where:

- \( \theta \) = model parameters  
- \( J(\theta) \) = cost (loss) function  

Gradient Descent iteratively adjusts parameters to minimize the cost.

---

## Intuition

Imagine you are standing on a mountain and want to reach the lowest point.

- The **gradient** tells you the direction of steepest ascent on a specific dimension.
- To go downhill, you move in the **opposite direction** of the gradient.
- You repeat this process until you reach a minimum.

---

## Mathematical Formulation

Let:

\[
J(\alpha)
\]

be a differentiable cost function.

The gradient is:

\[
\nabla J(\alpha)
\]

This tells us for each weight \(\alpha_i\) the direction we need to move it to lower \(J(\alpha)\) on the \(i\)-th component.

The update rule is:

\[
\alpha_i := \alpha_i - \eta \nabla J(\alpha)_i
\]

Each weight \(\alpha_i\) is updated with its corresponding gradient, in such a way to lower \(J(\alpha)\).

A learning rate is used to control the descent so that we do not overshoot the minimum point.

Where:

- \( \eta \) = learning rate  
- \( \nabla J(\alpha) \) = gradient vector for all \(\alpha\) weights

This update is repeated until convergence.

---

## Learning Rate

The learning rate \( \eta \) controls the step size.

If:

- \( \eta \) is too large → the algorithm may diverge  
- \( \eta \) is too small → convergence is very slow  
- \( \eta \) is well chosen → smooth and stable convergence  

Choosing a good learning rate is critical.

---

## Convergence

Gradient Descent stops when:

- The gradient becomes very small:
  \[
  \|\nabla J(\theta)\| \approx 0
  \]
- The change in parameters is very small  
- A fixed number of iterations is reached  

For convex functions, Gradient Descent converges to the global minimum.  
For non-convex functions (like deep neural networks), it may converge to a local minimum. In such cases, special optimizations are used.

---

## Example: Gradient Descent in Python

Here is a simple example of Gradient Descent applied to Linear Regression:

```python
import numpy as np

# Sample dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Initialize parameters
alpha = 0.0  # slope
bias = 0.0   # intercept
learning_rate = 0.01
iterations = 1000
N = len(y)

# Gradient Descent Loop
for _ in range(iterations):
    y_pred = alpha * X + bias
    # Compute gradients
    d_alpha = (-2/N) * np.sum(X * (y - y_pred))
    d_bias = (-2/N) * np.sum(y - y_pred)
    # Update parameters
    alpha -= learning_rate * d_alpha
    bias -= learning_rate * d_bias

print(f"Learned parameters: alpha = {alpha:.2f}, bias = {bias:.2f}")
```