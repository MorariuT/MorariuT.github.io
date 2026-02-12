# Linear Regression

## What is Linear Regression?

Linear Regression is used for predicting continious data. So the labels vector can be any number in the real space.

Suppose we have a dataset $X \in \mathbb{R}^{N \times P}$. In this notation $N$ is the number of observed points and $P$ is the number of features. $X$ is basically a matrix with $N$ lines and $P$ columns.

We also have a vector $y \in \mathbb{R}^{N}$. This is the label vector. For observation $X_i$, its corresponding label is $y_i$.  

Our goal is to find a function $f(x)$ such that $f(x_i)=y_i$ (or as closely to this).

## How does it work?

Our goal is to find a function $f(x)= \sum_{i=0}^{n} \alpha_i \times x_i$ that outputs a value as closely to $y_i$. So we need to find $\alpha_i$ so it decribes the dataset as closely as possible. 

Let's take a simpler example: 

* $X \in \mathbb{R}^{N \times 1}$ and $y \in \mathbb{R}^{N \times 1}$
* $X_i = 2 \times i + 1$
* $y_i = i$

So we need to find $f(x)=\frac{1}{2}x - \frac{1}{2}$. 

### Loss

For this explanation we will use MSE as a loss funtion. 

\[
L(\alpha) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
\]

Where:

* $\hat{y}_i$ is the predicted value for sample $X_i$
* $y_i$ is the expected value for sample $X_i$

### Optimization

Optimization is the process of finding the $\alpha_i$ parameters. For this we will use the Gradient Descent algorithm. 

At first we start by selecting random $\alpha_i$ and computeing the predictions for $X$. Let's say $\hat{y} = f(X)$. Then we compute the loss $L(\alpha)$. Finally we update each weight with its corresponding derrivative from the loss.