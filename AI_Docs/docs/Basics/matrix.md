# Matrix Algebra

## What is Matrix Algebra?

Matrix Algebra is a branch of mathematics that studies matrices and vectors, and the operations defined on them.

A **matrix** is a rectangular array of numbers arranged in rows and columns.

We denote a matrix:

\[
A \in \mathbb{R}^{m \times n}
\]

where:

- \( m \) = number of rows  
- \( n \) = number of columns  

A **vector** is a special case of a matrix:

- Column vector: \( x \in \mathbb{R}^{n} \) (or \( n \times 1 \))
- Row vector: \( x^T \in \mathbb{R}^{1 \times n} \)

Matrix algebra is fundamental in:

- Machine Learning  
- Computer Graphics  
- Physics  
- Optimization  
- Data Science  

---

## Basic Objects

### Scalars

A scalar is a single number:

\[
\lambda \in \mathbb{R}
\]

---

### Vectors

A column vector:

\[
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
\in \mathbb{R}^{n}
\]

---

### Matrices

A matrix:

\[
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn}
\end{bmatrix}
\in \mathbb{R}^{m \times n}
\]

---


### Dot Product (Inner Product)

The **dot product** is defined between two vectors of the same dimension.

If:

\[
x, y \in \mathbb{R}^{n}
\]

Then:

\[
x \cdot y = x^T y = \sum_{i=1}^{n} x_i y_i
\]

Example:

\[
x =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix},
\quad
y =
\begin{bmatrix}
4 \\
5 \\
6
\end{bmatrix}
\]

\[
x \cdot y = 1\cdot4 + 2\cdot5 + 3\cdot6 = 32
\]

#### Geometric Interpretation

The dot product also relates to the angle between vectors:

\[
x \cdot y = \|x\| \|y\| \cos(\theta)
\]

where:

- \( \theta \) is the angle between the vectors  
- If \( x \cdot y = 0 \), the vectors are **orthogonal**

---

## Matrix Operations

### 1. Matrix Addition

Two matrices can be added only if they have the same shape:

\[
A, B \in \mathbb{R}^{m \times n}
\]

\[
C = A + B
\]

Element-wise:

\[
c_{ij} = a_{ij} + b_{ij}
\]

---

### 2. Scalar Multiplication

\[
B = \lambda A
\]

Each element is multiplied by the scalar:

\[
b_{ij} = \lambda a_{ij}
\]

---

### 3. Matrix Multiplication

Matrix multiplication is defined only if dimensions match.

If:

\[
A \in \mathbb{R}^{m \times n}, \quad B \in \mathbb{R}^{n \times p}
\]

Then:

\[
C = AB \in \mathbb{R}^{m \times p}
\]

Each element:

\[
c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
\]

Important properties:

- Associative: \( (AB)C = A(BC) \)
- Distributive: \( A(B+C) = AB + AC \)
- **Not commutative**: \( AB \neq BA \)

---

### 4. Transpose

The transpose swaps rows and columns:

\[
A^T \in \mathbb{R}^{n \times m}
\]

\[
(A^T)_{ij} = A_{ji}
\]

Properties:

\[
(A + B)^T = A^T + B^T
\]

\[
(AB)^T = B^T A^T
\]

---
## Special Matrices

### Identity Matrix

\[
I_n =
\begin{bmatrix}
1 & 0 & \dots & 0 \\
0 & 1 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & 1
\end{bmatrix}
\]

Property:

\[
AI = IA = A
\]

---

### Zero Matrix

All elements are zero:

\[
0_{m \times n}
\]

---

### Diagonal Matrix

Non-zero elements only on the diagonal:

\[
D =
\begin{bmatrix}
d_1 & 0 & \dots & 0 \\
0 & d_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & d_n
\end{bmatrix}
\]

## Inverse Matrix

If:

\[
A \in \mathbb{R}^{n \times n}
\]

and \( \det(A) \neq 0 \), then there exists:

\[
A^{-1}
\]

such that:

\[
AA^{-1} = A^{-1}A = I
\]

In Machine Learning, we often compute:

\[
A^{-1}b
\]

to solve linear systems.

---

## Solving Linear Systems

Given:

\[
Ax = b
\]

If \( A \) is invertible:

\[
x = A^{-1}b
\]
