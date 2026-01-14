# Neural Network Fundamentals

## 1. Interview Summary

**Key Points to Remember:**
- **MLP**: Multi-Layer Perceptron - fully connected feedforward network
- **Backpropagation**: Efficient gradient computation via chain rule
- **Activation functions**: ReLU (most common), sigmoid, tanh
- **Universal approximation**: Sufficiently wide network can approximate any function
- Know the forward pass, backward pass, and gradient flow

**Common Interview Questions:**
- "Explain backpropagation"
- "Why do we need non-linear activation functions?"
- "What is the vanishing gradient problem?"

---

## 2. Core Definitions

### Multi-Layer Perceptron (MLP)
A feedforward neural network with:
- Input layer
- One or more hidden layers
- Output layer
- Fully connected (dense) layers

### Forward Pass

$$h^{(l)} = \sigma(W^{(l)}h^{(l-1)} + b^{(l)})$$

### Backpropagation
Efficient computation of $\frac{\partial \mathcal{L}}{\partial W}$ using chain rule:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

### Common Activation Functions

| Function | Formula | Range | Derivative |
|----------|---------|-------|------------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | (0, 1) | $\sigma(z)(1-\sigma(z))$ |
| Tanh | $\tanh(z)$ | (-1, 1) | $1 - \tanh^2(z)$ |
| ReLU | $\max(0, z)$ | [0, ∞) | 1 if z > 0, 0 otherwise |

---

## 3. Math and Derivations

### Backpropagation Derivation

For loss $\mathcal{L}$ and layer output $a^{(l)} = \sigma(z^{(l)})$ where $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$:

**Output layer gradient:**

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}}$$

**Hidden layer gradient (recursive):**

$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

**Weight gradient:**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

### Universal Approximation Theorem

A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, under mild assumptions on the activation function.

---

## 4. Algorithm Sketch

### Forward Pass
```
Input: x, weights W, biases b
a[0] = x

For l = 1 to L:
    z[l] = W[l] @ a[l-1] + b[l]
    a[l] = activation(z[l])

Output: a[L]
```

### Backward Pass
```
# Compute output layer delta
delta[L] = loss_gradient(a[L], y) * activation_derivative(z[L])

# Backpropagate
For l = L-1 down to 1:
    delta[l] = (W[l+1].T @ delta[l+1]) * activation_derivative(z[l])

# Compute gradients
For l = 1 to L:
    dW[l] = delta[l] @ a[l-1].T
    db[l] = delta[l]
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Vanishing gradients | Sigmoid/tanh saturate | Use ReLU, proper initialization |
| Exploding gradients | Deep networks, large weights | Gradient clipping, normalization |
| Dead ReLU | Neurons stuck at 0 | LeakyReLU, proper initialization |
| Not using non-linearity | Linear network collapses to single layer | Always use activation |
| Wrong initialization | Too large/small initial weights | He/Xavier initialization |

---

## 6. Mini Example

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return z1, a1, z2

def backward(X, y, z1, a1, z2, W2):
    m = len(y)
    dz2 = z2 - y  # Assuming MSE loss
    dW2 = (1/m) * a1.T @ dz2
    db2 = (1/m) * np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = (1/m) * X.T @ dz1
    db1 = (1/m) * np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Why do we need non-linear activation functions?</strong></summary>

Without non-linear activations, a multi-layer network would collapse to a single linear transformation. Composing linear functions gives a linear function:
$f(x) = W_2(W_1 x) = (W_2 W_1)x = Wx$

Non-linear activations allow the network to learn complex, non-linear decision boundaries.
</details>

<details>
<summary><strong>Q2: Explain the vanishing gradient problem.</strong></summary>

With sigmoid or tanh activations, gradients are multiplied by derivatives that are < 1 (e.g., sigmoid' max is 0.25). In deep networks, these multiply together:

$\frac{\partial \mathcal{L}}{\partial W^{(1)}} \propto \prod_{l=1}^{L} \sigma'(z^{(l)})$

This product becomes exponentially small, making early layer weights update very slowly.

**Solutions**: ReLU (derivative is 1), skip connections, proper initialization.
</details>

<details>
<summary><strong>Q3: What is the chain rule in backpropagation?</strong></summary>

Backpropagation uses the chain rule to decompose gradients:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}} \cdots \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

This allows efficient gradient computation in $O(n)$ time per sample.
</details>

<details>
<summary><strong>Q4: Compare ReLU, sigmoid, and tanh activations.</strong></summary>

| Property | ReLU | Sigmoid | Tanh |
|----------|------|---------|------|
| Range | [0, ∞) | (0, 1) | (-1, 1) |
| Gradient | 1 or 0 | ≤ 0.25 | ≤ 1 |
| Zero-centered | No | No | Yes |
| Vanishing gradient | Less | Yes | Yes |
| Dead neurons | Yes | No | No |
| Speed | Fast | Slow (exp) | Slow (exp) |
</details>

<details>
<summary><strong>Q5: What is Xavier/Glorot initialization?</strong></summary>

Initialize weights as:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

This keeps variance of activations roughly constant across layers, preventing vanishing/exploding signals during forward pass.

**He initialization** (for ReLU): $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$
</details>

<details>
<summary><strong>Q6: What is the universal approximation theorem?</strong></summary>

A single-hidden-layer network with finite neurons and non-polynomial activation can approximate any continuous function on a compact set to arbitrary accuracy.

**Caveats**:
- Says nothing about how many neurons needed
- Says nothing about learning (optimization)
- Compact sets only
- Deeper networks often more efficient in practice
</details>

---

## 8. References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Rumelhart, D., Hinton, G., & Williams, R. (1986). "Learning Representations by Back-propagating Errors." Nature.
3. Glorot, X., & Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." AISTATS.
4. He, K., et al. (2015). "Delving Deep into Rectifiers." ICCV.
5. Hornik, K. (1991). "Approximation Capabilities of Multilayer Feedforward Networks." Neural Networks.
