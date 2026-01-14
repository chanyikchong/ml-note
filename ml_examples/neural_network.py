"""
Simple Neural Network Demo
==========================

Demonstrates:
1. Multi-layer perceptron from scratch (NumPy)
2. Forward propagation
3. Backpropagation
4. Training with mini-batch gradient descent

Note: Using NumPy for educational clarity. For production, use PyTorch/TensorFlow.
"""

import numpy as np
from typing import List, Tuple, Callable


class NeuralNetwork:
    """
    Simple feedforward neural network.

    Architecture: Input -> Hidden Layer(s) -> Output
    Activation: ReLU for hidden, Sigmoid for binary / Softmax for multi-class
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        output_activation: str = "sigmoid",
        learning_rate: float = 0.01,
        random_state: int = None,
    ):
        """
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            activation: Hidden layer activation ('relu', 'tanh')
            output_activation: Output activation ('sigmoid', 'softmax')
            learning_rate: Learning rate for gradient descent
            random_state: Random seed
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            std = np.sqrt(2 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * std
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def _tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def _tanh_derivative(self, z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(z) ** 2

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _activate(self, z: np.ndarray, layer: int) -> np.ndarray:
        """Apply activation function."""
        if layer == len(self.weights) - 1:  # Output layer
            if self.output_activation == "sigmoid":
                return self._sigmoid(z)
            else:
                return self._softmax(z)
        else:  # Hidden layer
            if self.activation == "relu":
                return self._relu(z)
            else:
                return self._tanh(z)

    def _activate_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of hidden layer activation."""
        if self.activation == "relu":
            return self._relu_derivative(z)
        else:
            return self._tanh_derivative(z)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation.

        Returns:
            Tuple of (activations, pre_activations)
        """
        activations = [X]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            pre_activations.append(z)
            a = self._activate(z, i)
            activations.append(a)

        return activations, pre_activations

    def backward(
        self,
        y: np.ndarray,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation (backprop).

        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = y.shape[0]
        grad_weights = []
        grad_biases = []

        # Output layer gradient
        # For cross-entropy + sigmoid/softmax: delta = a - y
        delta = activations[-1] - y

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            grad_W = (1 / m) * activations[i].T @ delta
            grad_b = (1 / m) * np.sum(delta, axis=0, keepdims=True)

            grad_weights.insert(0, grad_W)
            grad_biases.insert(0, grad_b)

            if i > 0:  # Not input layer
                delta = delta @ self.weights[i].T * self._activate_derivative(
                    pre_activations[i - 1]
                )

        return grad_weights, grad_biases

    def update_weights(
        self, grad_weights: List[np.ndarray], grad_biases: List[np.ndarray]
    ):
        """Update weights using gradient descent."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the neural network.

        Args:
            X: Training features
            y: Training labels (one-hot for multi-class)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Print progress

        Returns:
            List of loss values per epoch
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        losses = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                activations, pre_activations = self.forward(X_batch)

                # Backward pass
                grad_W, grad_b = self.backward(y_batch, activations, pre_activations)

                # Update weights
                self.update_weights(grad_W, grad_b)

            # Compute epoch loss
            activations, _ = self.forward(X)
            loss = self.compute_loss(y, activations[-1])
            losses.append(loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        if proba.shape[1] == 1:
            return (proba >= threshold).astype(int).flatten()
        else:
            return np.argmax(proba, axis=1)


def run_demo():
    """Run the neural network demonstration."""
    np.random.seed(42)

    print("1. Generating XOR-like classification data...")
    # XOR problem - not linearly separable
    n_samples = 400
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Class distribution: {np.bincount(y_train)}")

    print("\n2. Creating neural network: 2 -> 8 -> 4 -> 1")
    nn = NeuralNetwork(
        layer_sizes=[2, 8, 4, 1],
        activation="relu",
        output_activation="sigmoid",
        learning_rate=0.1,
        random_state=42,
    )

    print("   Weight shapes:")
    for i, W in enumerate(nn.weights):
        print(f"     Layer {i}: {W.shape}")

    print("\n3. Training neural network...")
    losses = nn.fit(X_train, y_train, epochs=100, batch_size=32, verbose=False)
    print(f"   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")

    print("\n4. Evaluating on test set...")
    y_pred = nn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"   Test accuracy: {accuracy:.4f}")

    # Confusion matrix
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    print("\n   Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"   Actual 0    {tn:4d}   {fp:4d}")
    print(f"   Actual 1    {fn:4d}   {tp:4d}")

    print("\n5. Single linear layer comparison (should fail on XOR)...")
    nn_linear = NeuralNetwork(
        layer_sizes=[2, 1],  # No hidden layer
        learning_rate=0.1,
        random_state=42,
    )
    losses_linear = nn_linear.fit(X_train, y_train, epochs=100, verbose=False)
    y_pred_linear = nn_linear.predict(X_test)
    acc_linear = np.mean(y_pred_linear == y_test)
    print(f"   Linear model accuracy: {acc_linear:.4f} (expected ~50%)")

    print("\n6. Gradient check (numerical vs analytical)...")
    # Simple gradient check on first weight
    eps = 1e-5
    nn_check = NeuralNetwork([2, 4, 1], learning_rate=0.1, random_state=42)

    # Analytical gradient
    activations, pre_activations = nn_check.forward(X_train[:10])
    grad_W, grad_b = nn_check.backward(y_train[:10].reshape(-1, 1), activations, pre_activations)

    # Numerical gradient for W[0][0,0]
    W_orig = nn_check.weights[0][0, 0]

    nn_check.weights[0][0, 0] = W_orig + eps
    act_plus, _ = nn_check.forward(X_train[:10])
    loss_plus = nn_check.compute_loss(y_train[:10].reshape(-1, 1), act_plus[-1])

    nn_check.weights[0][0, 0] = W_orig - eps
    act_minus, _ = nn_check.forward(X_train[:10])
    loss_minus = nn_check.compute_loss(y_train[:10].reshape(-1, 1), act_minus[-1])

    nn_check.weights[0][0, 0] = W_orig
    numerical_grad = (loss_plus - loss_minus) / (2 * eps)
    analytical_grad = grad_W[0][0, 0]

    print(f"   Numerical gradient:  {numerical_grad:.6f}")
    print(f"   Analytical gradient: {analytical_grad:.6f}")
    print(f"   Relative error: {abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + 1e-8):.6f}")


if __name__ == "__main__":
    run_demo()
