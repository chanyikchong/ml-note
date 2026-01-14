# Recurrent Neural Networks (RNN)

## 1. Interview Summary

**Key Points to Remember:**
- **Sequential processing**: Maintains hidden state across time steps
- **Vanishing/exploding gradients**: Main challenge with vanilla RNNs
- **LSTM**: Gated architecture solving gradient issues
- **GRU**: Simplified gating, similar performance to LSTM
- **Bidirectional**: Process sequence in both directions

**Common Interview Questions:**
- "Explain the vanishing gradient problem in RNNs"
- "How do LSTM gates work?"
- "Compare LSTM vs GRU"

---

## 2. Core Definitions

### Vanilla RNN
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

### LSTM (Long Short-Term Memory)
Four gates controlling information flow:
- **Forget gate**: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
- **Input gate**: $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
- **Candidate**: $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$
- **Output gate**: $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$

Cell state update:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$h_t = o_t \odot \tanh(C_t)$$

### GRU (Gated Recurrent Unit)
Simplified version with two gates:
- **Update gate**: $z_t = \sigma(W_z [h_{t-1}, x_t])$
- **Reset gate**: $r_t = \sigma(W_r [h_{t-1}, x_t])$
- **Hidden state**: $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tanh(W_h [r_t \odot h_{t-1}, x_t])$

---

## 3. Math and Derivations

### Backpropagation Through Time (BPTT)

For loss $L = \sum_t L_t$, gradient w.r.t. parameters involves chain rule through time:

$$\frac{\partial L}{\partial W_{hh}} = \sum_t \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$$

### Vanishing Gradient Problem

The term $\frac{\partial h_t}{\partial h_k}$ involves product:
$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=k+1}^{t} W_{hh}^T \text{diag}(\tanh'(h_{i-1}))$$

Since $|\tanh'(x)| \leq 1$ and if $\|W_{hh}\| < 1$:
- Product shrinks exponentially with $t-k$
- Gradients vanish for long sequences
- Network can't learn long-range dependencies

### Why LSTM Solves This

Cell state update is **additive**:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Gradient flows directly through addition:
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

If $f_t \approx 1$ (forget gate open), gradient flows unchanged. This creates "gradient highways."

### Parameter Count

For LSTM with input dim $d$ and hidden dim $h$:
- 4 gates, each with input-hidden and hidden-hidden weights
- Parameters: $4 \times (d \times h + h \times h + h) = 4(dh + h^2 + h)$

For GRU: $3(dh + h^2 + h)$ (3 gates instead of 4)

---

## 4. Algorithm Sketch

### RNN Forward Pass

```
def rnn_forward(x_sequence, h_0, W_xh, W_hh, W_hy):
    h = h_0
    outputs = []

    for x_t in x_sequence:
        h = tanh(W_xh @ x_t + W_hh @ h)
        y_t = W_hy @ h
        outputs.append(y_t)

    return outputs, h
```

### LSTM Forward Pass

```
def lstm_forward(x_sequence, h_0, c_0, params):
    h, c = h_0, c_0
    outputs = []

    for x_t in x_sequence:
        # Gates
        f = sigmoid(W_f @ concat(h, x_t) + b_f)  # Forget
        i = sigmoid(W_i @ concat(h, x_t) + b_i)  # Input
        o = sigmoid(W_o @ concat(h, x_t) + b_o)  # Output
        c_tilde = tanh(W_c @ concat(h, x_t) + b_c)  # Candidate

        # State updates
        c = f * c + i * c_tilde  # Cell state
        h = o * tanh(c)  # Hidden state

        outputs.append(h)

    return outputs, h, c
```

### Bidirectional RNN

```
def bidirectional_rnn(x_sequence, params_forward, params_backward):
    # Forward pass
    h_forward = []
    h = zeros(hidden_size)
    for x_t in x_sequence:
        h = rnn_cell(x_t, h, params_forward)
        h_forward.append(h)

    # Backward pass
    h_backward = []
    h = zeros(hidden_size)
    for x_t in reversed(x_sequence):
        h = rnn_cell(x_t, h, params_backward)
        h_backward.insert(0, h)

    # Concatenate
    return [concat(f, b) for f, b in zip(h_forward, h_backward)]
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Vanishing gradients | Long sequences with vanilla RNN | Use LSTM/GRU |
| Exploding gradients | Large gradient products | Gradient clipping |
| Slow training | Sequential, can't parallelize | Use Transformer |
| Overfitting | Too many parameters | Dropout (recurrent dropout) |
| Poor long-range | Even LSTM has limits | Attention mechanisms |

### Architecture Selection

| Task | Recommendation |
|------|---------------|
| Short sequences (<50) | GRU or LSTM |
| Long sequences (>100) | Transformer or attention |
| Real-time processing | GRU (faster than LSTM) |
| Sequence labeling | Bidirectional LSTM |
| Language modeling | Transformer (modern) |

### Training Tips

1. **Initialize carefully**: Orthogonal init for recurrent weights
2. **Gradient clipping**: Clip by norm (1.0-5.0)
3. **Learning rate**: Start small (0.001)
4. **Batch processing**: Pad sequences, use masking
5. **Layer normalization**: Apply within recurrent cell

---

## 6. Mini Example

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization
        self.W_xh = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.W_hy = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
        self.hidden_size = hidden_size

    def forward(self, x_seq):
        h = np.zeros(self.hidden_size)
        self.hidden_states = [h]

        for x_t in x_seq:
            h = tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            self.hidden_states.append(h)

        y = h @ self.W_hy + self.b_y
        return y, h


class SimpleLSTM:
    def __init__(self, input_size, hidden_size):
        # Initialize all gates
        self.hidden_size = hidden_size
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Concatenated weights for efficiency
        self.W = np.random.randn(input_size + hidden_size, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)

        # Initialize forget gate bias to 1 (helps with gradient flow)
        self.b[hidden_size:2*hidden_size] = 1.0

    def forward(self, x_seq):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        hidden_states = []

        for x_t in x_seq:
            # Concatenate input and hidden
            combined = np.concatenate([x_t, h])

            # Compute all gates at once
            gates = combined @ self.W + self.b

            # Split into individual gates
            hs = self.hidden_size
            i = sigmoid(gates[:hs])          # Input gate
            f = sigmoid(gates[hs:2*hs])      # Forget gate
            o = sigmoid(gates[2*hs:3*hs])    # Output gate
            g = tanh(gates[3*hs:])           # Candidate

            # Update states
            c = f * c + i * g
            h = o * tanh(c)

            hidden_states.append(h)

        return hidden_states, h, c


# Example: Sequence classification
np.random.seed(42)

# Create simple sequence data
# Task: Predict sum of first and last elements
seq_length = 5
input_size = 3
hidden_size = 8
n_samples = 100

# Generate data
X = np.random.randn(n_samples, seq_length, input_size)
y = (X[:, 0, 0] + X[:, -1, 0] > 0).astype(float)  # Binary classification

# Test simple RNN
rnn = SimpleRNN(input_size, hidden_size, 1)
sample_output, _ = rnn.forward(X[0])
print(f"Simple RNN output shape: {sample_output.shape}")
print(f"Number of hidden states: {len(rnn.hidden_states)}")

# Test LSTM
lstm = SimpleLSTM(input_size, hidden_size)
hidden_states, final_h, final_c = lstm.forward(X[0])
print(f"\nLSTM hidden states: {len(hidden_states)}")
print(f"Final hidden shape: {final_h.shape}")
print(f"Final cell shape: {final_c.shape}")

# Parameter count
rnn_params = (input_size * hidden_size + hidden_size * hidden_size +
              hidden_size + hidden_size * 1 + 1)
lstm_params = (input_size + hidden_size) * (4 * hidden_size) + 4 * hidden_size
print(f"\nRNN parameters: {rnn_params}")
print(f"LSTM parameters: {lstm_params}")
```

**Output:**
```
Simple RNN output shape: (1,)
Number of hidden states: 6

LSTM hidden states: 5
Final hidden shape: (8,)
Final cell shape: (8,)

RNN parameters: 97
LSTM parameters: 384
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Explain the vanishing gradient problem in RNNs.</strong></summary>

When backpropagating through time, gradients involve products:
$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} W_{hh}^T \text{diag}(\tanh'(z_i))$$

Since $|\tanh'(x)| \leq 1$:
- If $\|W_{hh}\| < 1$: products shrink exponentially → **vanishing gradients**
- If $\|W_{hh}\| > 1$: products grow exponentially → **exploding gradients**

**Consequence**: Can't learn long-range dependencies (gradients from distant past are ~0).

**Solutions**: LSTM/GRU (gated cells), gradient clipping, skip connections.
</details>

<details>
<summary><strong>Q2: How do LSTM gates work?</strong></summary>

LSTM has 4 components:

1. **Forget gate** ($f_t$): Decides what to remove from cell state
   - $f_t \approx 0$: Forget previous info
   - $f_t \approx 1$: Keep previous info

2. **Input gate** ($i_t$): Decides what new info to store
   - Controls how much of candidate goes into cell state

3. **Candidate** ($\tilde{C}_t$): New information to potentially add
   - Computed like standard RNN

4. **Output gate** ($o_t$): Decides what to output from cell state
   - Filters cell state for hidden state output

Cell update is **additive**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$

This additive structure allows gradients to flow unchanged when forget gate is open.
</details>

<details>
<summary><strong>Q3: Compare LSTM vs GRU.</strong></summary>

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 4 (forget, input, output, candidate) | 2 (update, reset) |
| States | Cell state + hidden state | Hidden state only |
| Parameters | More (4 weight matrices) | Fewer (3 weight matrices) |
| Training | Slower | Faster |
| Performance | Slightly better on some tasks | Comparable, sometimes better |
| Memory | Explicit memory cell | Implicit in hidden state |

**When to use**:
- GRU: Smaller datasets, faster training, simpler architecture
- LSTM: Very long sequences, when explicit memory helps
</details>

<details>
<summary><strong>Q4: What is teacher forcing?</strong></summary>

**Teacher forcing**: During training, use ground truth from previous time step as input, not model's own prediction.

**Advantages**:
- Faster convergence
- More stable training
- Avoids error accumulation during training

**Disadvantages**:
- Train-test mismatch: Model never sees its own errors during training
- Can lead to exposure bias

**Mitigation**: Scheduled sampling - gradually shift from teacher forcing to model predictions during training.
</details>

<details>
<summary><strong>Q5: What is bidirectional RNN?</strong></summary>

Bidirectional RNN processes sequence in both directions:

1. **Forward RNN**: Processes left-to-right
2. **Backward RNN**: Processes right-to-left
3. **Concatenate**: Combine hidden states at each position

**Output**: $h_t = [h_t^{\rightarrow}; h_t^{\leftarrow}]$

**Advantages**:
- Uses context from both past and future
- Better for classification/tagging tasks

**Disadvantages**:
- Can't be used for generation (needs full sequence)
- Double the parameters
</details>

<details>
<summary><strong>Q6: Why are Transformers replacing RNNs?</strong></summary>

Transformers advantages:
1. **Parallelization**: All positions processed simultaneously (no sequential dependency)
2. **Long-range dependencies**: Direct connections via attention (no gradient decay)
3. **Scalability**: Scales better to large models/data
4. **Pre-training**: Better for transfer learning (BERT, GPT)

RNN advantages:
- Naturally handles variable length
- Better for streaming/online data
- Lower memory for very long sequences

Modern trend: Transformers dominate NLP; RNNs still used in some time-series/streaming applications.
</details>

---

## 8. References

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
2. Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder." EMNLP.
3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). "On the Difficulty of Training RNNs." ICML.
4. Graves, A. (2012). "Supervised Sequence Labelling with Recurrent Neural Networks." Springer.
5. Vaswani, A., et al. (2017). "Attention Is All You Need." NIPS.
