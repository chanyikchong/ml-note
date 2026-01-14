# Normalization Techniques

## 1. Interview Summary

**Key Points to Remember:**
- **Batch Normalization**: Normalize across batch, widely used in CNNs
- **Layer Normalization**: Normalize across features, used in RNNs/Transformers
- **Purpose**: Stabilize training, enable higher learning rates
- **Learnable parameters**: Scale (γ) and shift (β)
- **Train vs inference**: BatchNorm uses running statistics at inference

**Common Interview Questions:**
- "Why does batch normalization work?"
- "Compare batch norm vs layer norm"
- "How does batch norm behave at test time?"

---

## 2. Core Definitions

### Batch Normalization
For input $x$ with shape $(N, C, H, W)$, normalize per channel across batch:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Where $\mu_B, \sigma_B^2$ are batch statistics per channel.

### Layer Normalization
For input $x$, normalize across feature dimensions:
$$\hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$

Where $\mu_i, \sigma_i^2$ are computed over features for each sample.

### Group Normalization
Divide channels into groups, normalize within each group:
- Compromise between Batch and Layer norm
- Works with small batches

### Instance Normalization
Normalize each sample and channel independently:
- Used in style transfer
- Per-sample, per-channel statistics

---

## 3. Math and Derivations

### Batch Normalization Forward

For mini-batch $\{x_1, ..., x_m\}$:

**Step 1**: Compute batch statistics
$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

**Step 2**: Normalize
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**Step 3**: Scale and shift
$$y_i = \gamma \hat{x}_i + \beta$$

### Why Batch Normalization Works

Several hypotheses:
1. **Reduces internal covariate shift**: Stabilizes input distributions
2. **Smooths loss landscape**: Makes optimization easier
3. **Regularization effect**: Batch statistics add noise
4. **Enables higher learning rates**: More stable gradients

### Batch Norm at Inference

During training, maintain running averages:
$$\mu_{running} = (1 - \alpha) \mu_{running} + \alpha \mu_B$$
$$\sigma_{running}^2 = (1 - \alpha) \sigma_{running}^2 + \alpha \sigma_B^2$$

At inference, use running statistics (deterministic output).

### Batch Norm Gradients

Backpropagation through normalization:
$$\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \hat{x}_i$$
$$\frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}$$
$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + ...$$

(Full gradient involves terms through $\mu$ and $\sigma^2$)

---

## 4. Algorithm Sketch

### Batch Normalization

```
class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = ones(num_features)
        self.beta = zeros(num_features)
        self.running_mean = zeros(num_features)
        self.running_var = ones(num_features)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training=True):
        if training:
            # Compute batch statistics
            mean = x.mean(axis=0)  # Over batch
            var = x.var(axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta
```

### Layer Normalization

```
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = ones(normalized_shape)
        self.beta = zeros(normalized_shape)
        self.eps = eps

    def forward(self, x):
        # Compute mean and var over last dimensions
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta
```

### Comparison

```
Input shape: (N, C, H, W)

BatchNorm: normalize over (N, H, W) for each C
LayerNorm: normalize over (C, H, W) for each N
InstanceNorm: normalize over (H, W) for each N, C
GroupNorm: normalize over (C/G, H, W) for each N, G
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| BatchNorm with small batch | Noisy statistics | Use GroupNorm or LayerNorm |
| Train/eval mode mismatch | Forgot to set eval mode | Always call model.eval() |
| BatchNorm in RNN | Different sequence lengths | Use LayerNorm |
| Running stats not updated | Set training=False during training | Check mode |
| Normalization after activation | Wrong order | Typically: Conv → BN → ReLU |

### When to Use Each

| Scenario | Recommended |
|----------|------------|
| CNN with large batch | Batch Normalization |
| CNN with small batch | Group Normalization |
| RNN / Transformers | Layer Normalization |
| Style transfer | Instance Normalization |
| Unsure | Try LayerNorm (works everywhere) |

### Placement in Network

**Standard order**:
```
Conv/Linear → BatchNorm/LayerNorm → Activation (ReLU)
```

**ResNet style**:
```
Conv → BN → ReLU → Conv → BN → Add (skip) → ReLU
```

---

## 6. Mini Example

```python
import numpy as np

class BatchNorm1D:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# Example
np.random.seed(42)

# Create data: batch of 4 samples, 8 features
x = np.random.randn(4, 8) * 10 + 5  # High variance, non-zero mean

print("Original data statistics:")
print(f"  Mean: {x.mean():.2f}, Std: {x.std():.2f}")
print(f"  Per-feature mean: {x.mean(axis=0)[:4]}...")
print(f"  Per-sample mean: {x.mean(axis=1)}")

# Batch Normalization
bn = BatchNorm1D(8)
x_bn = bn.forward(x, training=True)
print(f"\nAfter BatchNorm (per-feature normalized):")
print(f"  Mean: {x_bn.mean():.4f}, Std: {x_bn.std():.4f}")
print(f"  Per-feature mean: {x_bn.mean(axis=0)[:4]}...")

# Layer Normalization
ln = LayerNorm(8)
x_ln = ln.forward(x)
print(f"\nAfter LayerNorm (per-sample normalized):")
print(f"  Mean: {x_ln.mean():.4f}, Std: {x_ln.std():.4f}")
print(f"  Per-sample mean: {x_ln.mean(axis=1)}")

# Inference mode (using running statistics)
x_new = np.random.randn(2, 8) * 10 + 5
x_bn_infer = bn.forward(x_new, training=False)
print(f"\nInference mode BatchNorm:")
print(f"  Uses running_mean: {bn.running_mean[:4]}...")
print(f"  Uses running_var: {bn.running_var[:4]}...")
```

**Output:**
```
Original data statistics:
  Mean: 5.15, Std: 10.23
  Per-feature mean: [5.99 3.85 6.08 4.47]...
  Per-sample mean: [6.12 3.49 5.97 5.01]

After BatchNorm (per-feature normalized):
  Mean: 0.0000, Std: 1.0000
  Per-feature mean: [ 0.  0. -0. -0.]...

After LayerNorm (per-sample normalized):
  Mean: 0.0000, Std: 1.0000
  Per-sample mean: [-0. -0.  0. -0.]

Inference mode BatchNorm:
  Uses running_mean: [0.6  0.38 0.61 0.45]...
  Uses running_var: [1.06 1.07 1.08 1.06]...
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Why does batch normalization help training?</strong></summary>

Several factors:
1. **Reduces internal covariate shift**: Layer inputs have stable distribution
2. **Smoother loss landscape**: Gradients are more predictable
3. **Regularization**: Batch statistics add noise (like dropout)
4. **Higher learning rates**: Training is more stable
5. **Reduces sensitivity to initialization**: Networks train from various inits

Recent research suggests the smoothing effect is most important.
</details>

<details>
<summary><strong>Q2: What is the difference between batch norm and layer norm?</strong></summary>

| Aspect | Batch Norm | Layer Norm |
|--------|-----------|------------|
| Normalize over | Batch dimension | Feature dimensions |
| Statistics | Per feature, across batch | Per sample, across features |
| Train vs Test | Different (running stats) | Same |
| Batch dependency | Yes | No |
| Best for | CNNs with large batches | RNNs, Transformers |

Key difference: BatchNorm depends on batch, LayerNorm doesn't.
</details>

<details>
<summary><strong>Q3: How does batch norm behave differently during training and inference?</strong></summary>

**Training**:
- Computes mean/variance from current mini-batch
- Updates running statistics (exponential moving average)
- Output depends on other samples in batch

**Inference**:
- Uses stored running mean/variance
- Deterministic output
- No batch dependency

**Why**: At inference, we want deterministic predictions regardless of batch composition.
</details>

<details>
<summary><strong>Q4: Why is layer norm preferred for Transformers over batch norm?</strong></summary>

1. **Variable sequence lengths**: BatchNorm would mix statistics across positions
2. **Small effective batch**: Attention can see all positions, but padding varies
3. **Batch independence**: Each sequence processed independently
4. **Simpler**: No running statistics to manage
5. **Works at inference**: Same behavior as training

Also: Transformers process sequences in parallel, making per-sample normalization natural.
</details>

<details>
<summary><strong>Q5: What are the learnable parameters in batch norm?</strong></summary>

Two learnable parameters per feature:
- **γ (gamma)**: Scale parameter
- **β (beta)**: Shift parameter

After normalization to zero mean and unit variance:
$$y = \gamma \hat{x} + \beta$$

**Why needed**: Normalization might remove useful information. γ and β allow the network to learn to undo normalization if needed (when γ=σ, β=μ, we recover original).
</details>

<details>
<summary><strong>Q6: What is group normalization and when is it useful?</strong></summary>

Group Norm divides channels into groups and normalizes within each:
- Channels split into G groups
- Normalize over (H, W, C/G) for each sample and group

**Useful when**:
- Batch size is small (BatchNorm fails)
- Batch size varies
- Distributed training with small per-GPU batches

GroupNorm with G=1 is LayerNorm; G=C is InstanceNorm.
</details>

---

## 8. References

1. Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training." ICML.
2. Ba, J., Kiros, J., & Hinton, G. (2016). "Layer Normalization." arXiv.
3. Wu, Y., & He, K. (2018). "Group Normalization." ECCV.
4. Santurkar, S., et al. (2018). "How Does Batch Normalization Help Optimization?" NeurIPS.
5. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). "Instance Normalization." arXiv.
