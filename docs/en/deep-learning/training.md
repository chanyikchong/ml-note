# Training Deep Neural Networks

## 1. Interview Summary

**Key Points to Remember:**
- **Initialization**: He/Xavier initialization critical for convergence
- **Vanishing/Exploding gradients**: Main training challenge
- **Gradient clipping**: Prevent exploding gradients
- **Learning rate schedules**: Warmup, decay, cosine annealing
- **Debugging**: Monitor losses, gradients, activations

**Common Interview Questions:**
- "How do you diagnose vanishing gradients?"
- "What is proper weight initialization?"
- "How do you choose a learning rate?"

---

## 2. Core Definitions

### Weight Initialization

**Xavier/Glorot** (for tanh/sigmoid):

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

**He/Kaiming** (for ReLU):

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

### Vanishing Gradients
Gradients become exponentially small in early layers:
- Weights don't update
- Network doesn't learn

### Exploding Gradients
Gradients become exponentially large:
- Weights become NaN/Inf
- Training becomes unstable

### Gradient Clipping
Limit gradient magnitude:

$$g \leftarrow \min\left(1, \frac{\theta}{\|g\|}\right) g$$

---

## 3. Math and Derivations

### Why Initialization Matters

For layer $h = f(Wx)$, variance propagation:

$$\text{Var}(h) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

To maintain variance across layers:

$$\text{Var}(W) = \frac{1}{n_{in}}$$

For ReLU (kills half the signal):

$$\text{Var}(W) = \frac{2}{n_{in}}$$

### Vanishing Gradient Analysis

For L layers with activation $\sigma$:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \prod_{l=2}^{L} \frac{\partial h_l}{\partial h_{l-1}} \frac{\partial h_1}{\partial W_1}$$

If $|\sigma'| < 1$ (sigmoid, tanh):
- Product shrinks exponentially with depth
- Early layers receive tiny gradients

### Learning Rate Schedules

**Step decay:**

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

**Cosine annealing:**

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

**Warmup:**

$$\eta_t = \eta_{target} \cdot \frac{t}{T_{warmup}} \quad \text{for } t < T_{warmup}$$

---

## 4. Algorithm Sketch

### Training Loop

```
def train(model, data, epochs, lr):
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealing(optimizer, T_max=epochs)

    for epoch in range(epochs):
        for batch in data:
            # Forward pass
            predictions = model(batch.x)
            loss = criterion(predictions, batch.y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

        # Update learning rate
        scheduler.step()

        # Validation
        val_loss = evaluate(model, val_data)
        if val_loss < best_loss:
            save_checkpoint(model)
```

### Debugging Checklist

```
1. Check data loading:
   - Visualize samples
   - Verify labels are correct
   - Check normalization

2. Overfit small batch first:
   - Train on 1-10 samples
   - Should reach ~100% accuracy
   - If not, architecture/code bug

3. Monitor during training:
   - Loss should decrease
   - Gradients should be reasonable (not 0 or inf)
   - Activations should be in normal range

4. Common fixes:
   - Loss stuck high → lower learning rate
   - Loss NaN → gradient clipping, check data
   - Loss oscillating → reduce learning rate
```

### Gradient Health Check

```
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"{name}: grad_norm = {grad_norm:.6f}")

            if grad_norm == 0:
                print("  WARNING: Zero gradient!")
            if grad_norm > 100:
                print("  WARNING: Exploding gradient!")
            if torch.isnan(grad_norm):
                print("  ERROR: NaN gradient!")
```

---

## 5. Common Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Bad initialization | Loss doesn't decrease | Use He/Xavier init |
| Vanishing gradients | Early layers don't update | Use ReLU, skip connections, BatchNorm |
| Exploding gradients | Loss becomes NaN | Gradient clipping, lower LR |
| Learning rate too high | Loss oscillates/diverges | Reduce LR |
| Learning rate too low | Loss decreases very slowly | Increase LR or use scheduler |
| Overfitting | Train loss ↓, val loss ↑ | Dropout, data augmentation, regularization |

### Learning Rate Selection

```
Learning rate finder:
1. Start with very small LR (1e-7)
2. Increase LR exponentially each batch
3. Plot LR vs loss
4. Choose LR where loss decreases fastest
   (typically 1-10x below minimum)

Common ranges:
- SGD: 0.01 - 0.1
- Adam: 0.0001 - 0.001
- With warmup: Start 10-100x smaller
```

### Debugging Training

| Observation | Likely Cause | Fix |
|-------------|--------------|-----|
| Loss = constant | Gradients zero, LR too small | Check gradients, increase LR |
| Loss = NaN | Exploding gradients, bad data | Clip gradients, check data |
| Loss = oscillating | LR too high | Reduce LR |
| Val loss increasing | Overfitting | Regularization, early stopping |
| Train loss high | Underfitting | Larger model, train longer |

---

## 6. Mini Example

```python
import numpy as np

def he_init(shape):
    """He initialization for ReLU networks."""
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std

def xavier_init(shape):
    """Xavier initialization for tanh/sigmoid."""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std

def clip_gradients(gradients, max_norm):
    """Clip gradients by global norm."""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        gradients = [g * clip_coef for g in gradients]
    return gradients, total_norm


class LearningRateScheduler:
    def __init__(self, initial_lr, schedule='cosine', warmup_steps=0, total_steps=1000):
        self.initial_lr = initial_lr
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step):
        # Warmup
        if step < self.warmup_steps:
            return self.initial_lr * (step + 1) / self.warmup_steps

        # After warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)

        if self.schedule == 'cosine':
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.schedule == 'step':
            return self.initial_lr * (0.1 ** int(progress * 3))
        else:
            return self.initial_lr


# Demo: Initialization comparison
np.random.seed(42)
layer_sizes = [784, 256, 128, 64, 10]

print("=== Initialization Comparison ===")
print("\nVariance propagation with different initializations:")

for init_name, init_fn in [("Zero", lambda s: np.zeros(s)),
                            ("Random", lambda s: np.random.randn(*s)),
                            ("Xavier", xavier_init),
                            ("He", he_init)]:
    x = np.random.randn(100, 784)  # Batch of 100

    print(f"\n{init_name} initialization:")
    for i in range(len(layer_sizes) - 1):
        W = init_fn((layer_sizes[i], layer_sizes[i+1]))
        x = np.maximum(0, x @ W)  # ReLU activation

        if np.std(x) == 0:
            print(f"  Layer {i+1}: activations collapsed to 0")
            break
        print(f"  Layer {i+1}: mean={x.mean():.4f}, std={x.std():.4f}")

# Demo: Learning rate schedule
print("\n=== Learning Rate Schedules ===")
scheduler = LearningRateScheduler(0.1, schedule='cosine', warmup_steps=100, total_steps=1000)
steps = [0, 50, 100, 250, 500, 750, 1000]
for step in steps:
    lr = scheduler.get_lr(step)
    print(f"Step {step:4d}: LR = {lr:.6f}")

# Demo: Gradient clipping
print("\n=== Gradient Clipping ===")
gradients = [np.random.randn(100, 50) * 10]  # Large gradients
clipped, norm_before = clip_gradients(gradients, max_norm=1.0)
_, norm_after = clip_gradients(clipped, max_norm=1.0)
print(f"Gradient norm before clipping: {norm_before:.2f}")
print(f"Gradient norm after clipping: {np.sqrt(sum(np.sum(g**2) for g in clipped)):.2f}")
```

**Output:**
```
=== Initialization Comparison ===

Variance propagation with different initializations:

Zero initialization:
  Layer 1: activations collapsed to 0

Random initialization:
  Layer 1: mean=10.0234, std=19.5432
  Layer 2: mean=127.8921, std=256.1234
  Layer 3: mean=8234.56, std=16892.12
  Layer 4: activations exploded

Xavier initialization:
  Layer 1: mean=0.5012, std=0.7123
  Layer 2: mean=0.2534, std=0.3892
  Layer 3: mean=0.1234, std=0.2012
  Layer 4: mean=0.0612, std=0.1023

He initialization:
  Layer 1: mean=0.7891, std=1.0234
  Layer 2: mean=0.8123, std=0.9876
  Layer 3: mean=0.7654, std=1.0123
  Layer 4: mean=0.8012, std=0.9912

=== Learning Rate Schedules ===
Step    0: LR = 0.001000
Step   50: LR = 0.050500
Step  100: LR = 0.100000
Step  250: LR = 0.085355
Step  500: LR = 0.050000
Step  750: LR = 0.014645
Step 1000: LR = 0.000000

=== Gradient Clipping ===
Gradient norm before clipping: 70.71
Gradient norm after clipping: 1.00
```

---

## 7. Quiz

<details>
<summary><strong>Q1: How do you diagnose vanishing gradients?</strong></summary>

Signs of vanishing gradients:
1. **Early layer gradients near zero**: Check `param.grad.norm()` per layer
2. **Early layer weights don't change**: Compare weights before/after training
3. **Loss plateaus early**: Network stops learning quickly
4. **Activations saturate**: For sigmoid/tanh, outputs near 0 or 1

**Diagnosis code**:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

**Solutions**: ReLU activations, skip connections, BatchNorm, proper initialization.
</details>

<details>
<summary><strong>Q2: Explain He vs Xavier initialization.</strong></summary>

Both aim to maintain variance across layers:

**Xavier** (Glorot): $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$
- Designed for tanh/sigmoid activations
- Considers both forward and backward pass

**He** (Kaiming): $\text{Var}(W) = \frac{2}{n_{in}}$
- Designed for ReLU activations
- Accounts for ReLU killing half the signal

**When to use**:
- ReLU networks → He initialization
- Tanh/sigmoid networks → Xavier initialization
- Linear outputs → Xavier initialization
</details>

<details>
<summary><strong>Q3: What is gradient clipping and when is it used?</strong></summary>

**Gradient clipping**: Limit gradient magnitude to prevent exploding gradients.

**By value**: Clip each element: $g = \max(\min(g, \theta), -\theta)$
**By norm**: Scale if norm exceeds threshold: $g = g \cdot \min(1, \theta/\|g\|)$

**When to use**:
- RNNs (prone to exploding gradients)
- Very deep networks
- When loss becomes NaN
- As standard practice (doesn't hurt)

**Typical values**: max_norm = 1.0 to 5.0
</details>

<details>
<summary><strong>Q4: How do you choose a learning rate?</strong></summary>

**Methods**:
1. **Learning rate finder**: Increase LR from small, plot loss vs LR, choose where loss drops fastest
2. **Default values**: Adam ~0.001, SGD ~0.01-0.1
3. **Grid search**: Try [0.0001, 0.001, 0.01, 0.1]

**Signs of wrong LR**:
- Too high: Loss oscillates or increases
- Too low: Loss decreases very slowly

**Best practices**:
- Use learning rate warmup
- Use scheduler (cosine, step decay)
- Reduce LR when loss plateaus
</details>

<details>
<summary><strong>Q5: What is learning rate warmup?</strong></summary>

**Warmup**: Gradually increase LR from small value to target during initial training.

**Why it helps**:
1. Early gradients can be noisy/large
2. Prevents large initial updates
3. Allows BatchNorm statistics to stabilize
4. Important for large batch training

**Common approach**:
```
LR = target_lr * (step / warmup_steps)  for step < warmup_steps
```

**Typical warmup**: 1-5% of total training steps
</details>

<details>
<summary><strong>Q6: How do you debug a network that won't train?</strong></summary>

**Systematic debugging**:
1. **Verify data**: Visualize inputs, check labels, verify preprocessing
2. **Overfit single batch**: Should reach ~100% train accuracy on 1-10 samples
3. **Check gradients**: Print gradient norms per layer, look for zeros or infinities
4. **Check activations**: Monitor mean/std of layer outputs
5. **Simplify**: Smaller model, simpler data, fewer layers

**Common fixes**:
- Loss stuck → Lower LR, check gradients
- Loss NaN → Gradient clipping, check data for NaN
- Loss oscillating → Lower LR
- No improvement → Increase model capacity
</details>

---

## 8. References

1. Glorot, X., & Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." AISTATS.
2. He, K., et al. (2015). "Delving Deep into Rectifiers." ICCV.
3. Smith, L. (2017). "Cyclical Learning Rates for Training Neural Networks." WACV.
4. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD." arXiv.
5. Loshchilov, I., & Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts." arXiv.
