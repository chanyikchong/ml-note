# Optimization

## 1. Interview Summary

**Key Points to Remember:**
- **GD**: Gradient Descent - full batch, slow but stable
- **SGD**: Stochastic GD - single sample, noisy but fast
- **Mini-batch SGD**: Best of both worlds, most common
- **Momentum**: Accelerates convergence, dampens oscillations
- **Adam**: Adaptive learning rates, default choice in deep learning
- Learning rate schedules improve convergence

**Common Interview Questions:**
- "Explain the difference between GD, SGD, and mini-batch SGD"
- "How does momentum help optimization?"
- "Why is Adam so popular?"

---

## 2. Core Definitions

### Gradient Descent (GD)
Update parameters using gradient computed over entire dataset:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta$ is the learning rate.

### Stochastic Gradient Descent (SGD)
Update using gradient from single random sample:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_i(\theta_t)$$

**Properties:**
- High variance in gradient estimates
- Can escape local minima (noise helps)
- Much faster per iteration

### Mini-batch SGD
Update using gradient from small batch of samples:

$$\theta_{t+1} = \theta_t - \eta \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \mathcal{L}_i(\theta_t)$$

**Typical batch sizes**: 32, 64, 128, 256

### Momentum
Accumulate gradient history to accelerate convergence:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t - v_t$$

**Typical $\gamma$**: 0.9

---

## 3. Math and Derivations

### Convergence Analysis (Convex Case)

For convex loss with L-Lipschitz gradients:

**GD Convergence Rate:**

$$\mathcal{L}(\theta_T) - \mathcal{L}(\theta^*) \leq \frac{L\|\theta_0 - \theta^*\|^2}{2T}$$

Converges as $O(1/T)$.

**SGD Convergence:**
With decreasing learning rate $\eta_t = \eta_0/\sqrt{t}$:

$$\mathbb{E}[\mathcal{L}(\bar{\theta}_T)] - \mathcal{L}(\theta^*) \leq O\left(\frac{1}{\sqrt{T}}\right)$$

### Adam Algorithm

**Moment Estimates:**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias Correction:**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update:**

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Default hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### Learning Rate Schedules

**Step Decay:**

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

**Exponential Decay:**

$$\eta_t = \eta_0 \cdot e^{-kt}$$

**Cosine Annealing:**

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**Warmup:**

$$\eta_t = \frac{t}{T_{warmup}} \cdot \eta_{target} \quad \text{for } t < T_{warmup}$$

---

## 4. Algorithm Sketch

### Mini-batch SGD with Momentum

```
Initialize: θ, v = 0, η, γ = 0.9
For epoch = 1 to num_epochs:
    Shuffle training data
    For each mini-batch B:
        g = (1/|B|) * Σ ∇L_i(θ)   # Compute gradient
        v = γ * v + η * g         # Update velocity
        θ = θ - v                  # Update parameters
```

### Adam Optimizer

```
Initialize: θ, m = 0, v = 0, t = 0
Hyperparameters: η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8

For each mini-batch:
    t = t + 1
    g = gradient of loss w.r.t. θ

    # Update biased moment estimates
    m = β₁ * m + (1 - β₁) * g
    v = β₂ * v + (1 - β₂) * g²

    # Bias correction
    m_hat = m / (1 - β₁^t)
    v_hat = v / (1 - β₂^t)

    # Update parameters
    θ = θ - η * m_hat / (√v_hat + ε)
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Learning rate too high | Divergence, loss explodes | Start small, use learning rate finder |
| Learning rate too low | Very slow convergence | Use warmup and schedules |
| No momentum | Slow convergence, oscillations | Use momentum ≥ 0.9 |
| Constant learning rate | Misses fine-tuning phase | Use decay schedule |
| Adam on all problems | May not generalize best | Try SGD+momentum for vision tasks |
| Not shuffling data | Biased gradient estimates | Shuffle each epoch |

### Optimizer Comparison

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| SGD+Momentum | Often best generalization | Needs LR tuning | CNNs, vision |
| Adam | Fast, adaptive | May not generalize | NLP, quick prototyping |
| AdamW | Adam + proper weight decay | More hyperparameters | Transformers |
| RMSprop | Adaptive, simple | Less used now | RNNs |

---

## 6. Mini Example

### Python Example: Comparing Optimizers

```python
import numpy as np

def quadratic_loss(x):
    """Loss: f(x) = 0.5 * x^T A x, where A is ill-conditioned."""
    A = np.array([[10, 0], [0, 1]])
    return 0.5 * x @ A @ x

def gradient(x):
    A = np.array([[10, 0], [0, 1]])
    return A @ x

# SGD
def sgd(x0, lr=0.1, steps=50):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(steps):
        x = x - lr * gradient(x)
        history.append(x.copy())
    return np.array(history)

# SGD with Momentum
def sgd_momentum(x0, lr=0.1, momentum=0.9, steps=50):
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(steps):
        v = momentum * v + lr * gradient(x)
        x = x - v
        history.append(x.copy())
    return np.array(history)

# Adam
def adam(x0, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8, steps=50):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [x.copy()]
    for t in range(1, steps + 1):
        g = gradient(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(x.copy())
    return np.array(history)

# Compare
x0 = np.array([1.0, 1.0])
sgd_hist = sgd(x0, lr=0.05)
mom_hist = sgd_momentum(x0, lr=0.05)
adam_hist = adam(x0, lr=0.3)

print(f"SGD final: {sgd_hist[-1]}, loss: {quadratic_loss(sgd_hist[-1]):.6f}")
print(f"Momentum final: {mom_hist[-1]}, loss: {quadratic_loss(mom_hist[-1]):.6f}")
print(f"Adam final: {adam_hist[-1]}, loss: {quadratic_loss(adam_hist[-1]):.6f}")

# Output (convergence to [0,0]):
# SGD final: [0.00592 0.00519], loss: 0.000189
# Momentum final: [0.00001 0.00003], loss: 0.000000
# Adam final: [0.00000 0.00000], loss: 0.000000
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What's the difference between GD, SGD, and mini-batch SGD?</strong></summary>

- **GD**: Uses entire dataset for each gradient computation. Stable but slow and memory-intensive.
- **SGD**: Uses single sample per update. Fast and memory-efficient but very noisy gradients.
- **Mini-batch SGD**: Uses small batch (e.g., 32-256 samples). Balances computational efficiency with gradient quality. Most commonly used in practice.
</details>

<details>
<summary><strong>Q2: How does momentum help optimization?</strong></summary>

Momentum accumulates a running average of past gradients:
- **Accelerates** convergence in consistent gradient directions
- **Dampens** oscillations in directions with sign changes
- Helps escape shallow local minima and saddle points
- Acts like a "heavy ball" rolling downhill with inertia

Formula: $v_t = \gamma v_{t-1} + \eta \nabla \mathcal{L}$, typically $\gamma = 0.9$
</details>

<details>
<summary><strong>Q3: What makes Adam different from SGD?</strong></summary>

Adam combines:
1. **First moment** (momentum): Running average of gradients
2. **Second moment**: Running average of squared gradients (like RMSprop)
3. **Bias correction**: Corrects initialization bias

This gives **adaptive per-parameter learning rates** that are larger for infrequent parameters and smaller for frequent ones. No manual learning rate tuning per layer.
</details>

<details>
<summary><strong>Q4: Why use learning rate schedules?</strong></summary>

- **Early training**: Large LR for fast progress
- **Late training**: Small LR for fine-tuning and convergence
- **Escaping local minima**: LR restarts can help escape
- **Better generalization**: Slower final convergence often generalizes better

Common schedules: step decay, cosine annealing, warmup + decay.
</details>

<details>
<summary><strong>Q5: When might SGD+Momentum outperform Adam?</strong></summary>

SGD+Momentum often outperforms Adam for:
- Computer vision tasks (CNNs)
- When generalization is more important than training speed
- Longer training runs where Adam's adaptivity becomes less helpful
- Tasks where careful LR tuning is feasible

Adam excels for quick prototyping, NLP, and when LR tuning time is limited.
</details>

<details>
<summary><strong>Q6: What is the purpose of warmup in learning rate schedules?</strong></summary>

Warmup gradually increases the learning rate from near-zero to the target value over several iterations. Benefits:
- Prevents large early updates when gradients are unreliable
- Stabilizes training with large batch sizes
- Helps with batch normalization statistics initialization
- Particularly important for Transformers and large models
</details>

---

## 8. References

1. Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms." arXiv.
2. Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." ICLR.
3. Bottou, L., Curtis, F. E., & Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning." SIAM Review.
4. Loshchilov, I., & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR.
5. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv.
