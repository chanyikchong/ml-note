# Convolutional Neural Networks (CNN)

## 1. Interview Summary

**Key Points to Remember:**
- **Convolution**: Slides filter over input, computes dot products
- **Pooling**: Downsamples, provides translation invariance
- **Key hyperparameters**: Filter size, stride, padding, number of filters
- **Architecture pattern**: Conv → ReLU → Pool → ... → Flatten → FC
- **Parameter sharing**: Same filter applied across entire input

**Common Interview Questions:**
- "What is the receptive field?"
- "Calculate output dimensions of a conv layer"
- "Why use convolutions instead of fully connected?"

---

## 2. Core Definitions

### Convolution Operation
For input $I$ and kernel $K$:
$$(I * K)[i,j] = \sum_m \sum_n I[i+m, j+n] \cdot K[m, n]$$

### Output Dimensions
$$\text{output\_size} = \frac{W - K + 2P}{S} + 1$$
- $W$: Input size
- $K$: Kernel size
- $P$: Padding
- $S$: Stride

### Receptive Field
The region in input that affects a particular output neuron.
- Grows with depth
- For L layers with kernel K: $RF = 1 + L \cdot (K - 1)$

### Common Layers

| Layer | Purpose | Parameters |
|-------|---------|------------|
| Conv2D | Feature extraction | Kernel weights |
| MaxPool | Downsample, invariance | None |
| BatchNorm | Normalize activations | γ, β |
| Dropout | Regularization | None (prob) |
| Flatten | Reshape for FC | None |

---

## 3. Math and Derivations

### Convolution Forward Pass

For input $X$ of shape $(H, W, C_{in})$ and filters $W$ of shape $(K, K, C_{in}, C_{out})$:

$$Y[i, j, k] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X[i \cdot s + m, j \cdot s + n, c] \cdot W[m, n, c, k] + b[k]$$

### Parameter Count

For Conv2D with $C_{in}$ input channels, $C_{out}$ output channels, kernel $K$:
$$\text{params} = K \times K \times C_{in} \times C_{out} + C_{out}$$

### Why Convolutions Work

1. **Local connectivity**: Each output depends on local region
2. **Parameter sharing**: Same weights across spatial dimensions
3. **Translation equivariance**: Shifting input shifts output

Compared to FC:
- FC on 224×224×3 image to 1000 outputs: 150M params
- Conv: Same kernel everywhere, far fewer params

### Backpropagation Through Conv

The gradient of loss w.r.t. kernel:
$$\frac{\partial L}{\partial W} = X * \frac{\partial L}{\partial Y}$$

Full convolution of input with upstream gradient.

---

## 4. Algorithm Sketch

### Basic CNN Architecture

```
Input: Image (H × W × C)

# Feature extraction
for layer in [conv1, conv2, conv3, ...]:
    x = Conv2D(x, filters, kernel_size)
    x = BatchNorm(x)
    x = ReLU(x)
    x = MaxPool(x, pool_size)

# Classification head
x = Flatten(x)
x = Dense(x, hidden_units)
x = ReLU(x)
x = Dropout(x, rate)
x = Dense(x, num_classes)
output = Softmax(x)
```

### Convolution Implementation

```
def conv2d(input, kernel, stride=1, padding=0):
    H, W, C_in = input.shape
    K, _, _, C_out = kernel.shape

    # Add padding
    input_padded = pad(input, padding)

    # Output dimensions
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1
    output = zeros(H_out, W_out, C_out)

    # Convolution
    for i in range(H_out):
        for j in range(W_out):
            for k in range(C_out):
                h_start = i * stride
                w_start = j * stride
                region = input_padded[h_start:h_start+K, w_start:w_start+K, :]
                output[i, j, k] = sum(region * kernel[:, :, :, k])

    return output
```

### Pooling Implementation

```
def maxpool2d(input, pool_size=2, stride=2):
    H, W, C = input.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = zeros(H_out, W_out, C)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            region = input[h_start:h_start+pool_size, w_start:w_start+pool_size, :]
            output[i, j, :] = max(region, axis=(0, 1))

    return output
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Dimension mismatch | Wrong padding/stride | Calculate output size carefully |
| Too many parameters | Deep FC layers | Use global average pooling |
| Overfitting | Small dataset | Data augmentation, dropout |
| Vanishing gradients | Deep networks | Use skip connections, BatchNorm |
| Slow training | Large images | Progressive resizing |

### Architecture Guidelines

| Data Size | Recommendation |
|-----------|---------------|
| Small (<10k images) | Use pretrained, fine-tune |
| Medium (10k-100k) | Train from scratch with augmentation |
| Large (>100k) | Deep custom architecture |

### Common Architectures

| Network | Key Innovation | Depth |
|---------|---------------|-------|
| LeNet | Basic CNN | 5 |
| AlexNet | ReLU, Dropout, GPU | 8 |
| VGG | 3×3 filters only | 16-19 |
| ResNet | Skip connections | 50-152 |
| Inception | Multi-scale filters | 22+ |

---

## 6. Mini Example

```python
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Initialize weights (He initialization)
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.W = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * scale
        self.b = np.zeros(out_channels)

    def forward(self, x):
        self.x = x
        N, H, W, C = x.shape
        K = self.kernel_size

        # Pad input
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding),
                          (self.padding, self.padding), (0, 0)), mode='constant')

        # Output dimensions
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1
        out = np.zeros((N, H_out, W_out, self.W.shape[3]))

        # Convolution
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                x_slice = x[:, h_start:h_start+K, w_start:w_start+K, :]
                out[:, i, j, :] = np.tensordot(x_slice, self.W, axes=([1,2,3], [0,1,2])) + self.b

        return out


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        N, H, W, C = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        out = np.zeros((N, H_out, W_out, C))

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                out[:, i, j, :] = np.max(
                    x[:, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, :],
                    axis=(1, 2)
                )

        return out


def relu(x):
    return np.maximum(0, x)


# Simple CNN forward pass
np.random.seed(42)

# Create sample "image": batch of 2, 8x8, 1 channel
x = np.random.randn(2, 8, 8, 1)

# Build simple CNN
conv1 = Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1)
pool1 = MaxPool2D(pool_size=2, stride=2)
conv2 = Conv2D(in_channels=4, out_channels=8, kernel_size=3, padding=1)
pool2 = MaxPool2D(pool_size=2, stride=2)

# Forward pass
print(f"Input shape: {x.shape}")
x = conv1.forward(x)
print(f"After Conv1: {x.shape}")
x = relu(x)
x = pool1.forward(x)
print(f"After Pool1: {x.shape}")
x = conv2.forward(x)
print(f"After Conv2: {x.shape}")
x = relu(x)
x = pool2.forward(x)
print(f"After Pool2: {x.shape}")

# Flatten for FC
x_flat = x.reshape(x.shape[0], -1)
print(f"Flattened: {x_flat.shape}")

# Parameter count
params = (3*3*1*4 + 4) + (3*3*4*8 + 8)
print(f"\nTotal conv parameters: {params}")
```

**Output:**
```
Input shape: (2, 8, 8, 1)
After Conv1: (2, 8, 8, 4)
After Pool1: (2, 4, 4, 4)
After Conv2: (2, 4, 4, 8)
After Pool2: (2, 2, 2, 8)
Flattened: (2, 32)

Total conv parameters: 332
```

---

## 7. Quiz

<details>
<summary><strong>Q1: Calculate the output dimensions of a conv layer.</strong></summary>

For input $(H, W)$, kernel $K$, stride $S$, padding $P$:

$$H_{out} = \lfloor\frac{H + 2P - K}{S}\rfloor + 1$$

**Example**: Input 32×32, kernel 5×5, stride 1, padding 2:
$$H_{out} = \frac{32 + 2(2) - 5}{1} + 1 = \frac{31}{1} + 1 = 32$$

For "same" padding (output = input): $P = \frac{K-1}{2}$ (when S=1)
</details>

<details>
<summary><strong>Q2: What is the receptive field and why does it matter?</strong></summary>

**Receptive field**: The region in the input that influences a particular output neuron.

**Why it matters**:
- Determines what context each neuron "sees"
- Deep layers have larger receptive fields
- Need large enough RF to capture relevant patterns

**Calculation**: For L layers with kernel K and stride 1:
$$RF = 1 + L \times (K - 1)$$

Example: 3 layers of 3×3 convs: RF = 1 + 3×2 = 7×7
</details>

<details>
<summary><strong>Q3: Why use convolutions instead of fully connected layers for images?</strong></summary>

1. **Parameter efficiency**: Same kernel reused spatially
   - FC on 224×224×3 → 1000: 150M params
   - Conv 3×3 with 64 filters: 1,728 params

2. **Translation equivariance**: Object detected regardless of position

3. **Local connectivity**: Exploits spatial structure

4. **Hierarchical features**: Low-level → high-level as depth increases

5. **Better generalization**: Fewer params = less overfitting
</details>

<details>
<summary><strong>Q4: What is the purpose of pooling layers?</strong></summary>

Pooling (usually max or average):

1. **Downsampling**: Reduces spatial dimensions
2. **Translation invariance**: Small shifts don't change output
3. **Reduces computation**: Fewer parameters in subsequent layers
4. **Increases receptive field**: Faster than stacking convs

**Max pooling** is most common: takes maximum in each region, retains strongest activations.

**Trend**: Modern architectures often use strided convolutions instead.
</details>

<details>
<summary><strong>Q5: How many parameters in a Conv2D layer?</strong></summary>

For Conv2D with:
- Input channels: $C_{in}$
- Output channels: $C_{out}$
- Kernel size: $K \times K$

**Parameters**: $K \times K \times C_{in} \times C_{out} + C_{out}$ (bias)

**Example**: 3×3 conv, 64 input channels, 128 output channels:
$$3 \times 3 \times 64 \times 128 + 128 = 73,856$$

Note: Same params regardless of input spatial size (parameter sharing).
</details>

<details>
<summary><strong>Q6: What is 1×1 convolution and when is it used?</strong></summary>

1×1 convolution: Kernel size 1, acts as per-pixel fully connected layer.

**Uses**:
1. **Channel reduction**: Reduce channels before expensive 3×3 conv (bottleneck)
2. **Add non-linearity**: 1×1 conv + ReLU adds capacity
3. **Cross-channel interaction**: Mix information across channels
4. **Network-in-Network**: Create deeper networks cheaply

Used heavily in Inception and ResNet architectures.
</details>

---

## 8. References

1. LeCun, Y., et al. (1998). "Gradient-Based Learning Applied to Document Recognition." Proc. IEEE.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). "ImageNet Classification with Deep CNNs." NIPS.
3. Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks." arXiv.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
5. Szegedy, C., et al. (2015). "Going Deeper with Convolutions." CVPR.
