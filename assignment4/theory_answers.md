# Assignment 4: Parts 2 and 3

## Part 2: Building blocks of a diffusion model

### Question 1: Sinusoidal timestep embedding

A timestep is converted to a vector of sine and cosine values at geometrically spaced frequencies. This gives the denoising network information about the amount of corruption at that timestep.

**Convention:** The supplied Practical 3 `SinusoidalTimeEmbedding` uses an even embedding dimension $d=2h$, frequencies including both endpoints $1$ and $1/10000$, and concatenates all sine values before all cosine values. Using zero-based indexing and maximum period $M=10000$:

$$\omega_j=\exp\left(-\frac{j\ln M}{h-1}\right)=M^{-j/(h-1)},\quad j=0,\ldots,h-1.$$

The $i$-th dimension is

$$e_i(t)=\begin{cases}\sin(tM^{-i/(h-1)}),&0\le i<h,\\\cos(tM^{-(i-h)/(h-1)}),&h\le i<2h.\end{cases}$$

All angles are in radians. As in the course code, use $\max(h-1,1)$ in the denominator for the degenerate $h=1$ case. The raw vector can then pass through learned linear layers before being injected into the U-Net blocks.

A second common convention, used in the original transformer positional encoding, interleaves pairs:

$$PE_{2j}(t)=\sin(t/10000^{2j/d}),\qquad PE_{2j+1}(t)=\cos(t/10000^{2j/d}).$$

That convention changes both the spacing and ordering. The answers below explicitly distinguish the two; a maximum period alone does not uniquely specify the implementation.

### Question 2: Dimension 8 at t = 1

Using the supplied Practical 3 convention, $d=8$, so $h=4$ and $h-1=3$:

$$\omega=[10000^0,10000^{-1/3},10000^{-2/3},10000^{-1}]
=[1,0.0464158883361,0.00215443469003,0.0001].$$

Multiply by $t=1$, take sine and cosine, then concatenate:

| $j$ | $t\omega_j$ | $\sin(t\omega_j)$ | $\cos(t\omega_j)$ |
|---|---:|---:|---:|
| 0 | 1 | 0.841470984808 | 0.540302305868 |
| 1 | 0.0464158883361 | 0.046399223465 | 0.998922976041 |
| 2 | 0.00215443469003 | 0.002154433023 | 0.999997679206 |
| 3 | 0.0001 | 0.000100000000 | 0.999999995000 |

$$e(1)\approx[0.841470984808,0.046399223465,0.002154433023,0.000100000000,
0.540302305868,0.998922976041,0.999997679206,0.999999995000].$$

If the intended convention is the standard transformer formula instead, its angles are $[1,0.1,0.01,0.001]$ and its **interleaved** vector is

$$[0.841470985,0.540302306,0.099833417,0.995004165,
0.009999833,0.999950000,0.000999999833,0.999999500].$$

`theory_examples.py` reproduces the course-convention calculation in float64.

### Question 3: Connection to transformer positional encoding

Both represent a scalar index using sine and cosine functions at multiple frequencies, allowing a neural network to distinguish index values and learn relationships between them. Transformer positional encoding represents **where a token occurs in a sequence**, typically adding a different vector to each token embedding. A diffusion time embedding represents **the current diffusion/noise step**; the same timestep condition is supplied throughout the processing of one image, usually through projections into the U-Net's residual blocks. Spatial position and diffusion time therefore play different roles even though their encodings can use the same mathematical construction.

### Question 4: Bottleneck resolution

Each stride-2 block halves both spatial dimensions, assuming the convolution padding and kernel preserve exact halving:

$$64\times64\longrightarrow32\times32\longrightarrow16\times16\longrightarrow8\times8.$$

Equivalently, $64/2^3=8$. The bottleneck spatial resolution is **8 × 8**; its channel count depends on the architecture. For example, kernel 3, padding 1, stride 2 gives $\lfloor(n+2-3)/2\rfloor+1=n/2$ for these even input sizes.

### Question 5: U-Net output and training loss

In the noise-prediction formulation used here, the U-Net receives $x_t$ and a timestep embedding and returns $\epsilon_\theta(x_t,t)$, a tensor with the **same shape as the image** that predicts the noise added during forward diffusion.

First draw a clean training image $x_0$, a timestep $t$, and independent noise $\epsilon\sim\mathcal N(0,I)$. Define $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$. Construct

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.$$

The target is the known sampled $\epsilon$, and the loss is

$$L(\theta)=\mathbb E_{x_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right].$$

In the implementation, `F.mse_loss(predicted_noise, noise)` averages squared errors over the batch, channels, and pixels. Backpropagation updates only the U-Net parameters. During sampling, the noise estimate is used to estimate a clean image and take a reverse step. Some diffusion parameterizations predict $x_0$ or velocity instead, but this assignment implementation predicts noise.

## Part 3: Building blocks of an energy model

### Question 6: Basic gradient calculations

**a)** For $y=x^2+3x$, the power rule and linearity give

$$\frac{dy}{dx}=2x+3;\qquad\left.\frac{dy}{dx}\right|_{x=2}=2(2)+3=7.$$

The code prints `x.grad = tensor([7.])`.

**b)** If `x` has `requires_grad=False`, no input to this expression requires gradients. Thus `y` has no recorded autograd history. Calling `y.backward()` raises a `RuntimeError` stating that the tensor does not require grad and does not have a `grad_fn`; `x.grad` remains `None`. It does not silently compute a zero gradient.

**c)** `torch.tensor` defaults to `requires_grad=False`. A new tensor created without that flag will not have gradients tracked as a differentiable leaf. A subsequent operation can still be tracked with respect to other inputs that do require gradients. [PyTorch torch.tensor documentation](https://docs.pytorch.org/docs/2.15/generated/torch.tensor.html).

### Question 7: Introduce weights

The expression is $y=w_0x^2+w_1x$, with $x=2$ and $w=[1,3]$.

**a)** With the code as supplied, `w.grad = None` because `w` was created with the default `requires_grad=False`. PyTorch uses its values when differentiating with respect to $x$, giving $\partial y/\partial x=2w_0x+w_1=7$, but does not accumulate gradients for $w$.

**b)** Set `requires_grad=True` on the weights. When only the weight gradient is needed, $x$ can remain a constant:

```python
import torch
x = torch.tensor([2.0])
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
y.backward()
print('w.grad =', w.grad)  # tensor([4., 2.])
```

The two derivatives are

$$\frac{\partial y}{\partial w_0}=x^2=2^2=4,\qquad
\frac{\partial y}{\partial w_1}=x=2.$$

Thus $\nabla_w y=[4,2]$. If both `x` and `w` require gradients, the same call also populates `x.grad` with 7.

**c)** Omitting `requires_grad` still defaults to `False`, regardless of the variable's name or whether we intend it to represent weights. [PyTorch torch.tensor documentation](https://docs.pytorch.org/docs/2.15/generated/torch.tensor.html).

### Question 8: Breaking the graph

`y = x * 3` has a differentiable path back to $x$. `z = y.detach()` returns a tensor that shares storage with `y` but is disconnected from that graph and does not require gradients. Consequently `w = z * 2` has no autograd history, and `w.backward()` raises an error. [PyTorch detach documentation](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.detach.html).

Use a differentiable copy instead:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.clone()
w = z * 2
w.backward()
print(x.grad)  # tensor([6.])
```

The chain rule gives $dw/dx=(dw/dz)(dz/dy)(dy/dx)=2\cdot1\cdot3=6$. Simply using `z = y` also works. Calling `z.requires_grad_(True)` **after detaching** starts a new leaf; it does not restore the path back to $x$.

### Question 9: Gradient accumulation

The first backward pass gives $dy_1/dx=2$, so `x.grad` is `tensor([2.])`. The second expression has derivative $dy_2/dx=3$, which PyTorch **adds** to the existing gradient:

$$x.grad=2+3=5.$$

Therefore the second print is `tensor([5.])`. Gradients accumulate in leaf tensors by design. These are separate forward graphs sharing a leaf, so `retain_graph=True` is unnecessary. [PyTorch backward documentation](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.backward.html).

For independent calculations, reset before the next backward pass:

```python
x.grad = None                 # alternatively: x.grad.zero_()
y2 = x * 3
# Build a new expression if the previous y2 graph has already been used.
y2.backward()
print(x.grad)                 # tensor([3.])
```

In a training loop use `optimizer.zero_grad(set_to_none=True)` before each independent optimization step. If accumulating microbatch gradients deliberately, clear them only at the intended step boundary.

## Connection to the implementation

For EBM sampling, hold $\theta$ fixed and update pixels with $x\leftarrow x-\eta\nabla_xE_\theta(x)$ plus sampling noise. For EBM training, detach sampled negatives and minimize $\mathbb E[E_\theta(x_{real})]-\mathbb E[E_\theta(x_{fake})]$ plus an energy regularizer with respect to $\theta$. This lowers real-image energy and raises negative-image energy. Diffusion training similarly updates network weights, while its generation loop needs no gradients.
