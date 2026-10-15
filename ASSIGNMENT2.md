# Assignment 2: CNN implementation and worked theory answers

## Part 1: Specified architecture

Implementation: `app/cnn_model.py`, class `AssignmentCNN`.

The assignment writes images as height × width × channels. PyTorch uses batch × channels × height × width, so a batch enters as `(N, 3, 64, 64)`.

| Operation | Output per image (C × H × W) | Trainable parameters |
| --- | --- | ---: |
| Input | 3 × 64 × 64 | 0 |
| Conv2d(3, 16, 3, stride=1, padding=1) | 16 × 64 × 64 | (3×3×3 + 1)×16 = 448 |
| ReLU | 16 × 64 × 64 | 0 |
| MaxPool2d(2, stride=2) | 16 × 32 × 32 | 0 |
| Conv2d(16, 32, 3, stride=1, padding=1) | 32 × 32 × 32 | (3×3×16 + 1)×32 = 4,640 |
| ReLU | 32 × 32 × 32 | 0 |
| MaxPool2d(2, stride=2) | 32 × 16 × 16 | 0 |
| Flatten | 8,192 | 0 |
| Linear(8192, 100) | 100 | 8,192×100 + 100 = 819,300 |
| ReLU | 100 | 0 |
| Linear(100, 10) | 10 | 100×10 + 10 = 1,010 |
| **Total** | **10 logits** | **825,398** |

The final layer produces logits. CrossEntropyLoss applies the required log-softmax internally, so the training model has no final softmax. The API applies softmax to obtain probabilities and returns the class with the largest probability.

## Part 2: Training and API

CIFAR-10 has RGB images of 32×32. Resize each to 64×64 to satisfy the specified input dimensions rather than changing the fully connected layer. Both training and inference use bilinear resize followed by ToTensor, scaling pixel values to [0, 1]. Inference converts decoded images to RGB first.

`training/train_cnn.py` uses a seeded 45,000/5,000 split of the original training set for training and validation. It trains with Adam (learning rate 0.0005), cross-entropy, and batches of 128 by default. It saves the epoch with the highest validation accuracy and evaluates that checkpoint once on the separate 10,000-image test set. Epoch count is configurable; the default is 10. No target accuracy is specified in the assignment. Random seeds improve repeatability but do not guarantee identical results across hardware and library versions.

`POST /classify` accepts a multipart field called `image` containing a JPEG or PNG. The response includes class ID, class name, winning probability, and probabilities for all ten CIFAR-10 classes. The classes, in index order, are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Softmax confidence is not a calibrated probability of correctness, and the classifier always chooses one of these classes, even for unrelated images.

See README.md for training, serving, and request commands. The generated `models/cifar10_cnn.metrics.json` records actual results when training finishes. The checkpoint is a local generated artifact excluded from Git; copy it to the serving machine before starting the API. No checkpoint is downloaded or trained during API startup. Without it, the text routes remain available and classification returns HTTP 503.

### Measured training result

The completed 10-epoch run used seed 42 on Apple MPS. The selected checkpoint is epoch 9, with validation accuracy **60.58%** (5,000 images). Epoch 10 tied that accuracy, so the earlier checkpoint was retained. Test accuracy is **61.45%** (6,145 of 10,000 images), with mean test cross-entropy **1.081308**. Training accuracy at the selected epoch was **66.72%**. These results demonstrate a working baseline; they are not a claim of state-of-the-art performance.

Full epoch history and configuration are saved in `models/cifar10_cnn.metrics.json`.

## Part 3: Arithmetic of CNNs

For a convolution with dilation 1, the output size along either spatial axis is:

    output = floor((input + 2×padding − kernel_size) / stride) + 1

The same spatial formula applies to the pooling layers in these questions, using the default floor rule. A convolution's output channel count equals its number of filters.

### Question 1

Input: 32×32×3. Eight 5×5 filters, stride 1, padding 0.

    H_out = floor((32 + 2×0 − 5) / 1) + 1
          = floor(27) + 1
          = 28
    W_out = 28
    C_out = number of filters = 8

**Output: 28×28×8** (height × width × channels). Each filter spans all three input channels; the input channel count does not multiply the number of output channels.

### Question 2

Keep the filters and stride from Question 1, but use “same” padding. For an odd 5×5 kernel at stride 1, pad by (5−1)/2 = 2 pixels on each side.

    H_out = floor((32 + 2×2 − 5) / 1) + 1
          = floor(31) + 1
          = 32
    W_out = 32
    C_out = 8

**Output: 32×32×8**, compared with 28×28×8 without padding. “Same” preserves the spatial dimensions here; it does not preserve the input channel count.

### Question 3

Input spatial size 64×64, kernel 3×3, stride 2, padding 0.

    H_out = floor((64 + 0 − 3) / 2) + 1
          = floor(61/2) + 1
          = 30 + 1
          = 31
    W_out = 31

**Output spatial size: 31×31.** The floor matters because the filter must fit entirely inside the unpadded input. The output channel count depends on the number of filters, which the question does not specify.

### Question 4

Input feature map 16×16, pooling window 2×2, stride 2, no padding.

    H_out = floor((16 − 2) / 2) + 1
          = 7 + 1
          = 8
    W_out = 8

**Output: 8×8 per feature map.** Pooling acts independently on each channel, so C input channels would give 8×8×C output.

### Question 5

Input spatial size 128×128. Both convolutions have kernel size 3, stride 1, and “same” padding. For each convolution, padding is (3−1)/2 = 1 on each side.

    After layer 1:
    H_1 = floor((128 + 2×1 − 3) / 1) + 1 = 128
    W_1 = 128

    After layer 2:
    H_2 = floor((128 + 2×1 − 3) / 1) + 1 = 128
    W_2 = 128

**Output spatial shape: 128×128.** If the second layer has F₂ filters, the full output shape is 128×128×F₂. The question does not give filter counts, so its full channel shape cannot be determined.

### Question 6

`model.train()` sets the model and its submodules to training mode. Newly created PyTorch modules start in training mode, so removing that line before the first training loop generally has no effect if no earlier call switched the model to evaluation mode.

However, `model.eval()` changes the mode and it persists until changed again. If validation calls `model.eval()`, removing `model.train()` at the start of the next epoch leaves the model in evaluation mode. For mode-dependent layers:

- Dropout stops randomly masking activations, changing its regularization behavior.
- Standard batch normalization uses stored running statistics instead of the current batch statistics and stops updating those running statistics.

The specified assignment CNN contains only convolution, ReLU, max pooling, flatten, and linear layers. Those layers behave the same in train and eval modes. Thus removing `model.train()` has no mode-dependent effect on this particular architecture, even after evaluation. Keeping the call is good practice for future architecture changes.

Training mode is separate from gradient tracking. Neither `model.train()` enables gradients nor `model.eval()` disables them. Gradients and optimizer updates can still occur in eval mode; `torch.no_grad()` or `torch.inference_mode()` controls gradient recording. Validation and API prediction use inference mode in this implementation.

## Sources

Architecture, deployment requirement, and question statements: supplied Assignment2.pdf, pages 1–2. Course context: supplied Module 4 slides and practical HTML notebooks, especially Practical 3 (CIFAR-10, PyTorch training, Adam, and cross-entropy). The 64×64 resize, validation split, API contract, and checkpoint format are implementation choices made to satisfy the assignment.
