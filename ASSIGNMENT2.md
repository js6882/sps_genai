# Assignment 2: CNN implementation and deployment

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
