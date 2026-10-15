# Assignment 2: CNN image classification

This project extends Assignment 1 with a CIFAR-10 classifier. It retains the `/`, `/generate`, and `/embedding` endpoints and adds `POST /classify`.

Run all commands below from the `assignment2/` folder. The sibling `assignment1/` project is not needed to run this version.

## Setup

Python 3.12 is selected by `.python-version`. Install dependencies with `uv sync --frozen`. This also installs the spaCy English model required by the existing embedding endpoint.

## CNN implementation


The implementation and deployment are described in [ASSIGNMENT2.md](ASSIGNMENT2.md). The CNN matches the assigned architecture exactly, including its 64×64 RGB input and 8,192-to-100 fully connected layer (825,398 trainable parameters).

### Train before serving

After installing the updated dependencies, run from this project folder:

```bash
uv sync --frozen
uv run python -m training.train_cnn --epochs 10
uv run fastapi dev main.py
```

Training downloads CIFAR-10 into `data/`, resizes images from 32×32 to 64×64, trains on 45,000 examples, and selects a checkpoint using 5,000 validation examples. It evaluates the selected checkpoint on the separate 10,000-image test set. Training prints progress after each epoch. Apple MPS or CUDA is selected when available; otherwise training uses CPU. Use `--device cpu` to override automatic selection. See `uv run python -m training.train_cnn --help` for options.

Outputs:

- `models/cifar10_cnn.pt`: trained checkpoint, excluded from Git.
- `models/cifar10_cnn.metrics.json`: training settings, epoch history, selected epoch, and actual test results.

The API loads the trained checkpoint once at startup and performs inference on CPU. Set `CNN_CHECKPOINT` to use another checkpoint path. Missing weights produce HTTP 503 for `/classify`; restart the API after training or copying the checkpoint. Invalid or incompatible checkpoints cause startup to fail so an incorrect model is not silently served. The existing spaCy model is still required for API startup.

### Classify an image

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -F 'image=@/absolute/path/to/image.png'
```

Upload a PNG or JPEG in the multipart field `image`. The endpoint converts it to RGB, resizes it to 64×64, and scales pixels to [0,1] exactly as during training. The JSON response contains `class_id` (0–9), `class_name`, `confidence`, and a `probabilities` object with all ten class probabilities. `confidence` is the highest softmax probability and is not a calibrated correctness guarantee. Classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Images outside those categories still receive one of those labels.

Empty, corrupt, unsupported, or unsafe images return HTTP 422; a missing image also returns 422. The endpoint rejects uploaded file content over 10 MiB with HTTP 413. Configure a request body limit in your hosting proxy too, since multipart parsing occurs before the endpoint reads the file.

### Validate and deploy

```bash
uv run pytest -q
```

Tests check every layer shape, parameter count, gradient flow, deterministic checkpoint loading, image formats, error responses, and the existing text endpoints. Synthetic test weights validate the API plumbing; classification quality comes from the separate CIFAR-10 evaluation.

Train or copy `models/cifar10_cnn.pt` before building the Docker image. The Dockerfile copies `models/` into the image. Training data is excluded from Docker and Git. Model weights are excluded from Git; preserve and transfer the trained checkpoint separately when deploying.

### Completed training run

The 10-epoch run (seed 42, Apple MPS) selected epoch 9: validation accuracy 60.58%, test accuracy **61.45%** on 10,000 CIFAR-10 test images, and mean test cross-entropy 1.081308. See `models/cifar10_cnn.metrics.json` for the full recorded history. The downloadable submission ZIP includes the trained checkpoint; a GitHub clone requires training or copying that checkpoint.
