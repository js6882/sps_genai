# sps_genai

APAN5560 Generative AI assignments, organized by assignment.

| Folder | Contents |
| --- | --- |
| [assignment1](assignment1/) | Word embedding API, text generation, tests, and setup instructions |
| [assignment2](assignment2/) | CNN architecture, CIFAR-10 training, image classification API, tests, and measured results |
| [assignment3](assignment3/) | MNIST GAN, trained generator, image generation API, and Docker setup |

Each folder is a complete project with its own README, dependencies and Dockerfile. Assignment 2 builds on Assignment 1 and includes its text endpoints.

## Run Assignment 1

```bash
cd assignment1
uv sync --frozen
uv run fastapi dev main.py
```

## Run Assignment 2

```bash
cd assignment2
uv sync --frozen
uv run python -m training.train_cnn --epochs 10
uv run fastapi dev main.py
```

For Assignment 2, train first or copy the trained checkpoint to `assignment2/models/cifar10_cnn.pt`. See its [README](assignment2/README.md) for API examples and training results.

Run tests or build Docker images from the relevant assignment folder. The APIs use port 8000 by default; run one at a time or choose another port.

Assignment 2 Part 3 theory answers are provided separately and are not included in the current repository files.
