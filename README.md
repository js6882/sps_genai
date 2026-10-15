# sps_genai

APAN5560 Generative AI Assignment

## Assignment 1 — Part 1: Word Embedding API

This project extends the [Module 3 FastAPI activity](https://gurgentus.github.io/applied_genai_notebooks/Module%203/gentext_project/) with a word embedding endpoint. It uses `en_core_web_lg` and `nlp(word).vector`, as in [Module 2 Practical 3](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_2_Practical_3_Word_Embeddings.py).

The classroom `/` and `/generate` routes and sample corpus are retained. Because the tutorial asks students to implement `BigramModel` themselves, `app/bigram_model.py` provides a small implementation: lowercase whitespace tokens, observed bigram counts, and weighted random sampling. Generation stops at a word with no observed successor; `length` includes the starting word.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). This project selects Python 3.12 through `.python-version`; uv can install it if needed. Open a terminal in this project folder, then run:

```bash
uv sync --frozen
uv run fastapi dev main.py
```

The first sync downloads the large spaCy model (approximately 560 MB). The model is declared as a dependency, so a separate spaCy download command is unnecessary. It loads once at application startup. If loading fails, the server will not start; check that dependency installation completed.

Open http://127.0.0.1:8000/docs to try the API, or open:

```text
http://127.0.0.1:8000/embedding?word=apple
```

## New endpoint

`GET /embedding?word=apple`

The required `word` query parameter accepts one alphabetic spaCy token, up to 100 characters. Leading and trailing whitespace is removed; capitalization is preserved. Use English words because the selected model is English. Phrases, punctuation, and words split into multiple tokens (including some hyphenated words and contractions) are rejected.

A successful JSON response contains:

| Field | Meaning |
| --- | --- |
| `word` | The trimmed query word |
| `model` | `en_core_web_lg` |
| `dimensions` | 300 |
| `embedding` | All 300 numeric vector components |

The vector is obtained from the pretrained model, not generated randomly. `.tolist()` converts the NumPy vector into JSON-compatible numbers. Missing, blank, too-long, and invalid inputs return HTTP 422. A word with no pretrained vector returns HTTP 404 rather than presenting a zero vector as a meaningful embedding.

```bash
curl "http://127.0.0.1:8000/embedding?word=apple"
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"start_word":"we","length":3}'
```

## Tests

```bash
uv run pytest -q
```

Tests use the actual spaCy model to check the returned vector against spaCy's output, verify all 300 values are finite and nonzero as a vector, exercise invalid and unknown words, and check the original routes.

## Optional Docker

```bash
docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai
```

Reference: [spaCy vectors documentation](https://spacy.io/usage/linguistic-features#vectors-similarity).

## Assignment 2: CNN image classification

The implementation and six worked theory answers are described in [ASSIGNMENT2.md](ASSIGNMENT2.md). The CNN matches the assigned architecture exactly, including its 64×64 RGB input and 8,192-to-100 fully connected layer (825,398 trainable parameters).

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
