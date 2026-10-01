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
