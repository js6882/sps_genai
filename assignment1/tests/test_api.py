"""Integration tests use the real pretrained model, including its numeric output."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_embedding_matches_spacy(client):
    response = client.get("/embedding", params={"word": "apple"})
    assert response.status_code == 200
    data = response.json()
    assert data["word"] == "apple"
    assert data["model"] == "en_core_web_lg"
    assert data["dimensions"] == len(data["embedding"]) == 300
    expected = app.state.embedding_model.nlp("apple").vector
    np.testing.assert_allclose(data["embedding"], expected)
    assert np.isfinite(data["embedding"]).all()
    assert np.linalg.norm(data["embedding"]) > 0


@pytest.mark.parametrize("word", ["", "   ", "two words", "!!!", "a" * 101])
def test_invalid_word(client, word):
    assert client.get("/embedding", params={"word": word}).status_code == 422


def test_missing_word(client):
    assert client.get("/embedding").status_code == 422


def test_unknown_word(client):
    response = client.get("/embedding", params={"word": "qzxwvvqzxwvvqzxwvv"})
    assert response.status_code == 404


def test_trim_whitespace(client):
    response = client.get("/embedding", params={"word": " apple "})
    assert response.status_code == 200
    assert response.json()["word"] == "apple"


def test_original_routes(client):
    assert client.get("/").json() == {"Hello": "World"}
    response = client.post("/generate", json={"start_word": "we", "length": 3})
    assert response.status_code == 200
    assert response.json()["generated_text"] in {"we are generating", "we are simple"}
