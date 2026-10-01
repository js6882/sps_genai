from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel, MODEL_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the large model once per server process, rather than once per request.
    app.state.embedding_model = EmbeddingModel()
    yield


app = FastAPI(title="Module 3 API with Word Embeddings", lifespan=lifespan)

# Sample corpus from the Module 3 activity.
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1)
    length: int = Field(ge=1, le=1000)


class EmbeddingResponse(BaseModel):
    word: str
    model: str
    dimensions: int
    embedding: list[float]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.get("/embedding", response_model=EmbeddingResponse)
def get_embedding(
    request: Request,
    word: Annotated[str, Query(min_length=1, max_length=100, description="One word, e.g. apple")],
):
    word = word.strip()
    if not word:
        raise HTTPException(status_code=422, detail="The word must not be blank.")
    try:
        vector = request.app.state.embedding_model.calculate_embedding(word)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return EmbeddingResponse(word=word, model=MODEL_NAME, dimensions=len(vector), embedding=vector)
