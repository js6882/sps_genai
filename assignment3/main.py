from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
import torch
from fastapi import FastAPI, Query, Response
from torchvision.utils import save_image
from helper_lib.model import Generator
from helper_lib.generator import generate_samples

CHECKPOINT = Path(__file__).resolve().parent / 'models' / 'mnist_generator.pt'


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    model = Generator()
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
    app.state.generator = model.eval()
    yield


app = FastAPI(title='Module 6 GAN API', lifespan=lifespan)


@app.get('/')
def read_root():
    return {'Hello': 'World'}


@app.get('/generate/gan')
def generate_digits(num_samples: int = Query(default=16, ge=1, le=64),
                    seed: int | None = Query(default=None, ge=0, le=2147483647)):
    images = generate_samples(app.state.generator, num_samples=num_samples, seed=seed)
    output = BytesIO()
    save_image(images, output, format='PNG', nrow=min(8, num_samples))
    return Response(content=output.getvalue(), media_type='image/png')
