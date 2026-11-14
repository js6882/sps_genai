from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
import torch
from fastapi import FastAPI, HTTPException, Query, Response
from torchvision.utils import save_image
from helper_lib.model import Generator
from helper_lib.ebm import EnergyModel
from helper_lib.diffusion import DiffusionModel
from helper_lib.generator import generate_samples

MODEL_DIR = Path(__file__).resolve().parent / 'models'
GAN_CHECKPOINT = Path(__file__).resolve().parents[1] / 'assignment3/models/mnist_generator.pt'


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    app.state.models = {}
    app.state.locks = {}
    for name in ('gan', 'ebm', 'diffusion'):
        path = GAN_CHECKPOINT if name == 'gan' else MODEL_DIR / f'cifar10_{name}.pt'
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if name == 'gan':
            model = Generator()
            model.load_state_dict(checkpoint)
        else:
            model = (EnergyModel(**checkpoint['config']) if name == 'ebm'
                     else DiffusionModel(**checkpoint['config']))
            model.load_state_dict(checkpoint['state_dict'])
            if name == 'ebm':
                model.sampling_config = checkpoint.get('sampling_config', model.sampling_config)
        app.state.models[name] = model.eval()
        app.state.locks[name] = Lock()
    yield


app = FastAPI(title='Module 6 + Module 8 Image Generator API', lifespan=lifespan)


@app.get('/')
def read_root():
    return {'models': sorted(app.state.models), 'docs': '/docs'}


@app.get('/generate/{model_name}')
def generate_images(model_name: str,
                    num_samples: int = Query(default=4, ge=1, le=16),
                    seed: int | None = Query(default=None, ge=0, le=2147483647)):
    if model_name not in ('gan', 'ebm', 'diffusion'):
        raise HTTPException(404, 'Choose gan, ebm, or diffusion')
    if model_name not in app.state.models:
        raise HTTPException(503, f'{model_name} checkpoint missing; follow README training instructions')
    # Handle one request per model while sampling changes its gradient settings.
    with app.state.locks[model_name]:
        images = generate_samples(app.state.models[model_name], num_samples=num_samples, seed=seed)
    output = BytesIO()
    save_image(images, output, format='PNG', nrow=min(8, num_samples))
    return Response(content=output.getvalue(), media_type='image/png')
