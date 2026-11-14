from io import BytesIO
import pytest
import torch
from torch import nn
from PIL import Image
from fastapi.testclient import TestClient
from helper_lib.ebm import EnergyModel, langevin_sample, train_ebm
from helper_lib.diffusion import DiffusionModel, train_diffusion
from helper_lib.model import get_model
from helper_lib.generator import generate_samples
import main

torch.set_num_threads(2)


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * x.square().flatten(1).sum(1)


def test_sampling_changes_pixels_only_and_restores_flags():
    model = Quadratic().train()
    model.scale.grad = torch.tensor(7.0)
    original = torch.full((2, 3, 32, 32), 0.5)
    with torch.no_grad():
        sampled = langevin_sample(model, original, steps=3, step_size=0.1,
                                  noise_std=0, grad_clip=None)
    assert torch.allclose(sampled, original * 0.8 ** 3)
    assert model(sampled).sum() < model(original).sum()
    assert torch.all(original == 0.5)
    assert model.scale.item() == 1 and model.scale.grad.item() == 7
    assert model.training and model.scale.requires_grad
    assert not sampled.requires_grad
    model.scale.requires_grad_(False)
    langevin_sample(model, original, steps=1)
    assert not model.scale.requires_grad


def test_exception_restores_sampler_state():
    class Broken(Quadratic):
        def forward(self, x):
            raise RuntimeError('test failure')
    model = Broken().train()
    with pytest.raises(RuntimeError, match='test failure'):
        langevin_sample(model, torch.zeros(1, 3, 32, 32))
    assert model.training and model.scale.requires_grad


def test_neighbor_penalty_has_the_expected_pixel_gradient():
    model = EnergyModel(channels=8, smoothness=5.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    image = torch.zeros(1, 3, 32, 32)
    image[0, 0, 16, 16] = 1
    image.requires_grad_(True)
    energy = model(image)
    gradient, = torch.autograd.grad(energy.sum(), image)
    # The impulse has four unit differences; each directional mean has 3*32*31 entries.
    assert energy.item() == pytest.approx(5 * 4 / (3 * 32 * 31))
    assert gradient[0, 0, 16, 16].item() == pytest.approx(5 * 8 / (3 * 32 * 31))
    sampled = langevin_sample(model, image, steps=1, noise_std=0, grad_clip=None)
    assert model(sampled).item() < energy.item()


@pytest.mark.parametrize('name', ['EBM', 'Diffusion'])
def test_training_changes_weights_without_input_gradients(name):
    model = get_model(name)
    images = torch.rand(2, 3, 32, 32, requires_grad=True)
    loader = [(images, torch.zeros(2))]
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if name == 'EBM':
        train_ebm(model, loader, optimizer, epochs=1, steps=2, buffer_size=4)
    else:
        train_diffusion(model, loader, optimizer, epochs=1)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert images.grad is None


def test_forward_diffusion_matches_formula():
    model = DiffusionModel(steps=10)
    clean = torch.ones(2, 3, 32, 32)
    noise = torch.full_like(clean, 0.25)
    times = torch.tensor([0, 9])
    noisy, returned = model.q_sample(clean, times, noise)
    a = model.alpha_bar[times].view(2, 1, 1, 1)
    assert torch.allclose(noisy, a.sqrt() * clean + (1 - a).sqrt() * noise)
    assert returned is noise
    assert model(clean, times).shape == clean.shape
    assert model.posterior_variance[0].item() == 0


@pytest.mark.parametrize('model', [EnergyModel(channels=8), DiffusionModel(steps=4)])
def test_seeded_generation_and_weights_unchanged(model):
    if isinstance(model, EnergyModel):
        model.sampling_config['steps'] = 4
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a = generate_samples(model, num_samples=2, seed=42)
    assert a.shape == (2, 3, 32, 32)
    assert 0 <= a.min() <= a.max() <= 1
    assert not a.requires_grad
    assert torch.equal(a, generate_samples(model, num_samples=2, seed=42))
    assert not torch.equal(a, generate_samples(model, num_samples=2, seed=43))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_api_checkpoints_and_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'MODEL_DIR', tmp_path)
    monkeypatch.setattr(main, 'GAN_CHECKPOINT', tmp_path / 'absent.pt')
    with TestClient(main.app) as client:
        assert client.get('/generate/ebm').status_code == 503
        assert client.get('/generate/unknown').status_code == 404
    for name, model in [('ebm', EnergyModel(channels=8, smoothness=2.0)),
                         ('diffusion', DiffusionModel(steps=4))]:
        sampling = dict(steps=4, step_size=10.0, noise_std=0.005)
        torch.save({'config': model.config, 'state_dict': model.state_dict(),
                    'sampling_config': sampling},
                   tmp_path / f'cifar10_{name}.pt')
    with TestClient(main.app) as client:
        assert main.app.state.models['ebm'].smoothness == 2.0
        assert main.app.state.models['ebm'].sampling_config['steps'] == 4
        for name in ['ebm', 'diffusion']:
            url = f'/generate/{name}?num_samples=1&seed=42'
            response = client.get(url)
            assert response.status_code == 200
            assert response.headers['content-type'] == 'image/png'
            image = Image.open(BytesIO(response.content))
            assert image.size == (32, 32)
            image.verify()
            assert client.get(url).content == response.content
            for query in ['num_samples=0', 'num_samples=17', 'seed=-1']:
                assert client.get(f'/generate/{name}?{query}').status_code == 422


def test_ebm_progress_saves_weights_optimizer_and_epoch_numbers(tmp_path):
    model = EnergyModel(channels=8)
    model.sampling_config['steps'] = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = [(torch.rand(2, 3, 32, 32) * 2 - 1, torch.zeros(2))]
    train_ebm(model, loader, optimizer, epochs=1, steps=2, buffer_size=4,
              sample_dir=tmp_path, history=[{'epoch': 1, 'loss': 0.0}])
    saved = torch.load(tmp_path / 'latest.pt', weights_only=True)
    restored = EnergyModel(**saved['config'])
    restored.load_state_dict(saved['state_dict'])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(saved['optimizer'])
    assert saved['history'][-1]['epoch'] == 2
    assert 'real_energy' in saved['history'][-1]
    assert 'fake_energy' in saved['history'][-1]
    assert restored_optimizer.state_dict()['state']
    assert (tmp_path / 'epoch_02.png').exists()
