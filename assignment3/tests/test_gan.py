from io import BytesIO
import torch
from PIL import Image
from fastapi.testclient import TestClient
from helper_lib.model import GAN, Generator, Discriminator
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
import main


def test_architecture():
    gen = Generator().eval()
    disc = Discriminator().eval()
    shapes = {}
    def capture(name):
        def hook(module, inputs, output):
            shapes[name] = tuple(output.shape)
        return hook
    gen.deconv1.register_forward_hook(capture('g1'))
    gen.deconv2.register_forward_hook(capture('g2'))
    disc.conv1.register_forward_hook(capture('d1'))
    disc.conv2.register_forward_hook(capture('d2'))
    images = gen(torch.randn(4, 100))
    probabilities = disc(images)
    assert shapes == {'g1': (4, 64, 14, 14), 'g2': (4, 1, 28, 28),
                      'd1': (4, 64, 14, 14), 'd2': (4, 128, 7, 7)}
    assert probabilities.shape == (4, 1)
    assert images.min() >= -1 and images.max() <= 1
    assert probabilities.min() >= 0 and probabilities.max() <= 1


def test_training_updates_both_networks():
    torch.set_num_threads(2)
    model = GAN()
    before_g = model.generator.fc.weight.detach().clone()
    before_d = model.discriminator.fc.weight.detach().clone()
    optimizer = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=0.0002)
                 for name in ['generator', 'discriminator']}
    data = [(torch.rand(4, 1, 28, 28) * 2 - 1, torch.zeros(4))]
    train_gan(model, data, torch.nn.BCELoss(), optimizer, epochs=1)
    assert not torch.equal(before_g, model.generator.fc.weight)
    assert not torch.equal(before_d, model.discriminator.fc.weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_api(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'generator.pt'
    torch.save(Generator().state_dict(), checkpoint)
    monkeypatch.setattr(main, 'CHECKPOINT', checkpoint)
    with TestClient(main.app) as client:
        response = client.get('/generate/gan?num_samples=16&seed=42')
        assert response.status_code == 200
        assert response.headers['content-type'] == 'image/png'
        Image.open(BytesIO(response.content)).verify()
        assert response.content == client.get('/generate/gan?num_samples=16&seed=42').content
        assert response.content != client.get('/generate/gan?num_samples=16&seed=43').content
        for query in ['num_samples=0', 'num_samples=65', 'seed=-1']:
            assert client.get('/generate/gan?' + query).status_code == 422


def test_saved_model_roundtrip(tmp_path):
    model = Generator().eval()
    path = tmp_path / 'generator.pt'
    torch.save(model.state_dict(), path)
    restored = Generator().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(generate_samples(model, seed=42), generate_samples(restored, seed=42))
