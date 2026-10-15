from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch

from app.cnn_model import AssignmentCNN, CLASSES
from app.image_classifier import ImageClassifier, MAX_UPLOAD_BYTES
import main


def test_architecture_and_backward():
    model = AssignmentCNN()
    x = torch.randn(2, 3, 64, 64)
    shapes = []
    handles = [layer.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
               for layer in model.layers]
    logits = model(x)
    for handle in handles:
        handle.remove()
    assert shapes == [(2,16,64,64), (2,16,64,64), (2,16,32,32),
                      (2,32,32,32), (2,32,32,32), (2,32,16,16),
                      (2,8192), (2,100), (2,100), (2,10)]
    assert sum(p.numel() for p in model.parameters()) == 825398
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 9])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Deterministic synthetic checkpoint tests plumbing, not trained accuracy.
    model = AssignmentCNN()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.layers[-1].bias[7] = 3
    checkpoint = tmp_path / 'test.pt'
    torch.save({'state_dict': model.state_dict(), 'classes': list(CLASSES),
                'preprocessing': 'resize64_to_tensor_v1'}, checkpoint)
    monkeypatch.setenv('CNN_CHECKPOINT', str(checkpoint))
    monkeypatch.setattr(main, 'EmbeddingModel', lambda: object())
    with TestClient(main.app) as c:
        yield c


@pytest.mark.parametrize('mode,fmt', [('RGB','JPEG'), ('L','PNG'), ('RGBA','PNG')])
def test_prediction(client, mode, fmt):
    content = BytesIO()
    Image.new(mode, (32, 32)).save(content, format=fmt)
    response = client.post('/classify', files={'image': ('image', content.getvalue())})
    assert response.status_code == 200
    result = response.json()
    assert result['class_name'] == 'horse'
    assert result['class_id'] == 7
    assert sum(result['probabilities'].values()) == pytest.approx(1)
    assert result['confidence'] == result['probabilities']['horse']


def test_invalid_uploads(client):
    assert client.post('/classify').status_code == 422
    for data in [b'', b'not an image', b'\x89PNG\r\n\x1a\ntruncated']:
        assert client.post('/classify', files={'image': ('bad.png', data)}).status_code == 422
    assert client.post('/classify', files={'image': ('big.png', b'x' * (MAX_UPLOAD_BYTES+1))}).status_code == 413
    content = BytesIO()
    Image.new('RGB', (8,8)).save(content, format='GIF')
    assert client.post('/classify', files={'image': ('bad.gif', content.getvalue())}).status_code == 422


def test_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv('CNN_CHECKPOINT', str(tmp_path / 'absent.pt'))
    monkeypatch.setattr(main, 'EmbeddingModel', lambda: object())
    with TestClient(main.app) as client:
        assert client.get('/').status_code == 200
        assert client.post('/classify', files={'image': ('a.png', b'a')}).status_code == 503
