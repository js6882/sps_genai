"""Energy network and contrastive-divergence training for CIFAR-10 in [-1, 1]."""
import math
import torch
from torch import nn


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 2 * 2, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, images):
        return self.net(images).squeeze(-1)


def langevin_sample(model, images, steps=60, step_size=10.0,
                    noise_std=0.005, generator=None, grad_clip=0.03):
    """Add noise and lower image energy by updating pixels, not weights.

    Pixel gradients are needed, so do not use torch.inference_mode().
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError('step_size must be finite and positive')
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError('noise_std must be finite and nonnegative')
    if grad_clip is not None and (not math.isfinite(grad_clip) or grad_clip <= 0):
        raise ValueError('grad_clip must be finite and positive, or None')
    flags = [p.requires_grad for p in model.parameters()]
    training = model.training
    try:
        model.eval()
        model.requires_grad_(False)
        x = images.detach().clone()
        for _ in range(steps):
            with torch.no_grad():
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                                    generator=generator)
                x = (x + noise_std * noise).clamp(-1, 1)
            # Enable gradients here to calculate how the pixels should change.
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                gradient, = torch.autograd.grad(model(x).sum(), x)
            with torch.no_grad():
                if grad_clip is not None:
                    gradient = gradient.clamp(-grad_clip, grad_clip)
                x = (x - step_size * gradient).clamp(-1, 1)
        return x.detach()
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        model.train(training)


def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10,
              steps=60, step_size=10.0, noise_std=0.005, alpha=0.1,
              buffer_size=1024):
    if epochs < 1 or buffer_size < 1 or alpha < 0:
        raise ValueError('epochs/buffer_size must be positive and alpha nonnegative')
    model.to(device)
    buffer = torch.empty(buffer_size, 3, 32, 32, device=device).uniform_(-1, 1)
    history = []
    for epoch in range(epochs):
        total, count = 0.0, 0
        for real, _ in data_loader:
            real = real.to(device).detach()
            n = real.shape[0]
            indices = torch.randint(buffer_size, (n,), device=device)
            initial = buffer[indices].clone()
            fresh = torch.rand(n, 1, 1, 1, device=device) < 0.05
            initial = torch.where(fresh, torch.empty_like(initial).uniform_(-1, 1), initial)
            optimizer.zero_grad(set_to_none=True)
            fake = langevin_sample(model, initial, steps, step_size, noise_std)
            buffer[indices] = fake
            model.train()
            real = (real + noise_std * torch.randn_like(real)).clamp(-1, 1)
            # Detach fake images so this step only trains the model weights.
            real_energy, fake_energy = model(real), model(fake.detach())
            contrastive = real_energy.mean() - fake_energy.mean()
            regularizer = alpha * (real_energy.square().mean() + fake_energy.square().mean())
            loss = contrastive + regularizer
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total += loss.item() * n
            count += n
        if count == 0:
            raise ValueError('data_loader must contain at least one batch')
        history.append({'epoch': epoch + 1, 'loss': total / count})
        print(f'EBM epoch {epoch + 1}/{epochs}: loss={total / count:.5f}', flush=True)
    model.history = history
    return model
