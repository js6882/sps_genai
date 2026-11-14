"""Energy network and contrastive-divergence training for CIFAR-10 in [-1, 1]."""
import math
import time
import torch
from torch import nn


class EnergyModel(nn.Module):
    def __init__(self, channels=32, smoothness=5.0):
        super().__init__()
        if not math.isfinite(smoothness) or smoothness < 0:
            raise ValueError('smoothness must be finite and nonnegative')
        self.config = {'channels': channels, 'smoothness': smoothness}
        self.smoothness = smoothness
        self.sampling_config = {'steps': 240, 'step_size': 10.0, 'noise_std': 0.005}
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(channels * 4, channels * 4, 4, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(channels * 4 * 2 * 2, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, images):
        learned_energy = self.net(images).squeeze(-1)
        horizontal = images[:, :, :, 1:] - images[:, :, :, :-1]
        vertical = images[:, :, 1:, :] - images[:, :, :-1, :]
        roughness = horizontal.square().mean((1, 2, 3)) + vertical.square().mean((1, 2, 3))
        return learned_energy + self.smoothness * roughness


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
              buffer_size=4096, sample_dir=None, history=None):
    if epochs < 1 or buffer_size < 1 or alpha < 0:
        raise ValueError('epochs/buffer_size must be positive and alpha nonnegative')
    model.to(device)
    buffer = torch.empty(buffer_size, 3, 32, 32, device=device).uniform_(-1, 1)
    history = list(history or [])
    first_epoch = len(history)
    for epoch in range(epochs):
        started = time.monotonic()
        total, count = 0.0, 0
        total_real, total_fake = 0.0, 0.0
        for batch_number, (real, _) in enumerate(data_loader, 1):
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
            energies = model(torch.cat([real, fake.detach()]))
            real_energy, fake_energy = energies.chunk(2)
            contrastive = real_energy.mean() - fake_energy.mean()
            regularizer = alpha * (real_energy.square().mean() + fake_energy.square().mean())
            loss = contrastive + regularizer
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total += loss.item() * n
            total_real += real_energy.detach().mean().item() * n
            total_fake += fake_energy.detach().mean().item() * n
            count += n
            if batch_number % 100 == 0:
                print(f'  batch {batch_number}: loss={total / count:.5f}', flush=True)
        if count == 0:
            raise ValueError('data_loader must contain at least one batch')
        epoch_number = first_epoch + epoch + 1
        history.append({'epoch': epoch_number, 'loss': total / count,
                        'real_energy': total_real / count, 'fake_energy': total_fake / count,
                        'seconds': time.monotonic() - started, 'config': dict(model.config),
                        'batch_size': getattr(data_loader, 'batch_size', None)})
        print(f'EBM epoch {epoch_number}/{first_epoch + epochs}: loss={total / count:.5f}', flush=True)
        if sample_dir is not None:
            from pathlib import Path
            from torchvision.utils import save_image
            sample_dir = Path(sample_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)
            # Save a small checkpoint after each epoch so longer runs can resume.
            pending = sample_dir / 'latest.tmp'
            torch.save({'state_dict': model.state_dict(), 'config': model.config,
                        'optimizer': optimizer.state_dict(), 'history': history},
                       pending)
            pending.replace(sample_dir / 'latest.pt')
            # This RNG is separate from the one used for training.
            rng = torch.Generator(device=device).manual_seed(123)
            initial = torch.rand(16, 3, 32, 32, device=device, generator=rng) * 2 - 1
            images = langevin_sample(model, initial, steps=model.sampling_config['steps'], step_size=step_size,
                                      noise_std=noise_std, generator=rng)
            save_image((images.cpu() + 1) / 2, sample_dir / f'epoch_{epoch_number:02d}.png', nrow=4)
    model.history = history
    return model
