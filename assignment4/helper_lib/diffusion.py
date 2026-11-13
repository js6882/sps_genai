"""Unconditional CIFAR-10 DDPM with a time-conditioned U-Net noise predictor."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def forward(self, timesteps):
        frequencies = torch.exp(torch.arange(16, device=timesteps.device) *
                                (-math.log(10000) / 15))
        angles = timesteps.float()[:, None] * frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.time = nn.Linear(128, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(time)[:, :, None, None]
        return self.skip(x) + F.silu(self.norm2(self.conv2(h)))


class NoiseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(TimeEmbedding(), nn.Linear(32, 128), nn.SiLU(),
                                  nn.Linear(128, 128))
        self.down1 = ResidualBlock(3, 32)
        self.down2 = ResidualBlock(32, 64)
        self.middle = ResidualBlock(64, 64)
        self.up2 = ResidualBlock(128, 32)
        self.up1 = ResidualBlock(64, 32)
        self.output = nn.Conv2d(32, 3, 1)

    def forward(self, images, timesteps):
        time = self.time(timesteps)
        skip1 = self.down1(images, time)  # 32 x 32
        skip2 = self.down2(F.avg_pool2d(skip1, 2), time)  # 16 x 16
        h = self.middle(F.avg_pool2d(skip2, 2), time)  # 8 x 8
        h = F.interpolate(h, size=skip2.shape[-2:], mode='nearest')
        h = self.up2(torch.cat([h, skip2], dim=1), time)
        h = F.interpolate(h, size=skip1.shape[-2:], mode='nearest')
        return self.output(self.up1(torch.cat([h, skip1], dim=1), time))


class DiffusionModel(nn.Module):
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        if not isinstance(steps, int) or steps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError('need steps >= 2 and 0 < beta_start <= beta_end < 1')
        self.config = dict(steps=steps, beta_start=beta_start, beta_end=beta_end)
        self.steps = steps
        self.network = NoiseUNet()
        beta = torch.linspace(beta_start, beta_end, steps)
        alpha = 1 - beta
        alpha_bar = alpha.cumprod(0)
        previous = torch.cat([torch.ones(1), alpha_bar[:-1]])
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('posterior_variance', beta * (1 - previous) / (1 - alpha_bar))
        self.register_buffer('clean_coef', beta * previous.sqrt() / (1 - alpha_bar))
        self.register_buffer('noisy_coef', (1 - previous) * alpha.sqrt() / (1 - alpha_bar))

    def forward(self, images, timesteps):
        return self.network(images, timesteps)

    def q_sample(self, clean, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bar[timesteps].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise, noise

    @torch.no_grad()
    def sample(self, num_samples=10, seed=None):
        device = self.alpha_bar.device
        rng = torch.Generator(device=device)
        rng.seed() if seed is None else rng.manual_seed(seed)
        training = self.training
        self.eval()
        try:
            x = torch.randn(num_samples, 3, 32, 32, device=device, generator=rng)
            for step in reversed(range(self.steps)):
                times = torch.full((num_samples,), step, device=device, dtype=torch.long)
                predicted_noise = self(x, times)
                clean = ((x - (1 - self.alpha_bar[step]).sqrt() * predicted_noise) /
                         self.alpha_bar[step].sqrt()).clamp(-1, 1)
                mean = self.clean_coef[step] * clean + self.noisy_coef[step] * x
                if step > 0:
                    noise = torch.randn(x.shape, device=device, generator=rng)
                    x = mean + self.posterior_variance[step].sqrt() * noise
                else:
                    x = mean  # No noise at the final step.
            return (x.clamp(-1, 1) + 1) / 2
        finally:
            self.train(training)


def train_diffusion(model, data_loader, optimizer, device='cpu', epochs=10):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    model.to(device)
    history = []
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for images, _ in data_loader:
            images = images.to(device).detach()
            timesteps = torch.randint(model.steps, (len(images),), device=device)
            # Forward corruption is fixed; only noise-predictor weights learn.
            with torch.no_grad():
                noisy, noise = model.q_sample(images, timesteps)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(noisy, timesteps), noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
        if count == 0:
            raise ValueError('data_loader must contain at least one batch')
        history.append({'epoch': epoch + 1, 'noise_mse': total / count})
        print(f'Diffusion epoch {epoch + 1}/{epochs}: noise_mse={total / count:.5f}', flush=True)
    model.history = history
    return model
