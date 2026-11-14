import torch


def generate_samples(model, device='cpu', num_samples=10, seed=None):
    from .ebm import EnergyModel, langevin_sample
    from .diffusion import DiffusionModel
    if isinstance(model, DiffusionModel):
        return model.sample(num_samples=num_samples, seed=seed).cpu()
    if isinstance(model, EnergyModel):
        rng = torch.Generator(device=device)
        rng.seed() if seed is None else rng.manual_seed(seed)
        initial = torch.rand(num_samples, 3, 32, 32, device=device, generator=rng) * 2 - 1
        return ((langevin_sample(model, initial, generator=rng,
                                 **model.sampling_config) + 1) / 2).cpu()
    model.eval()
    rng = torch.Generator(device='cpu')
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)
    noise = torch.randn(num_samples, model.z_dim, generator=rng).to(device)
    with torch.no_grad():
        images = model(noise).cpu()
    return (images + 1) / 2
