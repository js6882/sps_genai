import torch


def generate_samples(model, device='cpu', num_samples=10, seed=None):
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
