import time
import torch
from torchvision.utils import save_image


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=20,
              sample_dir=None):
    gen = model.generator
    disc = model.discriminator
    model.to(device)
    fixed_noise = torch.randn(64, model.z_dim, device=device)
    datalogs = []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_d = 0
        total_g = 0
        total_images = 0

        for real, _ in data_loader:
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train discriminator
            noise = torch.randn(batch_size, model.z_dim, device=device)
            fake = gen(noise)
            optimizer['discriminator'].zero_grad()
            loss_real = criterion(disc(real), real_labels)
            loss_fake = criterion(disc(fake.detach()), fake_labels)
            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            optimizer['discriminator'].step()

            # Train generator
            disc.requires_grad_(False)
            optimizer['generator'].zero_grad()
            loss_gen = criterion(disc(fake), real_labels)
            loss_gen.backward()
            optimizer['generator'].step()
            disc.requires_grad_(True)

            total_d += loss_disc.item() * batch_size
            total_g += loss_gen.item() * batch_size
            total_images += batch_size

        datalogs.append({'epoch': epoch + 1, 'D loss': total_d / total_images,
                        'G loss': total_g / total_images})
        print(f"Epoch {epoch + 1}/{epochs} | D loss: {total_d / total_images:.4f} | "
              f"G loss: {total_g / total_images:.4f} | {time.time() - start:.1f}s", flush=True)
        if sample_dir is not None:
            gen.eval()
            with torch.no_grad():
                images = (gen(fixed_noise).cpu() + 1) / 2
            save_image(images, sample_dir / f'epoch_{epoch + 1:02d}.png', nrow=8)

    model.history = datalogs
    return model
