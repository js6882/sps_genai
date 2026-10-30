import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch size must be positive')

    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = args.device
    output = Path(__file__).resolve().parents[1] / 'models'
    samples = output / 'samples'
    samples.mkdir(parents=True, exist_ok=True)
    data_loader = get_data_loader(batch_size=args.batch_size)
    model = get_model('GAN')
    optimizer = {
        'generator': optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        'discriminator': optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    }
    model = train_gan(model, data_loader, nn.BCELoss(), optimizer,
                      device=device, epochs=args.epochs, sample_dir=samples)
    model.cpu().eval()
    torch.save(model.generator.state_dict(), output / 'mnist_generator.pt')
    save_image(generate_samples(model.generator, num_samples=64, seed=123),
               output / 'generated_digits.png', nrow=8)
    report = {'dataset': 'MNIST training split', 'images': len(data_loader.dataset),
              'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': 42,
              'device': device, 'torch': torch.__version__, 'learning_rate': 0.0002,
              'betas': [0.5, 0.999], 'history': model.history}
    (output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
