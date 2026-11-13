import argparse
import json
import time
from pathlib import Path
import torch
from torchvision.utils import save_image
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm, train_diffusion
from helper_lib.generator import generate_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ebm', 'diffusion'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--sampling-steps', type=int, default=60, help='EBM negative-sampling steps')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.sampling_steps) < 1:
        parser.error('epochs, batch-size and sampling-steps must be positive')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    loader = get_data_loader(args.data_dir, args.batch_size)
    model = get_model('EBM' if args.model == 'ebm' else 'Diffusion').to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    started = time.monotonic()
    if args.model == 'ebm':
        train_ebm(model, loader, optimizer, args.device, args.epochs, steps=args.sampling_steps)
    else:
        train_diffusion(model, loader, optimizer, args.device, args.epochs)
    elapsed = time.monotonic() - started
    output = Path(__file__).resolve().parents[1] / 'models'
    output.mkdir(exist_ok=True)
    report = dict(dataset='CIFAR-10 training split', images=len(loader.dataset),
                  epochs=args.epochs, batch_size=args.batch_size, seed=42,
                  device=args.device, learning_rate=1e-4, training_seconds=elapsed,
                  sampling_steps=args.sampling_steps if args.model == 'ebm' else 1000,
                  torch=str(torch.__version__), history=model.history)
    model.cpu().eval()
    torch.save(dict(state_dict=model.state_dict(),
                    config=model.config if args.model == 'diffusion' else {},
                    training=report), output / f'cifar10_{args.model}.pt')
    (output / f'{args.model}_training.json').write_text(json.dumps(report, indent=2) + '\n')
    save_image(generate_samples(model, num_samples=16, seed=123),
               output / f'{args.model}_samples.png', nrow=4)


if __name__ == '__main__':
    main()
