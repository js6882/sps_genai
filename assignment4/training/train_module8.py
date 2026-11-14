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
from helper_lib.ebm import EnergyModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ebm', 'diffusion'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--sampling-steps', type=int, default=60, help='EBM negative-sampling steps')
    parser.add_argument('--channels', type=int, default=32, help='EBM first-layer channels')
    parser.add_argument('--smoothness', type=float, help='EBM neighbor penalty; default 5 for new runs')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'models')
    parser.add_argument('--sample-dir', type=Path, help='Save EBM progress after every epoch')
    parser.add_argument('--resume', type=Path, help='Continue an EBM progress checkpoint')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.sampling_steps, args.channels) < 1:
        parser.error('epochs, batch-size, sampling-steps and channels must be positive')
    if args.resume and args.model != 'ebm':
        parser.error('--resume is currently supported for EBM only')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    loader = get_data_loader(args.data_dir, args.batch_size)
    checkpoint = (torch.load(args.resume, map_location=args.device, weights_only=True)
                  if args.resume else None)
    config = checkpoint['config'] if checkpoint else {'channels': args.channels}
    if args.model == 'ebm':
        if checkpoint and 'smoothness' not in config:
            config = dict(config, smoothness=0.0)
        if args.smoothness is not None:
            config = dict(config, smoothness=args.smoothness)
    model = (EnergyModel(**config) if args.model == 'ebm'
             else get_model('Diffusion')).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    started = time.monotonic()
    if args.model == 'ebm':
        train_ebm(model, loader, optimizer, args.device, args.epochs, steps=args.sampling_steps,
                  sample_dir=args.sample_dir, history=checkpoint['history'] if checkpoint else None)
    else:
        train_diffusion(model, loader, optimizer, args.device, args.epochs)
    elapsed = time.monotonic() - started
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = dict(dataset='CIFAR-10 training split', images=len(loader.dataset),
                  epochs=len(model.history), current_run_epochs=args.epochs,
                  resumed=bool(checkpoint), batch_size=args.batch_size, seed=42,
                  device=args.device, learning_rate=1e-4,
                  training_seconds=sum(row.get('seconds', 0.0) for row in model.history)
                  if args.model == 'ebm' else elapsed,
                  current_run_seconds=elapsed,
                  sampling_steps=args.sampling_steps if args.model == 'ebm' else 1000,
                  torch=str(torch.__version__), history=model.history)
    if args.model == 'ebm':
        report.update(buffer_size=4096, step_size=10.0, noise_std=0.005,
                      regularization=0.1, generation_steps=model.sampling_config['steps'],
                      architecture='4x4 stride-2 convolutions', channels=model.config['channels'],
                      smoothness=model.smoothness)
    model.cpu().eval()
    torch.save(dict(state_dict=model.state_dict(),
                    config=model.config,
                    sampling_config=getattr(model, 'sampling_config', {}),
                    training=report), output / f'cifar10_{args.model}.pt')
    (output / f'{args.model}_training.json').write_text(json.dumps(report, indent=2) + '\n')
    save_image(generate_samples(model, num_samples=16, seed=123),
               output / f'{args.model}_samples.png', nrow=4)


if __name__ == '__main__':
    main()
