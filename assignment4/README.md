# Assignment 4: Energy-Based and Diffusion Models

This project extends the Module 6 helper-library/API structure with CIFAR-10 energy-based and diffusion image generators. Assignment 3 is preserved. The GAN endpoint continues to use its MNIST generator, while both new models use RGB CIFAR-10 images (3 × 32 × 32). Generation is unconditional; no class label is requested.

## Setup and training

Run from `assignment4` with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m training.train_module8 --model ebm --epochs 5 --sampling-steps 20 --device mps
python -m training.train_module8 --model diffusion --epochs 10 --device mps
```

Choose `--device cpu` or `--device cuda` for another machine. The dataset is downloaded to `data/`; `--data-dir ../assignment2/data` reuses the existing CIFAR-10 data. Training uses the full 50,000-image training split, batch size 128, seed 42, and Adam with learning rate 0.0001. Labels are ignored. No test-split data are used for training. Longer runs can improve quality; loss alone does not establish perceptual quality.

Checkpoints, loss histories, elapsed times, and seeded 16-image sample grids are written to `models/`. Checkpoint loading includes the diffusion configuration, so the reverse schedule matches training. `torch.load(..., weights_only=True)` is used for loading.

## Architecture and gradient control

- `helper_lib/model.py`: `get_model('GAN')`, `get_model('EBM')`, `get_model('Diffusion')`.
- `helper_lib/trainer.py`: `train_gan`, `train_ebm`, `train_diffusion`.
- `helper_lib/generator.py`: `generate_samples(model, device='cpu', num_samples=10, seed=None)` dispatches by model type and returns images in [0, 1]. The model must already be on the selected device.
- `helper_lib/ebm.py`: four stride-2 convolutions with SiLU activations produce one unrestricted scalar energy per image. Lower energy means greater relative preference; there is no sigmoid probability output.
- `helper_lib/diffusion.py`: a U-Net with skip connections processes 32 → 16 → 8 spatial resolutions. Sinusoidal timestep embeddings condition residual blocks. The output is a three-channel noise estimate. A 1,000-step linear beta schedule ranges from 0.0001 to 0.02, with MSE noise-prediction training and ancestral DDPM sampling.

For EBM negative sampling, model parameters are frozen and `x.detach().requires_grad_(True)` makes pixels the differentiable input. `torch.autograd.grad(energy.sum(), x)` computes only pixel gradients. Each iteration detaches the graph; noise addition, clipped descent, and projection to [-1, 1] happen under `no_grad`. Sampling restores the prior parameter flags and train/eval mode even on error. Do not wrap EBM sampling in `inference_mode`; it needs input gradients. `no_grad` callers are supported because the sampler locally enables gradients.

EBM training uses a replay buffer of 1,024 images, with 5% fresh random initializations. Negative chains use step size 10, noise standard deviation 0.005, and coordinate gradient clipping at 0.03. The loss is

`mean(E(real)) - mean(E(fake.detach())) + 0.1 * (mean(E(real)^2) + mean(E(fake)^2))`.

Weight gradients are clipped to norm 0.1. This finite-step, clipped Langevin-style sampler follows Practical 1; independent noise/step-size choices and clipping mean it is an approximate negative sampler, not an exact equilibrium MCMC claim. Generation uses 60 steps from uniform noise. The recorded short training run uses 20 negative steps per batch to limit compute.

Diffusion forward corruption is performed without gradients, the noise target is fixed, and backpropagation updates the denoiser weights. Generation runs under `no_grad` and adds posterior noise only before the final reverse step. Sampling uses a local seeded generator without resetting the global random generator. API access is serialized per model because EBM sampling temporarily changes its gradient flags.

## Recorded training results

The EBM checkpoint was trained for five full CIFAR-10 epochs with 20 negative-sampling steps per batch. It took 298.3 seconds on the Mac GPU and ended with mean regularized contrastive loss -0.000292. **Its generated samples remain dominated by colored noise; the short run has not learned recognizable CIFAR-10 objects.** The checkpoint demonstrates the training and serving pipeline, not satisfactory visual quality. Improving it requires further training and tuning the negative chain, architecture, and optimization settings; the loss is not a calibrated image-quality score.

The diffusion checkpoint was trained for ten full epochs in 738.9 seconds on the Mac GPU. Mean training noise MSE decreased from 0.160984 to 0.042609. This is training error, not a held-out quality metric. The seeded sample grid shows coarse color and scene structure, but images are blurry and objects are often ambiguous; this short run does not reach high-quality CIFAR-10 generation.

See `models/ebm_training.json` and `models/ebm_samples.png` for the recorded run and unfiltered samples. The diffusion run is recorded separately in `models/diffusion_training.json`.

## API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
curl 'http://localhost:8000/generate/ebm?num_samples=4&seed=42' -o ebm.png
curl 'http://localhost:8000/generate/diffusion?num_samples=4&seed=42' -o diffusion.png
curl 'http://localhost:8000/generate/gan?num_samples=4&seed=42' -o gan.png
```

Open `http://localhost:8000/docs` for interactive documentation. `num_samples` accepts 1–16; `seed` is optional. Each response is a PNG grid. `GET /` lists loaded models. Unknown model names return 404, missing checkpoints return 503, and invalid query values return 422. The server never serves an untrained random model as a fallback. GAN weights are read from `../assignment3/models/mnist_generator.pt`. All inference runs on CPU; DDPM uses 1,000 network passes per request and can take several seconds.

## Validation

```bash
pip install pytest httpx
python -m pytest -q
```

Tests cover analytic pixel descent, unchanged weights and existing weight gradients during sampling, state restoration after errors, weight updates during training, absent training-input gradients, the forward-diffusion equation, seeded generation, checkpoint loading, image responses, and API errors. Test checkpoints are randomly initialized fixtures; they do not establish trained-image quality.

## Docker

Build from the repository root so the existing GAN checkpoint is included:

```bash
docker build -f assignment4/Dockerfile -t assignment4 .
docker run --rm -p 8000:80 assignment4
```

The included CIFAR-10 checkpoints avoid training or dataset downloads on server startup.

## Sources

The supplied Module 6 activity establishes the helper-library interface. Module 8 Practical 1 supplies the EBM sampler/loss approach; Practicals 2 and 3 supply time conditioning, noise-prediction training, and reverse diffusion. The implementation adapts these ideas to CIFAR-10 and the existing API.
