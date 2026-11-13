# Assignment 4

This adds an energy-based model (EBM) and a diffusion model to the image generation API from Module 6. Both models are trained on CIFAR-10 and generate 32 × 32 color images. The GAN endpoint still uses the MNIST model from Assignment 3. None of the models lets you choose a class.

## Run the API

From the `assignment4` folder, using Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The trained weights are included in `models/`. The GAN weights are in `../assignment3/models/mnist_generator.pt`.

Open `http://localhost:8000/docs` to try the endpoints, or save images with:

```bash
curl 'http://localhost:8000/generate/ebm?num_samples=4&seed=42' -o ebm.png
curl 'http://localhost:8000/generate/diffusion?num_samples=4&seed=42' -o diffusion.png
curl 'http://localhost:8000/generate/gan?num_samples=4&seed=42' -o gan.png
```

`num_samples` can be 1–16. Setting `seed` gives the same result on repeated requests. Each endpoint returns a PNG grid. The API runs on CPU, so diffusion takes longer because it uses 1,000 denoising steps. If a model's weights are missing, its endpoint returns 503.

## Train the models

```bash
python -m training.train_module8 --model ebm --epochs 5 --sampling-steps 20 --device mps
python -m training.train_module8 --model diffusion --epochs 10 --device mps
```

Use `--device cpu` or `--device cuda` if needed. CIFAR-10 downloads to `data/`. If you already have it from Assignment 2, add `--data-dir ../assignment2/data`.

Both runs use all 50,000 training images, batch size 128, seed 42, and Adam with a learning rate of 0.0001. The class labels are not used. Training saves weights, losses, and a 16-image sample grid in `models/`.

## How the models work

The helper library keeps the same structure as the earlier assignment:

- `model.py` selects the GAN, EBM, or diffusion model.
- `trainer.py` provides the three training functions.
- `generator.py` generates images from each model and returns pixel values in [0, 1].

The EBM uses four convolution layers to give each image an energy value. To generate an image, it starts with random pixels and changes them to lower the energy. During sampling, the weights are frozen and `torch.autograd.grad` computes gradients for the pixels. During training, the sampled images are detached and the optimizer updates the weights instead.

The EBM loss is:

```text
mean(E(real)) - mean(E(fake))
+ 0.1 * (mean(E(real)^2) + mean(E(fake)^2))
```

Training reuses a buffer of 1,024 images, starting 5% of samples from fresh noise. The saved run uses 20 sampling steps per batch; generation uses 60. Step size is 10, noise standard deviation is 0.005, and pixel gradients are clipped to [-0.03, 0.03]. This is an approximate sampler. It needs pixel gradients, so it cannot run inside `torch.inference_mode()`.

The diffusion model uses a small U-Net with skip connections and timestep embeddings. It learns to predict the noise added to an image using MSE loss. Images go through 32 → 16 → 8 spatial sizes in the U-Net. Generation starts from Gaussian noise and runs 1,000 reverse steps, with beta increasing linearly from 0.0001 to 0.02. This generation loop does not need gradients.

## Results

The EBM trained for 5 epochs in about 5 minutes on the Mac GPU. Its final loss was -0.000292, but the samples are still mostly colored noise.

The diffusion model trained for 10 epochs in about 12 minutes. Its training MSE dropped from 0.160984 to 0.042609. The samples have some color and shape, but they are blurry and the objects are hard to identify.

The sample grids are `models/ebm_samples.png` and `models/diffusion_samples.png`. The JSON files in the same folder contain the losses and training settings.

## Tests

```bash
pip install pytest httpx
python -m pytest -q
```

The tests check gradient updates, sampling, repeatable results with a seed, loading saved weights, and API responses.

## Docker

Run these commands from the repository root:

```bash
docker build -f assignment4/Dockerfile -t assignment4 .
docker run --rm -p 8000:80 assignment4
```
