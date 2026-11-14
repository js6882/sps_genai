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
python -m training.train_module8 --model ebm --epochs 1 --sampling-steps 60 --smoothness 0 --sample-dir progress --device mps
python -m training.train_module8 --model ebm --epochs 1 --sampling-steps 60 --smoothness 5 --resume progress/latest.pt --sample-dir progress --device mps
python -m training.train_module8 --model diffusion --epochs 10 --device mps
```

Use `--device cpu` or `--device cuda` if needed. CIFAR-10 downloads to `data/`. If you already have it from Assignment 2, add `--data-dir ../assignment2/data`.

All training uses the 50,000-image training split, batch size 128, seed 42, and Adam with a learning rate of 0.0001. The class labels are not used. Training saves weights, losses, and a 16-image sample grid in `models/`. With `--sample-dir progress`, it also saves a checkpoint and sample grid after each EBM epoch. Use `--resume progress/latest.pt` to continue the weights and optimizer state; the replay buffer starts fresh.

## How the models work

The helper library keeps the same structure as the earlier assignment:

- `model.py` selects the GAN, EBM, or diffusion model.
- `trainer.py` provides the three training functions.
- `generator.py` generates images from each model and returns pixel values in [0, 1].

The EBM uses four 4×4 convolution layers with stride 2 to give each image an energy value. The channels are 32, 64, 128, and 128. To generate an image, it starts with random pixels and changes them to lower the energy. During sampling, the weights are frozen and `torch.autograd.grad` computes gradients for the pixels. During training, the sampled images are detached and the optimizer updates the weights instead.

The energy also includes a neighbor penalty:

```text
E(image) = CNN(image)
+ 5 * (mean(horizontal pixel differences^2) + mean(vertical pixel differences^2))
```

This penalizes high-frequency noise. The same energy is used during training and sampling, and its settings are saved with the weights.

The EBM loss is:

```text
mean(E(real)) - mean(E(fake))
+ 0.1 * (mean(E(real)^2) + mean(E(fake)^2))
```

Training reuses a buffer of 4,096 images, starting 5% of samples from fresh noise. Training uses 60 sampling steps per batch; generation uses 240. Step size is 10, noise standard deviation is 0.005, and pixel gradients are clipped to [-0.03, 0.03]. This is an approximate sampler. It needs pixel gradients, so it cannot run inside `torch.inference_mode()`.

The diffusion model uses a small U-Net with skip connections and timestep embeddings. It learns to predict the noise added to an image using MSE loss. Images go through 32 → 16 → 8 spatial sizes in the U-Net. Generation starts from Gaussian noise and runs 1,000 reverse steps, with beta increasing linearly from 0.0001 to 0.02. This generation loop does not need gradients.

## Results

The updated EBM used two full passes through CIFAR-10: one without the neighbor penalty, then one with its strength set to 5. The two passes took about 18 minutes on the Mac GPU. Compared with the earlier version, there is less speckled noise, but the images still have blocky patterns and do not show clear objects. `models/ebm_comparison.png` shows all 16 before-and-after samples using seed 123. Some of the smoothing comes from the neighbor penalty.

The diffusion model trained for 10 epochs in about 12 minutes. Its training MSE dropped from 0.160984 to 0.042609. The samples have some color and shape, but they are blurry and the objects are hard to identify.

The sample grids are `models/ebm_samples.png` and `models/diffusion_samples.png`. The JSON files in the same folder contain the losses and training settings.

## Tests

```bash
pip install pytest httpx
python -m pytest -q
```

The 10 tests check pixel and weight gradients, the neighbor penalty, repeatable sampling, saved settings, training checkpoints, and API responses.

## Docker

Run these commands from the repository root:

```bash
docker build -f assignment4/Dockerfile -t assignment4 .
docker run --rm -p 8000:80 assignment4
```

The Docker image uses CPU PyTorch. It does not install CUDA libraries.

After the server starts, check the endpoints from another terminal:

```bash
python assignment4/tests/smoke_api.py --url http://localhost:8000
```

This calls all three models, opens the returned PNGs, checks that seeds repeat, and checks invalid requests.

The final weights were tested in a Linux/arm64 CPU container on Docker Desktop. All three endpoints returned valid images with repeatable seeds; 404 and 422 responses were also checked. Results and weight checksums are in `models/deployment_check.json`.
