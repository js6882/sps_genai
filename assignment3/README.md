# Assignment 3

## Part 1: GAN architecture

The generator maps 100 noise values to 128 x 7 x 7, then uses two transposed convolutions to produce a 1 x 28 x 28 image. The discriminator uses two convolutions and a linear layer, followed by sigmoid to return a real/fake probability.

The helper library separates the model, data loader, trainer, and image generation.

## Part 2: MNIST training and API

Run the following commands from `assignment3`.

```bash
python -m pip install -r requirements.txt
python -m training.train_gan --epochs 20
```

Training uses all 60,000 MNIST training images, normalized to [-1, 1], binary cross-entropy, and separate Adam optimizers with learning rate 0.0002 and betas (0.5, 0.999). Labels are not used. The model generates random digits; it does not select a requested digit class.

The generator weights are saved in `models/mnist_generator.pt`. Epoch losses are saved in `models/training.json`. Sample grids are saved after each epoch. `models/generated_digits.png` contains 64 samples from the trained generator using seed 123.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs or save a grid with:

```bash
curl "http://localhost:8000/generate/gan?num_samples=16&seed=42" -o digits.png
```

`num_samples` accepts 1 to 64. `seed` is optional. The endpoint returns a PNG image. The saved generator is loaded once when the server starts.

The included checkpoint was trained for 20 epochs on all 60,000 images using the Mac GPU. Final average losses were 0.2627 for the discriminator and 3.4089 for the generator. Some generated digits remain distorted.

## Docker

The trained weights are included, so the container does not need to download MNIST or train at startup.

```bash
docker build -t assignment3 .
docker run --rm -p 8000:80 assignment3
```

## Tests

```bash
python -m pip install pytest httpx
python -m pytest -q
```
