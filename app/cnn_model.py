"""Assignment 2 architecture and the shared training/inference transform."""
import torch
from torch import nn
from torchvision import transforms

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def image_transform():
    # CIFAR-10 is 32x32; the specified architecture requires 64x64.
    return transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
    ])


class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images):
        return self.layers(images)  # logits for CrossEntropyLoss
