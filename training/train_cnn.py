"""Run from the project root: python -m training.train_cnn --epochs 10."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from app.cnn_model import AssignmentCNN, CLASSES, image_transform


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return {"loss": loss_sum / count, "accuracy": correct / count, "examples": count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/cifar10_cnn.pt"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}", flush=True)
    dataset = CIFAR10(args.data_dir, train=True, download=True, transform=image_transform())
    train, validation = random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_accuracy = -1
    for epoch in range(1, args.epochs + 1):
        model.train()  # evaluate() switches to eval mode after each epoch.
        total_loss = correct = count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
        metrics = {"epoch": epoch, "train_loss": total_loss / count,
                   "train_accuracy": correct / count,
                   "validation": evaluate(model, validation_loader, criterion, device)}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["validation"]["accuracy"] > best_accuracy:
            best_accuracy = metrics["validation"]["accuracy"]
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "classes": list(CLASSES), "preprocessing": "resize64_to_tensor_v1",
                        "epoch": epoch, "validation_accuracy": best_accuracy}, args.output)
    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test = CIFAR10(args.data_dir, train=False, download=True, transform=image_transform())
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "device": device, "parameters": sum(p.numel() for p in model.parameters()),
              "history": history, "selected_epoch": checkpoint["epoch"],
              "test": evaluate(model, DataLoader(test, batch_size=args.batch_size), criterion, device)}
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test": report["test"], "checkpoint": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
