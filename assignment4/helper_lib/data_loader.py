from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(data_dir='./data', batch_size=128, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=download,
                               transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
