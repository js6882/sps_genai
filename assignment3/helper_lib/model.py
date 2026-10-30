import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.deconv2(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super(GAN, self).__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, x):
        return self.generator(x)


def get_model(model_name):
    if model_name == 'GAN':
        return GAN()
    raise ValueError(f'Unknown model: {model_name}')
