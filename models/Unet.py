import torch
import torch.nn as nn
from einops import rearrange


class DoubleConv(nn.Module):
    """Some Information about DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.doubleconv(x)

    def __str__(self) -> str:
        return f"DoubleConv"


class Unet(nn.Module):
    """Some Information about Unet"""

    def __init__(self, in_channels: int, n_classes: int, base_dim: int, depth: int):
        super(Unet, self).__init__()
        self.base_dim = base_dim
        self.depth = depth
        self.inc = DoubleConv(in_channels, self.base_dim)
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()
        self.classifier = nn.Sequential(
            nn.Conv2d(self.base_dim, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weight()

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'c h w -> 1 c h w')
        features = [x := self.inc(x)]
        for m in self.encoder:
            x = m(x)
            if 'DoubleConv' == str(m):
                features.append(x)
        features.pop()
        for m in self.decoder:
            x = m(x)
            if 'DoubleConv' != str(m):
                x = torch.cat([features.pop(), x], axis=-3)
        x = self.classifier(x)
        return x

    def make_encoder(self):
        layers = []
        for _ in range(self.depth):
            layers += [
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(self.base_dim, self.base_dim*2)
            ]
            self.base_dim *= 2
        return nn.ModuleList(layers)

    def make_decoder(self):
        layers = []
        for _ in range(self.depth):
            layers += [
                nn.ConvTranspose2d(self.base_dim, self.base_dim//2,
                                   kernel_size=2, stride=2),
                DoubleConv(self.base_dim, self.base_dim//2)
            ]
            self.base_dim //= 2
        return nn.ModuleList(layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
