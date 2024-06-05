import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channel, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channel * 2, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        # [64,128,128] -> [128,64,64] -> [256, 32,32] -> [512,31,31]
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)  # [512,31,31] -> [1,30,30]
        )
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.rand((1, 3, 256, 256))
    y = torch.rand((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)


if __name__ == '__main__':
    test()