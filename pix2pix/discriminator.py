import torch
import torch.nn as nn


#
# class CNNBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, stride=2):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 4, stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.LeakyReLU(0.2)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class Discriminator(nn.Module):
#     def __init__(self, in_channel, features=[64, 128, 256, 512]):
#         super().__init__()
#         self.initial = nn.Sequential(
#             nn.Conv2d(in_channel * 2, features[0], kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2)
#         )
#         # [64,128,128] -> [128,64,64] -> [256, 32,32] -> [512,31,31]
#         layers = []
#         in_channels = features[0]
#         for feature in features[1:]:
#             layers.append(
#                 CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
#             )
#             in_channels = feature
#         layers.append(
#             nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)  # [512,31,31] -> [1,30,30]
#         )
#         layers.append(nn.Sigmoid())
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x, y):
#         x = torch.cat([x, y], dim=1)
#         x = self.initial(x)
#         x = self.model(x)
#         return x

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        use_bias = False

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc * 2, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x, y):
        """Standard forward."""
        input = torch.cat((x, y), dim=1)
        return self.model(input)


def test():
    x = torch.rand((1, 3, 256, 256))
    y = torch.rand((1, 3, 256, 256))
    model = Discriminator(input_nc=3)
    preds = model(x, y)
    print(preds.shape)


if __name__ == '__main__':
    test()
