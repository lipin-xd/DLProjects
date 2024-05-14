import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),

        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)  # [1,3,256,256] -> [1,64,128,128]
        d2 = self.down1(d1)  # [1,64,128,128] -> [1,128, 64,64]
        d3 = self.down2(d2)  # [1,128,64,64] -> [1,256,32,32]
        d4 = self.down3(d3)  # [1,256,32,32] -> [1,512,16,16]
        d5 = self.down4(d4)  # [1,512,16,16] -> [1,512,8,8]
        # d6 = self.down5(d5)  # [1,512,8,8] -> [1,512,4,4]
        # d7 = self.down6(d6)  # [1,512,4,4] -> [1,512,2,2]
        # bottleneck = self.bottleneck(d7)  # [1,512,2,2] -> [1,512,1,1]
        # up1 = self.up1(bottleneck)  # [1,512,1,1] -> [1,512,2,2]
        # up2 = self.up2(torch.cat([up1, d7], 1))  # [1,1024,2,2] -> [1,512,4,4]
        # up3 = self.up3(torch.cat([up2, d6], 1))  # [1,1024,4,4] -> [1,512,8,8]
        up4 = self.up4(d5)  # [1,512,8,8] -> [1,512,16,16]
        up5 = self.up5(torch.cat([up4, d4], 1))  # [1,1024,16,16] -> [1,256, 32,32]
        up6 = self.up6(torch.cat([up5, d3], 1))  # [1,512,32,32] -> [1,128,64,64]
        up7 = self.up7(torch.cat([up6, d2], 1))  # [1,256,64,64] -> [1,64,128,128]
        return self.final_up(torch.cat([up7, d1], 1))  # [1,128,128,128] -> [1,3,256,256]


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
