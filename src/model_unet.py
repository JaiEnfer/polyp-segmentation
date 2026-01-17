import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels

        for f in features:
            self.downs.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        rev = list(features)[::-1]
        ch = features[-1]*2

        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(ch, f))  # ch = f(from up) + f(from skip)
            ch = f

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for upconv, up, skip in zip(self.upconvs, self.ups, skips):
            x = upconv(x)

            # handle odd-sized inputs safely
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = up(x)

        return self.final(x)
