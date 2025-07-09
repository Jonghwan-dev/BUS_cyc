import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for idx in range(len(features)-1):
            self.downs.append(DoubleConv(features[idx], features[idx+1]))
        self.pool = nn.MaxPool2d(2)
        for idx in range(len(features)-1, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(features[idx], features[idx-1], kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(features[idx], features[idx-1]))
        self.first_conv = DoubleConv(in_channels, features[0])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.first_conv(x)
        skip_connections.append(x)
        for down in self.downs:
            x = self.pool(x)
            x = down(x)
            skip_connections.append(x)
        x = skip_connections.pop()  # bottleneck
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections.pop()  # <--- 여기 pop()으로 수정!
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)