import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AttentionGate(nn.Module):
    """
    Attention Gate as in Attention U-Net (Oktay et al.)
    g: gating signal (from decoder, low-res)
    x: skip connection (from encoder, high-res)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (decoder), x: skip connection (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # Encoder
        self.first_conv = DoubleConv(in_channels, features[0])
        self.down1 = DoubleConv(features[0], features[1])
        self.down2 = DoubleConv(features[1], features[2])
        self.down3 = DoubleConv(features[2], features[3])
        self.down4 = DoubleConv(features[3], features[4])
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[3]//2)
        self.dec4 = DoubleConv(features[4], features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[2]//2)
        self.dec3 = DoubleConv(features[3], features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[1]//2)
        self.dec2 = DoubleConv(features[2], features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0]//2)
        self.dec1 = DoubleConv(features[1], features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.first_conv(x)      # [B, 64, H, W]
        x2 = self.down1(self.pool(x1))  # [B, 128, H/2, W/2]
        x3 = self.down2(self.pool(x2))  # [B, 256, H/4, W/4]
        x4 = self.down3(self.pool(x3))  # [B, 512, H/8, W/8]
        x5 = self.down4(self.pool(x4))  # [B, 1024, H/16, W/16]

        # Decoder + Attention
        d4 = self.up4(x5)  # [B, 512, H/8, W/8]
        x4_att = self.att4(g=d4, x=x4)
        d4 = torch.cat([x4_att, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)  # [B, 256, H/4, W/4]
        x3_att = self.att3(g=d3, x=x3)
        d3 = torch.cat([x3_att, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # [B, 128, H/2, W/2]
        x2_att = self.att2(g=d2, x=x2)
        d2 = torch.cat([x2_att, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # [B, 64, H, W]
        x1_att = self.att1(g=d1, x=x1)
        d1 = torch.cat([x1_att, d1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return out