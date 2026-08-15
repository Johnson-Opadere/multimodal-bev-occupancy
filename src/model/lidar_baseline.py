from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict([
                (
                    "conv",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                ),
                ("gn", nn.GroupNorm(8, out_channels)),
                ("relu", nn.ReLU(inplace=True)),
                ("pool", nn.MaxPool2d(2)),
            ])
        )

    def forward(self, x):
        return self.features(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.refine = nn.Sequential(
            OrderedDict([
                (
                    "conv",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                ),
                ("gn", nn.GroupNorm(8, out_channels)),
                ("relu", nn.ReLU(inplace=True)),
            ])
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.refine(x)

        return x


class LiDARBaselineModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.dec1 = DecoderBlock(128, 64)
        self.dec2 = DecoderBlock(64, 32)
        self.dec3 = DecoderBlock(32, 16)

        self.head = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, bev):

        x = self.enc1(bev)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        logits = self.head(x)

        return logits


def sanity_check():

    model = LiDARBaselineModel()

    x = torch.randn(2, 1, 256, 256)

    logits = model(x)

    probs = torch.sigmoid(logits)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(model)
    print()

    print(f"Input shape       : {tuple(x.shape)}")
    print(f"Logits shape      : {tuple(logits.shape)}")
    print(f"Probability shape : {tuple(probs.shape)}")
    print(f"Total parameters  : {total_params:,}")
    print(f"Trainable params  : {trainable_params:,}")


if __name__ == "__main__":
    sanity_check()