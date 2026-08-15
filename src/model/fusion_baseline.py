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


class IMUEncoder(nn.Module):

    def __init__(
        self,
        input_dim=60,
        hidden_dim=64,
        output_dim=64,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            OrderedDict([
                (
                    "fc1",
                    nn.Linear(
                        input_dim,
                        hidden_dim,
                    ),
                ),
                ("relu1", nn.ReLU(inplace=True)),
                (
                    "fc2",
                    nn.Linear(
                        hidden_dim,
                        output_dim,
                    ),
                ),
                ("relu2", nn.ReLU(inplace=True)),
            ])
        )

    def forward(self, imu):

        imu = imu.view(imu.size(0), -1)

        imu = self.mlp(imu)

        return imu     
        
		
class FusionBaselineModel(nn.Module):

    def __init__(self):
        super().__init__()
        
        # -----------------------------
        # RGB encoder
        # -----------------------------
        self.rgb_enc1 = ConvBlock(3, 32)
        self.rgb_enc2 = ConvBlock(32, 64)

        # -----------------------------
        # LiDAR encoder
        # -----------------------------
        self.bev_enc1 = ConvBlock(1, 32)
        self.bev_enc2 = ConvBlock(32, 64)

        # -----------------------------
        # IMU encoder
        # -----------------------------
        self.imu_encoder = IMUEncoder()

        # -----------------------------
        # Fusion refinement
        # -----------------------------
        self.fusion = nn.Sequential(
            OrderedDict([
                (
                    "conv",
                    nn.Conv2d(
                        in_channels=192,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                ),
                ("gn", nn.GroupNorm(8, 128)),
                ("relu", nn.ReLU(inplace=True)),
            ])
        )

        # -----------------------------
        # Decoder
        # -----------------------------
        self.dec1 = DecoderBlock(128, 64)
        self.dec2 = DecoderBlock(64, 32)
        self.dec3 = DecoderBlock(32, 16)

        # -----------------------------
        # Prediction head
        # -----------------------------
        self.head = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
				
    def forward(self, rgb, bev, imu):

        # -----------------------------
        # RGB branch
        # -----------------------------
        rgb = self.rgb_enc1(rgb)
        rgb = self.rgb_enc2(rgb)

        # -----------------------------
        # LiDAR branch
        # -----------------------------
        bev = self.bev_enc1(bev)
        bev = self.bev_enc2(bev)

        # -----------------------------
        # Match spatial resolutions
        # -----------------------------
        if bev.shape[-2:] != rgb.shape[-2:]:

            bev = F.interpolate(
                bev,
                size=rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # -----------------------------
        # IMU branch
        # -----------------------------
        imu = self.imu_encoder(imu)

        imu = imu.unsqueeze(-1).unsqueeze(-1)

        imu = imu.expand(
            -1,
            -1,
            rgb.size(2),
            rgb.size(3),
        )

        # -----------------------------
        # Feature fusion
        # -----------------------------
        fused = torch.cat(
            [
                rgb,
                bev,
                imu,
            ],
            dim=1,
        )

        fused = self.fusion(fused)

        # -----------------------------
        # Decoder
        # -----------------------------
        x = self.dec1(fused)
        x = self.dec2(x)
        x = self.dec3(x)

        logits = self.head(x)

        return logits


def sanity_check():

    model = FusionBaselineModel()

    rgb = torch.randn(
        2,
        3,
        256,
        256,
    )

    bev = torch.randn(
        2,
        1,
        256,
        256,
    )

    imu = torch.randn(
        2,
        10,
        6,
    )

    logits = model(
        rgb,
        bev,
        imu,
    )

    probs = torch.sigmoid(logits)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(model)
    print()

    print(f"RGB input shape     : {tuple(rgb.shape)}")
    print(f"LiDAR input shape   : {tuple(bev.shape)}")
    print(f"IMU input shape     : {tuple(imu.shape)}")
    print(f"Logits shape        : {tuple(logits.shape)}")
    print(f"Probability shape   : {tuple(probs.shape)}")
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable params    : {trainable_params:,}")


if __name__ == "__main__":

    sanity_check()