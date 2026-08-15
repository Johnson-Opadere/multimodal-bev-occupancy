
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.cbam import CBAM


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
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
                    (
                        "gn",
                        nn.GroupNorm(
                            8,
                            out_channels,
                        ),
                    ),
                    (
                        "relu",
                        nn.ReLU(
                            inplace=True,
                        ),
                    ),
                ]
            )
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        x,
    ):

        x = self.features(x)

        x = self.pool(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.refine = nn.Sequential(
            OrderedDict(
                [
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
                    (
                        "gn",
                        nn.GroupNorm(
                            8,
                            out_channels,
                        ),
                    ),
                    (
                        "relu",
                        nn.ReLU(
                            inplace=True,
                        ),
                    ),
                ]
            )
        )

    def forward(
        self,
        x,
    ):

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
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(
                            input_dim,
                            hidden_dim,
                        ),
                    ),
                    (
                        "relu1",
                        nn.ReLU(
                            inplace=True,
                        ),
                    ),
                    (
                        "fc2",
                        nn.Linear(
                            hidden_dim,
                            output_dim,
                        ),
                    ),
                    (
                        "relu2",
                        nn.ReLU(
                            inplace=True,
                        ),
                    ),
                ]
            )
        )

    def forward(
        self,
        imu,
    ):

        imu = imu.view(
            imu.size(0),
            -1,
        )

        imu = self.mlp(imu)

        return imu
		
		
class FusionCBAMModel(nn.Module):

    def __init__(self):
        super().__init__()

        # -----------------------------
        # RGB encoder
        # -----------------------------
        self.rgb_enc1 = ConvBlock(
            in_channels=3,
            out_channels=32,
        )

        self.rgb_enc2 = ConvBlock(
            in_channels=32,
            out_channels=64,
        )

        # -----------------------------
        # LiDAR encoder
        # -----------------------------
        self.bev_enc1 = ConvBlock(
            in_channels=1,
            out_channels=32,
        )

        self.bev_enc2 = ConvBlock(
            in_channels=32,
            out_channels=64,
        )

        # -----------------------------
        # IMU encoder
        # -----------------------------
        self.imu_encoder = IMUEncoder(
            input_dim=60,
            hidden_dim=64,
            output_dim=64,
        )

        # -----------------------------
        # CBAM attention
        # -----------------------------
        self.cbam = CBAM(
            in_channels=192,
            reduction_ratio=8,
            kernel_size=7,
        )

        # -----------------------------
        # Fusion refinement
        # -----------------------------
        self.fusion = nn.Sequential(
            OrderedDict(
                [
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
                    (
                        "gn",
                        nn.GroupNorm(
                            8,
                            128,
                        ),
                    ),
                    (
                        "relu",
                        nn.ReLU(
                            inplace=True,
                        ),
                    ),
                ]
            )
        )

        # -----------------------------
        # Decoder
        # -----------------------------
        self.dec1 = DecoderBlock(
            in_channels=128,
            out_channels=64,
        )

        self.dec2 = DecoderBlock(
            in_channels=64,
            out_channels=32,
        )

        self.dec3 = DecoderBlock(
            in_channels=32,
            out_channels=16,
        )

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
		
    def forward(
        self,
        rgb,
        bev,
        imu,
    ):

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

        # -----------------------------
        # CBAM refinement
        # -----------------------------
        fused = self.cbam(fused)

        fused = self.fusion(fused)

        # -----------------------------
        # Decoder
        # -----------------------------
        fused = self.dec1(fused)

        fused = self.dec2(fused)

        fused = self.dec3(fused)

        logits = self.head(fused)

        return logits


def sanity_check():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = FusionCBAMModel().to(device)

    rgb = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    bev = torch.randn(
        2,
        1,
        200,
        200,
        device=device,
    )

    imu = torch.randn(
        2,
        10,
        6,
        device=device,
    )

    logits = model(
        rgb,
        bev,
        imu,
    )
    
    assert logits.ndim == 4
    assert logits.shape[1] == 1

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

    print(f"RGB shape            : {tuple(rgb.shape)}")
    print(f"LiDAR shape          : {tuple(bev.shape)}")
    print(f"IMU shape            : {tuple(imu.shape)}")
    print(f"Output logits shape  : {tuple(logits.shape)}")
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    print("\nSanity check passed.")


if __name__ == "__main__":

    sanity_check()