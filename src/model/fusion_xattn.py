"""
Fusion Cross-Attention Model

A lightweight multimodal RGB–LiDAR–IMU fusion network
that introduces symmetric cross-modal attention prior
to feature fusion.

The model predicts binary BEV occupancy logits.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Encoder block consisting of

        Conv3×3
            ↓
        GroupNorm
            ↓
        ReLU
            ↓
        MaxPool2D

    Used by both the RGB and LiDAR encoder branches.
    """

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
                            num_groups=8,
                            num_channels=out_channels,
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
    """
    Decoder block consisting of

        Bilinear Upsampling
              ↓
           Conv3×3
              ↓
         GroupNorm
              ↓
            ReLU
    """

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
                            num_groups=8,
                            num_channels=out_channels,
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
    """
    Lightweight MLP used to encode IMU measurements into
    a compact feature representation for multimodal fusion.
    """

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
        
class CrossAttentionBlock(nn.Module):
    """
    Lightweight symmetric cross-attention between RGB and LiDAR features.

    To keep the computational cost manageable, spatial feature maps are
    first downsampled before attention is computed. The attended features
    are then upsampled back to the encoder resolution.

    Pipeline

        RGB Feature Map
              │
              ▼
         Adaptive Average Pool
              │
              ▼
         Flatten Tokens
              │
              ▼
       RGB ← BEV Attention
              │
              ▼
      Residual Connection
              │
              ▼
         Bilinear Upsample

    The same procedure is applied symmetrically for the BEV branch.
    """

    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        pool_size=8,
        dropout=0.1,
    ):
        super().__init__()

        # ---------------------------------------------
        # Reduce spatial resolution before attention
        # ---------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d(
            (8, 8)
        )

        self.upsample = nn.Upsample(
            scale_factor=pool_size,
            mode="bilinear",
            align_corners=False,
        )

        # ---------------------------------------------
        # Layer normalization
        # ---------------------------------------------
        self.rgb_norm = nn.LayerNorm(
            embed_dim,
        )

        self.bev_norm = nn.LayerNorm(
            embed_dim,
        )

        # ---------------------------------------------
        # Cross-attention
        # ---------------------------------------------
        self.rgb_to_bev = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.bev_to_rgb = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        rgb_feat,
        bev_feat,
    ):
        """
        Parameters
        ----------
        rgb_feat : Tensor
            RGB feature map of shape
            [B, C, H, W]

        bev_feat : Tensor
            LiDAR BEV feature map of shape
            [B, C, H, W]

        Returns
        -------
        rgb_feat : Tensor
            Cross-attended RGB feature map.

        bev_feat : Tensor
            Cross-attended BEV feature map.
        """

        batch_size, channels, H, W = rgb_feat.shape

        # ---------------------------------------------
        # Downsample feature maps
        # ---------------------------------------------
        rgb_small = self.pool(
            rgb_feat,
        )

        bev_small = self.pool(
            bev_feat,
        )

        pooled_height = rgb_small.shape[2]
        pooled_width = rgb_small.shape[3]

        # ---------------------------------------------
        # Convert feature maps into token sequences
        #
        # [B, C, H, W]
        #        ↓
        # [B, HW, C]
        # ---------------------------------------------
        rgb_tokens = rgb_small.flatten(2).transpose(
            1,
            2,
        )

        bev_tokens = bev_small.flatten(2).transpose(
            1,
            2,
        )

        # ---------------------------------------------
        # RGB attends to BEV
        # ---------------------------------------------
        rgb_attended, _ = self.rgb_to_bev(
            query=self.rgb_norm(
                rgb_tokens,
            ),
            key=bev_tokens,
            value=bev_tokens,
        )

        # ---------------------------------------------
        # BEV attends to RGB
        # ---------------------------------------------
        bev_attended, _ = self.bev_to_rgb(
            query=self.bev_norm(
                bev_tokens,
            ),
            key=rgb_tokens,
            value=rgb_tokens,
        )

        # ---------------------------------------------
        # Residual connections
        # ---------------------------------------------
        rgb_tokens = rgb_tokens + rgb_attended

        bev_tokens = bev_tokens + bev_attended

        # ---------------------------------------------
        # Restore feature maps
        # ---------------------------------------------
        rgb_feat = rgb_tokens.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            channels,
            pooled_height,
            pooled_width,
        )

        bev_feat = bev_tokens.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            channels,
            pooled_height,
            pooled_width,
        )

        # ---------------------------------------------
        # Restore encoder resolution
        # ---------------------------------------------
        rgb_feat = self.upsample(
            rgb_feat,
        )

        bev_feat = self.upsample(
            bev_feat,
        )

        return (
            rgb_feat,
            bev_feat,
        )
        
        
class FusionCrossAttentionModel(nn.Module):
    """
                  RGB Input                     LiDAR BEV
                    │                            │
                    ▼                            ▼
             RGB Encoder                  BEV Encoder
             (2 ConvBlocks)              (2 ConvBlocks)
                    │                            │
                    └──────────┬─────────────────┘
                               ▼
                 Cross Attention Block
          (RGB ↔ LiDAR Feature Interaction)
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             RGB Attended          BEV Attended
                    │                     │
                    └──────────┬──────────┘
                               ▼
                RGB + BEV Concatenation
                     (64 + 64 = 128)
                               │
                        IMU Projection
                               │
                     Spatial Broadcasting
                               │
                Concatenate IMU Features
                     (128 + 64 = 192)
                               │
                    Multimodal Fusion Conv
                               │
                         Decoder ×3
                               │
                      Prediction Head
                               │
                            Logits
    """

    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        # RGB encoder
        # -------------------------------------------------
        self.rgb_enc1 = ConvBlock(
            in_channels=3,
            out_channels=32,
        )

        self.rgb_enc2 = ConvBlock(
            in_channels=32,
            out_channels=64,
        )

        # -------------------------------------------------
        # LiDAR BEV encoder
        # -------------------------------------------------
        self.bev_enc1 = ConvBlock(
            in_channels=1,
            out_channels=32,
        )

        self.bev_enc2 = ConvBlock(
            in_channels=32,
            out_channels=64,
        )

        # -------------------------------------------------
        # IMU encoder
        # -------------------------------------------------
        self.imu_encoder = IMUEncoder()

        # -------------------------------------------------
        # Cross-modal attention
        # -------------------------------------------------
        self.cross_attention = CrossAttentionBlock(
            embed_dim=64,
            num_heads=4,
            pool_size=8,
            dropout=0.1,
        )

        # -------------------------------------------------
        # Feature fusion
        #
        # RGB : 64
        # BEV : 64
        # IMU : 64
        #
        # Total = 192 channels
        # -------------------------------------------------
        self.fusion = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels=192,
                            out_channels=128,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        "gn",
                        nn.GroupNorm(
                            num_groups=8,
                            num_channels=128,
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

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Prediction head
        #
        # Returns logits.
        # Sigmoid is applied externally during inference.
        # -------------------------------------------------
        self.head = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=1,
        )
        
    def forward(
        self,
        rgb,
        bev,
        imu,
    ):
        """
        Forward pass.

        Parameters
        ----------
        rgb : Tensor
            RGB image tensor of shape [B, 3, H, W].

        bev : Tensor
            LiDAR BEV tensor of shape [B, 1, H, W].

        imu : Tensor
            IMU tensor of shape [B, T, F].

        Returns
        -------
        Tensor
            BEV occupancy logits of shape
            [B, 1, H_out, W_out].
        """

        # -------------------------------------------------
        # RGB encoder
        # -------------------------------------------------
        rgb = self.rgb_enc1(rgb)
        rgb = self.rgb_enc2(rgb)

        # -------------------------------------------------
        # LiDAR encoder
        # -------------------------------------------------
        bev = self.bev_enc1(bev)
        bev = self.bev_enc2(bev)

        # -------------------------------------------------
        # Ensure identical spatial resolution
        # -------------------------------------------------
        if rgb.shape[-2:] != bev.shape[-2:]:

            bev = F.interpolate(
                bev,
                size=rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # -------------------------------------------------
        # Cross-modal attention
        # -------------------------------------------------
        rgb, bev = self.cross_attention(
            rgb,
            bev,
        )

        # -------------------------------------------------
        # Encode IMU
        # -------------------------------------------------
        imu = self.imu_encoder(imu)

        imu = imu.unsqueeze(-1).unsqueeze(-1)

        imu = imu.expand(
            -1,
            -1,
            rgb.size(2),
            rgb.size(3),
        )

        # -------------------------------------------------
        # Multimodal feature fusion
        # -------------------------------------------------
        fused = torch.cat(
            [
                rgb,
                bev,
                imu,
            ],
            dim=1,
        )

        fused = self.fusion(fused)

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------
        fused = self.dec1(fused)
        fused = self.dec2(fused)
        fused = self.dec3(fused)

        # -------------------------------------------------
        # Prediction logits
        # -------------------------------------------------
        logits = self.head(fused)

        return logits


def sanity_check():
    """
    Verify model construction, tensor dimensions,
    and forward propagation.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = FusionCrossAttentionModel().to(device)

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

    print(
        f"Model initialized on {device}"
    )

    print(
        f"RGB input shape      : {tuple(rgb.shape)}"
    )

    print(
        f"LiDAR input shape    : {tuple(bev.shape)}"
    )

    print(
        f"IMU input shape      : {tuple(imu.shape)}"
    )

    print(
        f"Output logits shape  : {tuple(logits.shape)}"
    )

    print()

    print(
        f"Total parameters     : {total_params:,}"
    )

    print(
        f"Trainable parameters : {trainable_params:,}"
    )

    print("\nSanity check passed.")


if __name__ == "__main__":

    sanity_check()