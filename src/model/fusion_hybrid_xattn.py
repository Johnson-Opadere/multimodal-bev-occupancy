"""
fusion_hybrid_xattn.py
======================

Phase 4 — Hybrid X-Attention Fusion Network

This model combines:

    • RGB encoder
    • LiDAR BEV encoder
    • Symmetric cross-modal attention
    • Hybrid modality integration
    • IMU feature encoding
    • Multimodal fusion
    • Lightweight decoder

Unlike Phase 3c, the Hybrid X-Attention architecture
preserves both the original convolutional features
and the cross-attended features before multimodal
fusion. This allows the network to balance local
feature representations with global cross-modal
context.

The model predicts binary BEV occupancy logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Convolution Block
# ============================================================

class ConvBlock(nn.Module):
    """
    Convolution → GroupNorm → ReLU → MaxPool block.

    Used throughout the RGB and LiDAR BEV encoders.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),

            nn.ReLU(
                inplace=True,
            ),
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        x,
    ):

        x = self.features(
            x,
        )

        x = self.pool(
            x,
        )

        return x


# ============================================================
# Decoder Block
# ============================================================

class DecoderBlock(nn.Module):
    """
    Bilinear upsampling followed by convolutional
    feature refinement.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.refine = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),

            nn.ReLU(
                inplace=True,
            ),
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

        x = self.refine(
            x,
        )

        return x


# ============================================================
# IMU Encoder
# ============================================================

class IMUEncoder(nn.Module):
    """
    Lightweight MLP for IMU feature encoding.

    Input
    -----
    IMU sequence

        Shape:
            (B, 10, 6)

    Output
    ------
    IMU embedding

        Shape:
            (B, 64)
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                60,
                64,
            ),

            nn.ReLU(
                inplace=True,
            ),

            nn.Linear(
                64,
                64,
            ),

            nn.ReLU(
                inplace=True,
            ),
        )

    def forward(
        self,
        imu,
    ):

        imu = imu.view(
            imu.size(0),
            -1,
        )

        imu = self.encoder(
            imu,
        )

        return imu
		

# ============================================================
# Cross-Attention Block
# ============================================================

class CrossAttentionBlock(nn.Module):
    """
    Symmetric cross-modal attention between RGB and
    LiDAR feature representations.

    The feature maps are first compressed using
    adaptive average pooling to reduce computational
    cost before bidirectional multi-head attention is
    applied.

    Pipeline
    --------
        RGB Feature Map
               │
        Adaptive AvgPool
               │
        RGB Tokens
               │
          RGB → LiDAR
        Multi-Head Attention
               │
        Residual Addition
               │
        Upsample to Original Size
               │
        RGB Attended Features

    The same procedure is performed in the opposite
    direction (LiDAR → RGB).

    Parameters
    ----------
    channels : int
        Feature dimension.

    pool_size : int, default=8
        Spatial resolution used for attention.

    num_heads : int, default=4
        Number of attention heads.
    """

    def __init__(
        self,
        channels,
        pool_size=8,
        num_heads=4,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(
            (pool_size, pool_size),
        )

        self.rgb_norm = nn.LayerNorm(
            channels,
        )

        self.bev_norm = nn.LayerNorm(
            channels,
        )

        self.rgb_to_bev = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

        self.bev_to_rgb = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        rgb_features,
        bev_features,
    ):
        """
        Apply symmetric cross attention.

        Parameters
        ----------
        rgb_features : Tensor
            Shape:
                (B, C, H, W)

        bev_features : Tensor
            Shape:
                (B, C, H, W)

        Returns
        -------
        tuple(Tensor, Tensor)

            rgb_attended
                (B, C, H, W)

            bev_attended
                (B, C, H, W)
        """

        B, C, H_rgb, W_rgb = rgb_features.shape
        _, _, H_bev, W_bev = bev_features.shape

        # ---------------------------------------------------
        # Adaptive average pooling
        # ---------------------------------------------------
        rgb_small = self.pool(
            rgb_features,
        )

        bev_small = self.pool(
            bev_features,
        )

        # ---------------------------------------------------
        # Convert feature maps into token sequences
        # ---------------------------------------------------
        rgb_tokens = rgb_small.flatten(
            2,
        ).transpose(
            1,
            2,
        )

        bev_tokens = bev_small.flatten(
            2,
        ).transpose(
            1,
            2,
        )

        rgb_tokens = self.rgb_norm(
            rgb_tokens,
        )

        bev_tokens = self.bev_norm(
            bev_tokens,
        )

        # ---------------------------------------------------
        # RGB attends to LiDAR
        # ---------------------------------------------------
        rgb_delta, _ = self.rgb_to_bev(
            query=rgb_tokens,
            key=bev_tokens,
            value=bev_tokens,
        )

        # ---------------------------------------------------
        # LiDAR attends to RGB
        # ---------------------------------------------------
        bev_delta, _ = self.bev_to_rgb(
            query=bev_tokens,
            key=rgb_tokens,
            value=rgb_tokens,
        )

        # ---------------------------------------------------
        # Residual updates
        # ---------------------------------------------------
        rgb_tokens = rgb_tokens + rgb_delta

        bev_tokens = bev_tokens + bev_delta

        # ---------------------------------------------------
        # Restore spatial feature maps
        # ---------------------------------------------------
        rgb_small = rgb_tokens.transpose(
            1,
            2,
        ).reshape(
            B,
            C,
            rgb_small.shape[-2],
            rgb_small.shape[-1],
        )

        bev_small = bev_tokens.transpose(
            1,
            2,
        ).reshape(
            B,
            C,
            bev_small.shape[-2],
            bev_small.shape[-1],
        )

        # ---------------------------------------------------
        # Upsample back to the original resolution
        # ---------------------------------------------------
        rgb_attended = F.interpolate(
            rgb_small,
            size=(H_rgb, W_rgb),
            mode="bilinear",
            align_corners=False,
        )

        bev_attended = F.interpolate(
            bev_small,
            size=(H_bev, W_bev),
            mode="bilinear",
            align_corners=False,
        )

        return (
            rgb_attended,
            bev_attended,
        )
		
		
# ============================================================
# Hybrid Fusion Block
# ============================================================

class HybridFusionBlock(nn.Module):
    """
    Hierarchical hybrid multimodal fusion.

    The block performs fusion in two stages.

    Stage 1
    -------
    Learn modality-specific hybrid representations by
    combining original encoder features with their
    corresponding cross-attended features.

        RGB Original
            +
        RGB Attended
            ↓
        RGB Hybrid
            ↓
      Residual Refinement

        LiDAR Original
            +
        LiDAR Attended
            ↓
        LiDAR Hybrid
            ↓
      Residual Refinement

    Stage 2
    -------
    Fuse the hybrid RGB representation, hybrid LiDAR
    representation, and IMU embedding.

        RGB Hybrid
            +
        LiDAR Hybrid
            +
        IMU
            ↓
      Multimodal Fusion

    Residual refinement preserves useful low-level
    convolutional features learned by the encoders while
    allowing the network to incorporate cross-modal
    information.

    Parameters
    ----------
    feature_channels : int, default=64
        Number of channels produced by each visual encoder.

    imu_channels : int, default=64
        IMU embedding dimension.

    fusion_channels : int, default=128
        Number of output channels after multimodal fusion.
    """

    def __init__(
        self,
        feature_channels=64,
        imu_channels=64,
        fusion_channels=128,
    ):
        super().__init__()

        # ---------------------------------------------------
        # RGB hybrid refinement
        # ---------------------------------------------------
        self.rgb_hybrid = nn.Sequential(

            nn.Conv2d(
                feature_channels * 2,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                8,
                feature_channels,
            ),

            nn.ReLU(
                inplace=True,
            ),
        )

        # ---------------------------------------------------
        # LiDAR hybrid refinement
        # ---------------------------------------------------
        self.bev_hybrid = nn.Sequential(

            nn.Conv2d(
                feature_channels * 2,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                8,
                feature_channels,
            ),

            nn.ReLU(
                inplace=True,
            ),
        )

        # ---------------------------------------------------
        # Final multimodal fusion
        #
        # RGB Hybrid   : 64
        # LiDAR Hybrid : 64
        # IMU          : 64
        #
        # Total = 192 channels
        # ---------------------------------------------------
        self.fusion = nn.Sequential(

            nn.Conv2d(
                feature_channels * 2 + imu_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                8,
                fusion_channels,
            ),

            nn.ReLU(
                inplace=True,
            ),

            nn.Dropout2d(
                p=0.10,
            ),
        )

    def forward(
        self,
        rgb_original,
        rgb_attended,
        bev_original,
        bev_attended,
        imu_features,
    ):
        """
        Perform hierarchical hybrid fusion.

        Parameters
        ----------
        rgb_original : Tensor
            Original RGB encoder features.

        rgb_attended : Tensor
            Cross-attended RGB features.

        bev_original : Tensor
            Original LiDAR BEV encoder features.

        bev_attended : Tensor
            Cross-attended LiDAR features.

        imu_features : Tensor
            Broadcast IMU embedding.

        Returns
        -------
        Tensor

            Fused multimodal feature representation.

            Shape:
                (B, fusion_channels, H, W)
        """
        
        # ---------------------------------------------------
        # Ensure spatial dimensions match
        # ---------------------------------------------------
        if rgb_attended.shape[-2:] != rgb_original.shape[-2:]:

            rgb_attended = F.interpolate(
                rgb_attended,
                size=rgb_original.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if bev_attended.shape[-2:] != bev_original.shape[-2:]:

            bev_attended = F.interpolate(
                bev_attended,
                size=bev_original.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        # ---------------------------------------------------
        # RGB hybrid refinement
        # ---------------------------------------------------
        rgb_hybrid = torch.cat(
            [
                rgb_original,
                rgb_attended,
            ],
            dim=1,
        )

        rgb_hybrid = self.rgb_hybrid(
            rgb_hybrid,
        )

        # Residual refinement
        rgb_hybrid = rgb_hybrid + rgb_original

        # ---------------------------------------------------
        # LiDAR hybrid refinement
        # ---------------------------------------------------
        bev_hybrid = torch.cat(
            [
                bev_original,
                bev_attended,
            ],
            dim=1,
        )

        bev_hybrid = self.bev_hybrid(
            bev_hybrid,
        )

        # Residual refinement
        bev_hybrid = bev_hybrid + bev_original
        
        # ---------------------------------------------------
        # Final multimodal fusion
        # ---------------------------------------------------
        fused = torch.cat(
            [
                rgb_hybrid,
                bev_hybrid,
                imu_features,
            ],
            dim=1,
        )

        fused = self.fusion(
            fused,
        )

        return fused
        
		
# ============================================================
# Fusion Hybrid X-Attention Model
# ============================================================

class FusionHybridXAttentionModel(nn.Module):
    """
    Hybrid X-Attention multimodal fusion network.

    Pipeline
    --------
        RGB Image
             │
        RGB Encoder
             │
     Original RGB Features
             │
             ├──────────────────────┐
             │                      │
             ▼                      │
        Cross Attention             │
             │                      │
             ▼                      │
      RGB Attended Features         │
                                    │
        LiDAR BEV                   │
             │                      │
       LiDAR BEV Encoder                │
             │                      │
 Original BEV Features            │
             │                      │
             ├──────────────────────┘
             ▼
     BEV Attended Features

             │
             ▼
      Hybrid Fusion Block
             │
     RGB Hybrid Features
     LiDAR Hybrid Features
     IMU Embedding
             │
             ▼
      Multimodal Fusion
             │
             ▼
          Decoder
             │
             ▼
      BEV Occupancy Logits
    """

    def __init__(
        self,
    ):
        super().__init__()

        # ---------------------------------------------------
        # RGB encoder
        # ---------------------------------------------------
        self.rgb_encoder = nn.Sequential(

            ConvBlock(
                3,
                32,
            ),

            ConvBlock(
                32,
                64,
            ),
        )

        # ---------------------------------------------------
        # LiDAR BEV encoder
        # ---------------------------------------------------
        self.bev_encoder = nn.Sequential(

            ConvBlock(
                1,
                32,
            ),

            ConvBlock(
                32,
                64,
            ),
        )

        # ---------------------------------------------------
        # IMU encoder
        # ---------------------------------------------------
        self.imu_encoder = IMUEncoder()

        # ---------------------------------------------------
        # Cross-modal attention
        # ---------------------------------------------------
        self.cross_attention = CrossAttentionBlock(
            channels=64,
            pool_size=8,
            num_heads=4,
        )

        # ---------------------------------------------------
        # Hybrid multimodal fusion
        # ---------------------------------------------------
        self.hybrid_fusion = HybridFusionBlock(
            feature_channels=64,
            imu_channels=64,
            fusion_channels=128,
        )

        # ---------------------------------------------------
        # Decoder
        # ---------------------------------------------------
        self.decoder = nn.Sequential(

            DecoderBlock(
                128,
                64,
            ),

            DecoderBlock(
                64,
                32,
            ),

            DecoderBlock(
                32,
                16,
            ),

            nn.Conv2d(
                16,
                1,
                kernel_size=1,
            ),
        )
		
		
    def forward(
        self,
        rgb,
        bev,
        imu,
    ):
        """
        Forward propagation.

        Parameters
        ----------
        rgb : Tensor
            RGB image.

            Shape:
                (B, 3, H, W)

        bev : Tensor
            LiDAR BEV occupancy map.

            Shape:
                (B, 1, H, W)

        imu : Tensor
            IMU sequence.

            Shape:
                (B, 10, 6)

        Returns
        -------
        Tensor

            Binary BEV occupancy logits.

            Shape:
                (B, 1, H_out, W_out)
        """

        # ---------------------------------------------------
        # Encode RGB and LiDAR modalities
        # ---------------------------------------------------
        rgb_features = self.rgb_encoder(
            rgb,
        )

        bev_features = self.bev_encoder(
            bev,
        )
        
        # ---------------------------------------------------
        # Align encoder feature resolutions.
        #
        # RGB and LiDAR BEV inputs may produce feature maps
        # with different spatial resolutions. The LiDAR BEV
        # features are resized to the RGB feature resolution
        # before cross-modal attention and hybrid fusion.
        # ---------------------------------------------------
        if rgb_features.shape[-2:] != bev_features.shape[-2:]:

            bev_features = F.interpolate(
                bev_features,
                size=rgb_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # ---------------------------------------------------
        # Encode IMU sequence
        # ---------------------------------------------------
        imu_features = self.imu_encoder(
            imu,
        )

        # ---------------------------------------------------
        # Cross-modal attention
        #
        # Produces attention-enhanced feature maps while
        # preserving the original encoder features.
        # ---------------------------------------------------
        rgb_attended, bev_attended = (
            self.cross_attention(
                rgb_features,
                bev_features,
            )
        )

        # ---------------------------------------------------
        # Broadcast IMU embedding to match the spatial
        # resolution of the visual feature maps.
        # ---------------------------------------------------
        imu_features = imu_features.unsqueeze(
            -1,
        ).unsqueeze(
            -1,
        )

        imu_features = imu_features.expand(
            -1,
            -1,
            rgb_features.shape[-2],
            rgb_features.shape[-1],
        )

        # ---------------------------------------------------
        # Hybrid multimodal fusion
        #
        # Stage 1:
        #   RGB Original + RGB Attended
        #
        #   LiDAR Original + LiDAR Attended
        #
        # Stage 2:
        #   RGB Hybrid + LiDAR Hybrid + IMU
        # ---------------------------------------------------
        fused_features = self.hybrid_fusion(
            rgb_original=rgb_features,
            rgb_attended=rgb_attended,
            bev_original=bev_features,
            bev_attended=bev_attended,
            imu_features=imu_features,
        )

        # ---------------------------------------------------
        # Decode fused representation into BEV occupancy
        # logits.
        # ---------------------------------------------------
        logits = self.decoder(
            fused_features,
        )

        return logits
		
		
# ============================================================
# Sanity Check
# ============================================================

def sanity_check():
    """
    Perform a forward-pass sanity check.

    This routine verifies:

        • Input tensor compatibility
        • Forward propagation
        • Output tensor dimensions
        • Output datatype
        • Total parameter count
        • Trainable parameter count

    No training is performed.
    """

    model = FusionHybridXAttentionModel()

    # ---------------------------------------------------
    # Dummy inputs
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # Forward pass
    # ---------------------------------------------------
    logits = model(
        rgb,
        bev,
        imu,
    )

    # ---------------------------------------------------
    # Parameter statistics
    # ---------------------------------------------------
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # ---------------------------------------------------
    # Display summary
    # ---------------------------------------------------
    print()

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
        f"Output shape         : {tuple(logits.shape)}"
    )

    print(
        f"Output dtype         : {logits.dtype}"
    )

    print(
        f"Total parameters     : {total_params:,}"
    )

    print(
        f"Trainable parameters : {trainable_params:,}"
    )

    # ---------------------------------------------------
    # Assertions
    # ---------------------------------------------------
    assert logits.ndim == 4

    assert logits.shape[0] == rgb.shape[0]

    assert logits.shape[1] == 1

    print()

    print(
        "Sanity check passed."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    sanity_check()