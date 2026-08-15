"""
rgb_baseline.py
===============

Phase 2 — RGB Baseline Network

This module implements the RGB-only baseline model used in
Project 3 for RGB-to-BEV occupancy prediction.

Architecture
------------
RGB Image
    ↓
Encoder
    ↓
Latent Feature Representation
    ↓
Decoder
    ↓
BEV Occupancy Logits

Design Philosophy
-----------------
This lightweight encoder-decoder CNN serves as the single-modality
reference model for subsequent multimodal fusion experiments.

Modernizations
--------------
- Group Normalization for stable small-batch training
- Bilinear upsampling + convolution decoder
- Logit output compatible with BCEWithLogitsLoss

Training Loss
-------------
BCEWithLogitsLoss + Dice Loss

Inference
---------
torch.sigmoid(logits)
        ↓
Occupancy probabilities
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Encoder building block.

    Performs:

        Conv → GroupNorm → ReLU → MaxPool

    Each block extracts increasingly abstract visual features
    while reducing spatial resolution by a factor of two.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convolutional feature extraction followed by
        # normalization, activation, and spatial downsampling.        
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
                ("pool", nn.MaxPool2d(kernel_size=2, stride=2)),
            ])
        )

    def forward(self, x):
        """
        Forward pass through one encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Downsampled feature map.
        """
        return self.features(x)


class DecoderBlock(nn.Module):
    """
    Decoder building block.

    Performs:

        Bilinear Upsampling
            ↓
        Conv → GroupNorm → ReLU

    Each block progressively reconstructs the spatial
    resolution of the BEV occupancy map.
    """

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
        
        # Bilinear interpolation restores spatial resolution.
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        
        # Convolution refines the upsampled feature map.
        x = self.refine(x)

        return x


class RGBBaselineModel(nn.Module):
    """
    RGB-only encoder-decoder baseline.

    Input
    -----
    RGB image

        Shape: [B,3,H,W]

    Output
    ------
    BEV occupancy logits

        Shape: [B,1,H,W]

    Notes
    -----
    This model establishes the RGB-only reference
    for evaluating multimodal sensor fusion in
    later project phases.
    """

    def __init__(self):
        super().__init__()

        # ---------------------------------------------------
        # Encoder
        #
        # Extract hierarchical visual features from the
        # input RGB image.
        # ---------------------------------------------------
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        # ---------------------------------------------------
        # Decoder
        #
        # Progressively reconstruct the BEV feature map
        # from the compressed latent representation.
        # ---------------------------------------------------
        self.dec1 = DecoderBlock(128, 64)
        self.dec2 = DecoderBlock(64, 32)
        self.dec3 = DecoderBlock(32, 16)

        # ---------------------------------------------------
        # Prediction Head
        #
        # Produce occupancy logits for each BEV cell.
        # Sigmoid is intentionally omitted and applied
        # outside the model during inference.
        # ---------------------------------------------------
        self.head = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, rgb):
        """
        Forward propagation.

        Parameters
        ----------
        rgb : torch.Tensor

            RGB image tensor of shape

            [B,3,H,W]

        Returns
        -------
        torch.Tensor

            BEV occupancy logits

            [B,1,H,W]
        """
        
        # Encode RGB appearance.
        x = self.enc1(rgb)
        x = self.enc2(x)
        x = self.enc3(x)
        
        # Decode latent representation.
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        
        # Predict occupancy logits.
        logits = self.head(x)

        return logits


def sanity_check():
    """
    Run a lightweight model sanity check.

    Verifies

    - model construction
    - tensor dimensions
    - parameter counts
    - forward propagation
    """

    model = RGBBaselineModel()
    
    # ---------------------------------------------------
    # Generate dummy RGB input.
    # ---------------------------------------------------
    x = torch.randn(2, 3, 256, 256)

    # ---------------------------------------------------
    # Forward propagation.
    # ---------------------------------------------------
    logits = model(x)

    # ---------------------------------------------------
    # Convert logits to probabilities.
    # ---------------------------------------------------
    probs = torch.sigmoid(logits)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
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