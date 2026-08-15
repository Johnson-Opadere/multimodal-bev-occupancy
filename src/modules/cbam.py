
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):

    def __init__(
        self,
        in_channels,
        reduction_ratio=8,
    ):
        super().__init__()

        reduced_channels = max(
            1,
            in_channels // reduction_ratio,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.expand = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        x,
    ):

        avg_features = self.avg_pool(x)
        max_features = self.max_pool(x)

        avg_features = self.expand(
            F.relu(
                self.reduce(avg_features),
                inplace=True,
            )
        )

        max_features = self.expand(
            F.relu(
                self.reduce(max_features),
                inplace=True,
            )
        )

        attention = torch.sigmoid(
            avg_features + max_features
        )

        return x * attention        
        
        
class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size=7,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(
        self,
        x,
    ):

        avg_features = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_features, _ = torch.max(
            x,
            dim=1,
            keepdim=True,
        )

        features = torch.cat(
            [
                avg_features,
                max_features,
            ],
            dim=1,
        )

        attention = torch.sigmoid(
            self.conv(features)
        )

        return x * attention


class CBAM(nn.Module):

    def __init__(
        self,
        in_channels,
        reduction_ratio=8,
        kernel_size=7,
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size,
        )

    def forward(
        self,
        x,
    ):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x        
        
        
def sanity_check():

    module = CBAM(
        in_channels=192,
    )

    x = torch.randn(
        2,
        192,
        64,
        64,
    )

    y = module(x)

    total_params = sum(
        p.numel()
        for p in module.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )

    print(module)
    print()

    print(f"Input shape         : {tuple(x.shape)}")
    print(f"Output shape        : {tuple(y.shape)}")
    print(f"Input dtype         : {x.dtype}")
    print(f"Output dtype        : {y.dtype}")
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable params    : {trainable_params:,}")

    assert y.shape == x.shape, (
        "CBAM should preserve the input tensor shape."
    )

    print("\nSanity check passed.")


if __name__ == "__main__":

    sanity_check()