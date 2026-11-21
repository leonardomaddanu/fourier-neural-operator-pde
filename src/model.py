"""Two-dimensional Fourier neural operator with shared weights across resolutions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class SpectralConv2d(nn.Module):
    """Apply learned channel mixing to retained two-dimensional Fourier modes."""

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels), ("out_channels", out_channels),
            ("modes1", modes1), ("modes2", modes2),
        ):
            _positive_integer(value, name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(
            scale * torch.complex(torch.rand(shape, dtype=torch.float32),
                                  torch.rand(shape, dtype=torch.float32)),
        )
        self.weights2 = nn.Parameter(
            scale * torch.complex(torch.rand(shape, dtype=torch.float32),
                                  torch.rand(shape, dtype=torch.float32)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map [B, C_in, H, W] to [B, C_out, H, W]."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected [B, in_channels, H, W].")
        if x.dtype != torch.float32:
            raise ValueError("SpectralConv2d expects real float32 inputs.")
        batch, _, height, width = x.shape
        if 2*self.modes1 > height or self.modes2 > width//2 + 1:
            raise ValueError("Retained Fourier modes exceed nonoverlapping FFT support.")
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            (batch, self.out_channels, height, width//2 + 1),
            dtype=x_ft.dtype, device=x.device,
        )
        # Keep positive and negative vertical modes; the horizontal FFT is one-sided.
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1,
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2,
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNO2d(nn.Module):
    """Predict a scalar solution from permeability and two coordinate channels."""

    def __init__(self, modes: int = 12, width: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        for name, value in (("modes", modes), ("width", width), ("n_layers", n_layers)):
            _positive_integer(value, name)
        self.lifting = nn.Conv2d(3, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)
        )
        self.projection_hidden = nn.Conv2d(width, 128, kernel_size=1)
        self.projection_output = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Map [B, 3, H, W] to [B, 1, H, W] without changing spatial resolution."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected [B, 3, H, W].")
        x = self.lifting(x)
        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers)
        ):
            x = spectral(x) + pointwise(x)
            if index < len(self.spectral_layers)-1:
                x = F.gelu(x)
        return self.projection_output(F.gelu(self.projection_hidden(x)))


def relative_l2_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Return mean per-sample relative L2 error for nonzero reference fields."""
    if pred.shape != target.shape or pred.ndim < 2 or pred.shape[0] == 0:
        raise ValueError("Prediction and target must have matching nonempty batch shapes.")
    difference = pred.flatten(1) - target.flatten(1)
    reference_norm = torch.norm(target.flatten(1), dim=1)
    if torch.any(reference_norm == 0):
        raise ValueError("Relative L2 error is undefined for a zero reference sample.")
    return (torch.norm(difference, dim=1) / reference_norm).mean()
