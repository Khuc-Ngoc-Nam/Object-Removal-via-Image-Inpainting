from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = self.spatial_conv(x)

        ffted = torch.fft.rfft2(x_spatial, norm="backward")
        ffted_cat = torch.cat([ffted.real, ffted.imag], dim=1)
        ffted_cat = self.spectral_conv(ffted_cat)

        channels = ffted_cat.shape[1] // 2
        real, imag = torch.split(ffted_cat, channels, dim=1)
        ffted_complex = torch.complex(real, imag)
        x_spectral = torch.fft.irfft2(
            ffted_complex,
            s=(x_spatial.size(-2), x_spatial.size(-1)),
            norm="backward",
        )

        return self.final_conv(x_spatial + x_spectral)


class FFC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio_global: float = 0.5):
        super().__init__()

        self.cg = int(in_channels * ratio_global)
        self.cl = in_channels - self.cg
        self.out_cg = int(out_channels * ratio_global)
        self.out_cl = out_channels - self.out_cg

        self.conv_l2l = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.cl, self.out_cl, kernel_size=3, padding=0, bias=False),
        )
        self.conv_l2g = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.cl, self.out_cg, kernel_size=3, padding=0, bias=False),
        )
        self.conv_g2l = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.cg, self.out_cl, kernel_size=3, padding=0, bias=False),
        )
        self.conv_g2g = SpectralTransform(self.cg, self.out_cg)

        self.bn_l = nn.BatchNorm2d(self.out_cl)
        self.bn_g = nn.BatchNorm2d(self.out_cg)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local, x_global = torch.split(x, [self.cl, self.cg], dim=1)

        out_local = self.conv_l2l(x_local) + self.conv_g2l(x_global)
        out_global = self.conv_l2g(x_local) + self.conv_g2g(x_global)

        out_local = self.relu(self.bn_l(out_local))
        out_global = self.relu(self.bn_g(out_global))

        return torch.cat([out_local, out_global], dim=1)


class FFCResBlock(nn.Module):
    def __init__(self, channels: int, ratio_global: float = 0.5):
        super().__init__()
        self.ffc1 = FFC(channels, channels, ratio_global)
        self.ffc2 = FFC(channels, channels, ratio_global)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffc2(self.ffc1(x))


class ConcatTupleLayer(nn.Module):
    def forward(self, x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(x, tuple):
            return torch.cat(x, dim=1)
        return x


class LaMaGenerator(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 3,
        ngf: int = 64,
        num_ffc_blocks: int = 9,
    ):
        super().__init__()

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        for i in range(3):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        mult = 2**3
        for _ in range(num_ffc_blocks):
            model += [FFCResBlock(ngf * mult)]

        for i in range(3):
            mult = 2 ** (3 - i)
            in_ch = ngf * mult
            out_ch = int(ngf * mult / 2)
            model += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
