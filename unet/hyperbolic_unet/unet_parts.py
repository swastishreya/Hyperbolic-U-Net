import torch.nn as nn
import torch.nn.functional as F
import math

import hypll.nn as hnn

from .transpose_convolution import HConvTranspose2d
from .upsample import HBilinearUpsample
from .poincare_ops import poincare_cat  

def create_norm_logger():
    norm_log = []
    def norm_logger_hook(name):
        def hook(module, input, output):
            input_norm = math.log10((module.manifold.logmap(x=None, y=input[0])).tensor.norm().item())
            output_norm = math.log10((module.manifold.logmap(x=None, y=output)).tensor.norm().item())
            print(f"[{name}] Input norm: {input_norm:.4f}, Output norm: {output_norm:.4f}")
            norm_log.append({
                'layer': name,
                'input_norm': input_norm,
                'output_norm': output_norm
            })
        return hook
    return norm_logger_hook, norm_log

def register_norm_hooks(model):
    norm_logger_hook, norm_log = create_norm_logger()
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (hnn.HConvolution2d, hnn.HBatchNorm2d, HConvTranspose2d,
                               hnn.HMaxPool2d, hnn.HReLU)):
            hook = module.register_forward_hook(norm_logger_hook(name))
            hooks.append(hook)
    return hooks, norm_log


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, manifold, mid_channels=None):
        super().__init__()
        self.manifold = manifold
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = hnn.HConvolution2d(
            in_channels, mid_channels, kernel_size=3, manifold=manifold, padding=1, bias=False
        )
        self.bn1 = hnn.HBatchNorm2d(mid_channels, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)
        self.conv2 = hnn.HConvolution2d(
            mid_channels, out_channels, kernel_size=3, manifold=manifold, padding=1, bias=False
        )
        self.bn2 = hnn.HBatchNorm2d(out_channels, manifold=manifold)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, manifold):
        super().__init__()
        self.manifold = manifold
        self.maxpool_conv = nn.Sequential(
            hnn.HMaxPool2d(2, manifold=manifold),
            DoubleConv(in_channels, out_channels, manifold),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, manifold, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        self.manifold = manifold

        if bilinear:
            self.up = HBilinearUpsample(manifold=self.manifold)
        else:
            self.up = HConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, manifold=manifold, stride=2
            )
        self.conv = DoubleConv(in_channels, out_channels, manifold=manifold)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if not self.bilinear:
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            
            x1.tensor = F.pad(
                x1.tensor,
                (
                    diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2,
                )
            )

        x = poincare_cat(x2, x1, dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, manifold):
        super(OutConv, self).__init__()
        self.conv = hnn.HConvolution2d(in_channels, out_channels, kernel_size=1, manifold=manifold)

    def forward(self, x):
        return self.conv(x)
