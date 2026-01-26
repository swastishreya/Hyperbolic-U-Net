from torch.nn import Module
from torch.nn.common_types import _size_2_t

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match

from .poincare_ops import poincare_fold


class HConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        manifold: Manifold,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(HConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple) and len(kernel_size) == 2
            else (kernel_size, kernel_size)
        )
        self.kernel_vol = self.kernel_size[0] * self.kernel_size[1]
        self.manifold = manifold
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.weights, self.bias = self.manifold.construct_dl_parameters(
            in_features=in_channels,
            out_features=self.kernel_vol * out_channels,
            bias=self.bias,
        )
        self.manifold.reset_parameters(self.weights, self.bias)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=1, input=x)

        out_height = (
            (x.size(2) - 1) * self.stride
            - 2 * self.padding
            + self.kernel_size[0]
        )
        out_width = (
            (x.size(3) - 1) * self.stride
            - 2 * self.padding
            + self.kernel_size[1]
        )
        if x.tensor.isnan().any():
            print("break")
        x = self.manifold.fully_connected(x=x, z=self.weights, bias=self.bias)
        if x.tensor.isnan().any():
            print("break")
        x = ManifoldTensor(
            data=x.tensor.flatten(start_dim=2),
            manifold=x.manifold,
            man_dim=1,
        )
        x = poincare_fold(
            input=x.flatten(start_dim=2),
            output_size=(out_height, out_width),
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        if x.tensor.isnan().any():
            print("break")
        return x
