from torch import cat
from torch.nn.common_types import _size_2_t
from torch.nn.functional import fold

from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.utils.math import beta_func


def poincare_fold(
    input: ManifoldTensor,
    output_size: _size_2_t,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
) -> ManifoldTensor:
    if len(kernel_size) == 2:
        kernel_vol = kernel_size[0] * kernel_size[1]
    else:
        kernel_vol = kernel_size ** 2
        kernel_size = (kernel_size, kernel_size)

    stacked_channels = input.size(1)
    if stacked_channels % kernel_vol:
        raise RuntimeError(
            f"Expected size of input's dimension 1 to be divisible by the product of kernel_size, "
            f"but got input.size(1)={stacked_channels} and kernel_size={kernel_size}."
        )
    output_channels = stacked_channels // kernel_vol
    
    beta_n = beta_func(stacked_channels, 1 / 2)
    beta_ni = beta_func(output_channels, 1 / 2)

    input = input.manifold.logmap(x=None, y=input)
    input.tensor = input.tensor * beta_ni / beta_n
    new_tensor = fold(
        input=input.tensor,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
    )
    new_tensor = TangentTensor(
        data=new_tensor, manifold_points=None, manifold=input.manifold, man_dim=1
    )
    return input.manifold.expmap(new_tensor)


def poincare_cat(
    x: ManifoldTensor,
    y: ManifoldTensor,
    dim: int = -1,
):
    if x.dim() != y.dim():
        raise RuntimeError(
            f"Tensors must have same number of dimensions: got {x.dim()} and {y.dim()}"
        )

    if x.man_dim != y.man_dim:
        raise RuntimeError(
            f"Trying to concatenate ManifoldTensors with differing manifold dimensions: "
            f"{x.man_dim} and {y.man_dim}"
        )
    man_dim = x.man_dim
    
    if x.manifold != y.manifold:
        raise RuntimeError(
            f"Trying to concatenate tensors on different manifolds."
        )
    manifold = x.manifold
    
    if dim != man_dim:
        return ManifoldTensor(
            data=cat((x.tensor, y.tensor), dim=dim),
            manifold=manifold,
            man_dim=man_dim,
        )
    else:
        beta_x = beta_func(x.size(man_dim), 1 / 2)
        beta_y = beta_func(y.size(man_dim), 1 / 2)
        beta_xy = beta_func(x.size(man_dim) + y.size(man_dim), 1 / 2)

        x = manifold.logmap(x=None, y=x)
        y = manifold.logmap(x=None, y=y)
        
        x_scaled = x.tensor * beta_xy / beta_x
        y_scaled = y.tensor * beta_xy / beta_y

        new_tensor = cat((x_scaled, y_scaled), dim=dim)
        new_tensor = TangentTensor(
            data=new_tensor, manifold_points=None, manifold=manifold, man_dim=man_dim
        )

        return manifold.expmap(new_tensor)
