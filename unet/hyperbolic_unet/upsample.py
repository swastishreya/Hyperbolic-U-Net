import torch.nn as nn
from torch import Tensor
from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match, op_in_tangent_space

class HUpsample(nn.Module):
    def __init__(
        self,
        size=None, 
        scale_factor=None, 
        mode='nearest', 
        align_corners=None, 
        recompute_scale_factor=None,
        manifold: Manifold = None
    ) -> None:
        super(HUpsample, self).__init__()
        self.op = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
        self.manifold = manifold

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=1, input=x)
        x = op_in_tangent_space(self.op, self.manifold, x)
        return x

class HBilinearUpsample(nn.Module):
    def __init__(
        self,
        manifold: Manifold,
        torch_like_size: bool = True
    ) -> None:
        super(HBilinearUpsample, self).__init__()
        self.manifold = manifold
        self.torch_like_size = torch_like_size

    def geodesic_midpoint(self, x: ManifoldTensor, y: ManifoldTensor) -> ManifoldTensor:
        if not isinstance(x, ManifoldTensor) and isinstance(x, Tensor):
            x = ManifoldTensor(data=x, manifold=self.manifold, man_dim=1)
        if not isinstance(y, ManifoldTensor):
            y = ManifoldTensor(data=y, manifold=self.manifold, man_dim=1)
        # Compute the geodesic midpoint between two points on the manifold
        first_term = self.manifold.logmap(x, y)
        if first_term.tensor.isnan().any():
            raise ValueError("NaN detected in first term of geodesic midpoint calculation.")
        first_term = first_term.tensor * 0.5
        first_term = TangentTensor(data=first_term, manifold_points=x, manifold=self.manifold, man_dim=1)
        m = self.manifold.expmap(first_term)
        return m
    
    def bilinear_cell(self, x00: ManifoldTensor, x10: ManifoldTensor, x01: ManifoldTensor, x11: ManifoldTensor) -> ManifoldTensor:
        """
        xij: (..., C) four Poincaré points at corners of a cell.
        u,v in [0,1]; first lerp along x, then along y.
        """
        a = self.geodesic_midpoint(x00, x10)  # bottom edge
        b = self.geodesic_midpoint(x01, x11)  # top edge
        return self.geodesic_midpoint(a, b)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=1, input=x)

        B, C, H, W = x.shape

        # Output sizes
        Ho = 2*H if self.torch_like_size else 2*H - 1
        Wo = 2*W if self.torch_like_size else 2*W - 1
        x = x.tensor
        out = x.new_empty((B, C, Ho, Wo))

        out[:, :, 0::2, 0::2] = x

        if W > 1:
            left  = x[:, :, :, 0:W-1]
            right = x[:, :, :, 1:W]
            mid_h = self.geodesic_midpoint(left, right)  # Manifold dimension = channels axis = 1
            if mid_h.tensor.isnan().any():
                raise ValueError("NaN detected in mid_h calculation.")
            out[:, :, 0::2, 1:2*(W-1):2] = mid_h.tensor

        if H > 1:
            top = x[:, :, 0:H-1, :]
            bot = x[:, :, 1:H,   :]
            mid_v = self.geodesic_midpoint(top, bot)
            if mid_v.tensor.isnan().any():
                raise ValueError("NaN detected in mid_v calculation.")
            out[:, :, 1:2*(H-1):2, 0::2] = mid_v.tensor

        if H > 1 and W > 1:
            x00 = x[:, :, 0:H-1, 0:W-1]
            x10 = x[:, :, 0:H-1, 1:W   ]
            x01 = x[:, :, 1:H,   0:W-1 ]
            x11 = x[:, :, 1:H,   1:W   ]
            center = self.bilinear_cell(x00, x10, x01, x11)
            if center.tensor.isnan().any():
                raise ValueError("NaN detected in center calculation.")
            out[:, :, 1:2*(H-1):2, 1:2*(W-1):2] = center.tensor

        if self.torch_like_size:
            # Duplicate last column/row to reach (2H, 2W), similar to F.interpolate(..., align_corners=False)
            out[:, :, :, -1] = out[:, :, :, -2]
            out[:, :, -1, :] = out[:, :, -2, :]
        out = ManifoldTensor(data=out, manifold=self.manifold, man_dim=1)
        if out.tensor.isnan().any():
            raise ValueError("NaN detected in output calculation.")
        return out
