import math
from typing import Dict, Iterable, List, Tuple, Optional
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn
import hypll.nn as hnn
from hypll.tensors import ManifoldTensor
from unet.hyperbolic_unet.transpose_convolution import HConvTranspose2d

def he_or_orthogonal_init(module: nn.Module, mode: str = "kaiming", nonlinearity: str = "relu") -> None:
    for m in module.modules():
        if hasattr(m, "weights") and isinstance(getattr(m, "weights", None), torch.Tensor):
            if mode.lower() in ["kaiming"]:
                nn.init.kaiming_normal_(m.weights.tensor, nonlinearity=nonlinearity)
            elif mode.lower().startswith("orth"):
                nn.init.orthogonal_(m.weights.tensor)
            else:
                raise ValueError(f"Unknown init mode: {mode}")
            if getattr(m, "bias", None) is not None and isinstance(m.bias, torch.Tensor):
                nn.init.zeros_(m.bias)

def hyperbolic_sqdist_to_origin(x: torch.Tensor, m: nn.Module) -> torch.Tensor:
    origin = torch.zeros_like(x.tensor)
    origin = ManifoldTensor(data=origin, manifold=m.manifold, man_dim=1)
    d = m.manifold.dist(x, origin)
    return (d.pow(2)).mean()


def _get_layer_curvature(m: nn.Module, default_c: Optional[float]) -> torch.Tensor:
    if hasattr(m, "manifold") and hasattr(m.manifold, "c"):
        try:
            c_val = m.manifold.c()
            if not isinstance(c_val, torch.Tensor):
                c_val = torch.tensor(float(c_val))
            return c_val
        except Exception as e:
            raise RuntimeError(f"Could not read curvature from layer.manifold.c(): {e}")
    if default_c is None:
        raise ValueError(
            f"Layer {m} has no manifold.c(), and no default curvature_c was provided."
        )
    return torch.tensor(float(default_c))

class _TempWeight:
    """
    Context manager to temporarily replace a layer's weight with a given tensor.
    Restores on exit.
    """
    def __init__(self, layer: nn.Module, new_weight: torch.Tensor):
        self.layer = layer
        self.new_weight = new_weight
        self._backup = None

    def __enter__(self):
        w = getattr(self.layer, "weights", None)
        if not hasattr(w, "tensor"):
            raise TypeError(f"Layer {type(self.layer)} has no 'weights.tensor'")
        self._backup = w.tensor
        w.tensor = self.new_weight
        return self.layer

    def __exit__(self, exc_type, exc, tb):
        if self._backup is not None:
            self.layer.weights.tensor = self._backup
        self._backup = None

def _is_scale_target(m: nn.Module) -> bool:
    return isinstance(
        m, tuple(t for t in (hnn.HConvolution2d, HConvTranspose2d) if t)
    )

@contextmanager
def _collect_inputs(model: nn.Module, max_per_layer: int = 4):
    """
    Register forward hooks to collect inputs seen by selected (scalable) layers.
    """
    buffers: Dict[int, List[torch.Tensor]] = defaultdict(list)
    handles = []

    def hook_fn(module, inputs, output):
        if len(inputs) == 0:
            return
        x = inputs[0].detach()
        if len(buffers[id(module)]) < max_per_layer:
            buffers[id(module)].append(x)

    for m in model.modules():
        if _is_scale_target(m):
            handles.append(m.register_forward_hook(hook_fn))

    try:
        yield buffers
    finally:
        for h in handles:
            h.remove()

def _g_value_for_layer_autograd(
    m: nn.Module,
    xs: List[torch.Tensor],
    W0: torch.Tensor,
    s_val: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device, dtype = W0.device, W0.dtype
    s = torch.tensor(s_val, device=device, dtype=dtype, requires_grad=True)

    Ey_acc = torch.zeros((), device=device, dtype=dtype)
    Ex_acc = torch.zeros((), device=device, dtype=dtype)

    for x in xs:
        x = x.to(device)
        scaled_W = s * W0
        with _TempWeight(m, scaled_W):
            y = m(x)

        Ey_acc = Ey_acc + hyperbolic_sqdist_to_origin(y, m)
        Ex_acc = Ex_acc + hyperbolic_sqdist_to_origin(x, m)

    Ey = Ey_acc / max(len(xs), 1)
    Ex = Ex_acc / max(len(xs), 1)
    g = Ey - Ex
    return g, s


def scale_layers_to_preserve_hyperbolic_norm(
    model: nn.Module,
    data_iter: Iterable,
    *,
    curvature_c: Optional[float] = None,
    batches: int = 2,
    newton_iters: int = 8,
    device: Optional[torch.device] = None,
    tol: float = 1e-9
) -> Dict[str, float]:
    
    model.eval()
    target_layers: List[Tuple[str, nn.Module]] = [
        (name, m) for name, m in model.named_modules() if _is_scale_target(m)
    ]

    with _collect_inputs(model) as input_bufs:
        with torch.no_grad():
            it = iter(data_iter)
            for _ in range(batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                x = batch['image'] if isinstance(batch, (list, tuple, dict)) else batch
                x = x.to(device) if device else x
                model(x)

    scales: Dict[str, float] = {}
    for name, m in target_layers:
        xs = input_bufs.get(id(m), [])
        if not xs:
            continue  # no data passed through this layer

        W0 = m.weights.tensor.detach().clone()
        _ = _get_layer_curvature(m, curvature_c)  # ensure curvature is accessible
        s = 1.0

        for _ in range(newton_iters):
            g_tensor, s_var = _g_value_for_layer_autograd(m, xs, W0, s)
            g_val = g_tensor.item()

            gp_val = torch.autograd.grad(
                g_tensor, s_var, retain_graph=False, create_graph=False
            )[0].item()

            if abs(gp_val) < 1e-12:
                break

            step = g_val / gp_val
            s_new = s - step

            if not math.isfinite(s_new) or s_new <= 0:
                s_new = max(1e-6, s * 0.5)

            if abs(s_new - s) < tol:
                s = s_new
                break

            s = s_new

        with torch.no_grad():
            m.weights.tensor.mul_(s)
        scales[name] = float(s)

    return scales

def init_then_hyperbolic_scale(
    model: nn.Module,
    data_iter: Iterable,
    *,
    init_mode: str = "kaiming",           # or "orthogonal"
    nonlinearity: str = "relu",
    curvature_c: Optional[float] = None,  # set if not on layers themselves
    batches: int = 2,
    newton_iters: int = 8,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    1) Initialize weights (Kaiming/Orthogonal)
    2) Empirically scale each layer to preserve mean squared hyperbolic distance to origin.
    """
    he_or_orthogonal_init(model, mode=init_mode, nonlinearity=nonlinearity)
    return scale_layers_to_preserve_hyperbolic_norm(
        model, data_iter,
        curvature_c=curvature_c,
        batches=batches,
        newton_iters=newton_iters,
        device=device
    )