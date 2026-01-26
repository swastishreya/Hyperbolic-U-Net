from torch import Tensor

from hypll.tensors import ManifoldTensor


def check_vals_hook(module, input, output):
    if isinstance(output, Tensor):
        if output.isnan().any():
            raise ValueError(f"Encountered a nan value in {module}.")
    elif isinstance(output, ManifoldTensor):
        if output.tensor.isnan().any():
            raise ValueError(f"Encountered a nan value in {module}.")
    else:
        raise RuntimeError(f"Module {module} had an output of type {type(output)}.")
