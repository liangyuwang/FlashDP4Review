import torch
import torch.nn as nn

from ..core import bmtm_clip
from ..core.clip_fn import get_clip_factor_torch
from .utils import RuntimeAutoTuner


@torch.jit.script
def _weight_grad_block_clip(grad_output: torch.Tensor, input: torch.Tensor, clip_args: torch.Tensor = None, BK: int = 1):
    M, K, N = grad_output.shape[-1], input.shape[0], input.shape[-1]
    clipped_grad_weight = torch.zeros([M, N], device=input.device, dtype=input.dtype)
    for k in range(K // BK):
        grad_weight_preclip = torch.einsum('k...m,k...n->kmn', grad_output[k*BK:(k+1)*BK], input[k*BK:(k+1)*BK])
        grad_weight_norm = torch.norm(grad_weight_preclip, p=2, dim=(-1, -2), keepdim=True)
        clip_factor = get_clip_factor_torch(grad_weight_norm, clip_args)
        grad_weight_preclip.mul_(clip_factor)
        clipped_grad_weight += grad_weight_preclip.sum(0)
    return clipped_grad_weight


def conv1d_weight_flashdp(input: torch.Tensor, grad_output: torch.Tensor, C: float, clamp_value: float, runtime_tuner: RuntimeAutoTuner):
    clip_args = torch.Tensor([C, clamp_value]).to(input.device)
    tuned_func = runtime_tuner.choose_function(
        [bmtm_clip, _weight_grad_block_clip],
        input.to(grad_output.dtype), grad_output, clip_args=clip_args
    )
    return tuned_func(input.to(grad_output.dtype), grad_output, clip_args=clip_args)


# @torch.jit.script
def conv1d_bias_grad_clip(grad_output: torch.Tensor, C: float, clamp_value: float):
    grad_bias_flatten = grad_output.view(grad_output.shape[0], -1)
    grad_bias_norm = torch.norm(grad_bias_flatten, p=2, dim=-1, keepdim=True)
    clip_factor = torch.clamp(C / (grad_bias_norm + 1e-6), max=clamp_value)
    grad_bias_flatten.mul_(clip_factor)
    clipped_grad_bias = grad_bias_flatten.view(-1, grad_output.shape[-1]).mean(0)

    return clipped_grad_bias


def conv1d_bias_flashdp(grad_output: torch.Tensor, C: float, clamp_value: float):
    raise NotImplementedError

