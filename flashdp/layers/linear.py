import torch
import torch.nn as nn
from torch.autograd.function import Function
import math

from ..core import mm_fused, linear_weight_flashdp, linear_bias_grad_clip
from .utils import DPParameter as Parameter
from ..core.utils import RuntimeAutoTuner


class DPLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, C: float = 1.0, clamp_value: float = 1.0, noise_multiplier: float = 1.0, auto_tune: bool = True):
        use_bias = False if bias is None else True
        super(DPLinear, self).__init__(in_features, out_features, use_bias)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.dp = True
        self.C = C
        self.clamp_value = clamp_value
        self.noise_multiplier = noise_multiplier
        self.runtime_tuner = RuntimeAutoTuner(enable=auto_tune)
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if use_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        self.runtime_tuner.final_tune()
        return _ApplyDPLinearFunc(
            self.C, self.clamp_value, self.noise_multiplier, self.runtime_tuner)(input, self.weight, self.bias)


def _ApplyDPLinearFunc(C, clamp_value, noise_multiplier, runtime_tuner):
    """
        Returns a function that computes the linear function with differential privacy.
    """
    class _DPLinearFunc(Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            ctx.save_for_backward(input, weight, bias)
            output = input @ weight.t()
            if bias is not None:
                output += bias
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            if input.dim() == 2:
                input = input.unsqueeze(1)
                grad_output = grad_output.unsqueeze(1)
            if input.dim() > 3:
                B, M, N = input.shape[0], input.shape[-1], grad_output.shape[-1]
                K = input.numel() // (B * M)
                input = input.reshape(B, K, M)
                grad_output = grad_output.reshape(B, K, N)
            grad_input = grad_weight = grad_bias = None
            B = input.shape[0]

            if ctx.needs_input_grad[0]:
                if grad_output.dtype == weight.dtype:
                    grad_input = grad_output @ weight
                else:
                    # AMP support
                    grad_input = grad_output @ weight.to(grad_output.dtype)
            else:
                grad_input = None

            if ctx.needs_input_grad[1]:
                grad_weight = linear_weight_flashdp(
                    input, grad_output, C, clamp_value, runtime_tuner).div_(B)
                grad_weight.add_(torch.normal(
                    mean=0,
                    std=noise_multiplier / math.sqrt(B),
                    size=grad_weight.shape,
                    device=grad_weight.device,
                    dtype=grad_weight.dtype,
                ))
            else:
                grad_weight = None

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = linear_bias_grad_clip(grad_output, C, clamp_value)
                grad_bias.add_(torch.normal(
                    mean=0,
                    std=noise_multiplier / math.sqrt(B),
                    size=grad_bias.shape,
                    device=grad_bias.device,
                    dtype=grad_bias.dtype
                ))
            else:
                grad_bias = None
            
            # Check if the grad shape is correct
            if grad_input is not None and grad_input.shape != input.shape:
                raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
            if grad_weight is not None and grad_weight.shape != weight.shape:
                raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
            if grad_bias is not None and grad_bias.shape != bias.shape:
                raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")

            return grad_input, grad_weight, grad_bias
    
    return _DPLinearFunc.apply

