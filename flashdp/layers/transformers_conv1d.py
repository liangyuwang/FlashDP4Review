import torch
import torch.nn as nn
from torch.autograd.function import Function
from transformers.pytorch_utils import Conv1D
import math

from ..core import linear_weight_flashdp, linear_bias_grad_clip
from .utils import DPParameter as Parameter
from ..core.utils import RuntimeAutoTuner


class DPConv1D(Conv1D):
    def __init__(self, nf, nx, C: float = 1.0, clamp_value: float = 1.0, noise_multiplier: float = 1.0, auto_tune: bool = True):
        super().__init__(nf, nx)
        self.dp = True
        self.C = C
        self.clamp_value = clamp_value
        self.noise_multiplier = noise_multiplier
        self.nf = nf
        self.runtime_tuner = RuntimeAutoTuner(enable=auto_tune)
        self.weight = Parameter(torch.empty(nx, nf))
        self.bias = Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        self.runtime_tuner.final_tune()
        size_out = x.size()[:-1] + (self.nf,)
        x = _ApplyDPConv1DFunc(
            self.C, self.clamp_value, self.noise_multiplier, self.runtime_tuner)(self.bias, x, self.weight)
        x = x.view(size_out)
        return x


def _ApplyDPConv1DFunc(C, clamp_value, noise_multiplier, runtime_tuner):
    """
        Returns a function that computes the conv1d function with differential privacy.
    """
    class _DPConv1DFunc(Function):
        @staticmethod
        def forward(ctx, bias, input, weight):
            ctx.save_for_backward(bias, input, weight)
            output = input @ weight + bias
            return output

        @staticmethod
        def backward(ctx, grad_output):
            bias, input, weight = ctx.saved_tensors
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
                grad_bias = linear_bias_grad_clip(grad_output, C, clamp_value)
                grad_bias.add_(torch.normal(
                    mean=0,
                    std=noise_multiplier / math.sqrt(B),
                    size=grad_bias.shape,
                    device=grad_bias.device,
                ))
            else:
                grad_bias = None

            if ctx.needs_input_grad[1]:
                if grad_output.dtype == weight.dtype:
                    grad_input = grad_output @ weight.t()
                else:
                    # AMP support
                    grad_input = grad_output @ weight.to(grad_output.dtype).t()
            else:
                grad_input = None

            if ctx.needs_input_grad[2]:
                grad_weight = linear_weight_flashdp(
                    input, grad_output, C, clamp_value, runtime_tuner).div_(B)
                grad_weight.add_(torch.normal(
                    mean=0,
                    std=noise_multiplier / math.sqrt(B),
                    size=grad_weight.shape,
                    device=grad_weight.device,
                ))
            else:
                grad_weight = None

            # Check if the grad shape is correct
            if grad_input is not None and grad_input.shape != input.shape:
                raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
            if grad_weight is not None and grad_weight.shape != weight.shape:
                raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
            if grad_bias is not None and grad_bias.shape != bias.shape:
                raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")

            return grad_bias, grad_input, grad_weight
    
    return _DPConv1DFunc.apply
