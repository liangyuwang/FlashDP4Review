import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.normalization import (
    _shape_t,
    numbers
)
import triton
import triton.language as tl
from typing import Optional, List, Tuple, Union
from torch.autograd.function import Function

from ..core import (
    layernorm_dw_clip, 
    layernorm_db_clip, 
    layernorm_fwd_fused,
    layernorm_bwd_clip_fused,
    layernorm_dwdb_fused,
)
from .utils import DPParameter as Parameter


class DPLayerNorm(nn.LayerNorm):
    
    def __init__(
        self, 
        normalized_shape: _shape_t, 
        eps: float = 1e-5, 
        elementwise_affine: bool = True,
        bias: bool = True, 
        device=None, 
        dtype=None,
        use_fused: bool = True,
        C: float = 1.0, 
        clamp_value: float = 1.0,
        noise_multiplier: float = 1.0
    ) -> None:
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        use_bias = False if bias is None else True
        if use_fused and not use_bias:
            raise NotImplementedError("In fused mode, bias must be enabled. Set use_fused=False to use the default implementation.")
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(normalized_shape, eps, elementwise_affine, use_bias, device, dtype)
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_fused = use_fused
        
        self.dp = True
        self.C = C
        self.clamp_value = clamp_value
        self.noise_multiplier = noise_multiplier

        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if use_bias:
                self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_fused:
            if len(self.normalized_shape) != 1:
                raise NotImplementedError("Currently layernorm only support normalized_dim=-1.")
            if self.normalized_shape[0] != input.shape[-1]:
                raise NotImplementedError("In fused mode, currently layernorm only support normalized_dim=-1. Set use_fused=False to use the default implementation.")
            return ApplyDPFusedFunc(self.eps, self.C, self.clamp_value, self.noise_multiplier)(input, self.weight, self.bias)
        else:
            # Compute dimensions for normalization
            normalized_indices = []
            input_shape = input.shape[1:]  # exclude batch dimension
            start_dim = len(input_shape) - len(self.normalized_shape)

            # Check if normalized_shape matches the tail of input_shape
            if input_shape[start_dim:] == self.normalized_shape:
                normalized_indices = list(range(1 + start_dim, input.dim()))  # 1+ to account for batch dimension
            else:
                # Complex case: Handle more flexible matching if necessary
                for dim, size in enumerate(input_shape):
                    if size in self.normalized_shape:
                        normalized_indices.append(dim + 1)  # +1 to account for batch dimension

            mean = input.mean(dim=normalized_indices, keepdim=True)
            var = input.var(dim=normalized_indices, keepdim=True, unbiased=False)

            # Normalize the input
            input = (input - mean) / torch.sqrt(var + self.eps)

            # Apply affine transformation if it's set
            if self.elementwise_affine:
                # return input * self.weight + self.bias
                return ApplyDPAffineFunc(
                    self.normalized_shape, self.C, self.clamp_value, self.noise_multiplier
                    )(input, self.weight, self.bias)
            else:
                return input


def ApplyDPAffineFunc(normalized_shape, C, clamp_value, noise_multiplier):
    """
        Returns a function that computes the matmul function with differential privacy.
    """
    class DPAffineFunction(Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            # This ensures that weight and bias are broadcast correctly across the input dimensions
            if input.dim() > len(normalized_shape):
                shape = [1] * input.dim()
                for i, dim in enumerate(normalized_shape, start=-len(normalized_shape)):
                    shape[i] = dim
                weight = weight.reshape(shape)
                bias = bias.reshape(shape)

            ctx.save_for_backward(input, weight, bias)
            output = input * weight + bias
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None
            B = input.shape[0]

            if ctx.needs_input_grad[0]:
                grad_input = grad_output * weight
            
            if ctx.needs_input_grad[1]:
                # axes = tuple(i for i in range(grad_output.dim()) if grad_output.size(i) != weight.size(i))
                # grad_weight = (grad_output * input).sum(dim=axes, keepdim=True)
                grad_weight = layernorm_dw_clip(input, grad_output, weight, torch.Tensor([C]), torch.Tensor([clamp_value]))
                grad_weight.add_(torch.normal(
                    mean=0,
                    std=noise_multiplier / math.sqrt(B),
                    size=grad_weight.shape,
                    device=grad_weight.device,
                ))
            if ctx.needs_input_grad[2]:
                # axes = tuple(i for i in range(grad_output.dim()) if grad_output.size(i) != bias.size(i))
                # grad_bias = grad_output.sum(dim=axes, keepdim=True)
                grad_bias = layernorm_db_clip(grad_output, bias, torch.Tensor([C]), torch.Tensor([clamp_value]))
                grad_weight.add_(torch.normal(
                    mean=0,
                    std=noise_multiplier / math.sqrt(B),
                    size=grad_weight.shape,
                    device=grad_weight.device,
                ))

            # Check if the grad shape is correct
            if grad_input is not None and grad_input.shape != input.shape:
                raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
            if grad_weight is not None and grad_weight.shape != weight.shape:
                raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
            if grad_bias is not None and grad_bias.shape != bias.shape:
                raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")

            return grad_input, grad_weight, grad_bias
        
    return DPAffineFunction.apply


def ApplyDPFusedFunc(eps, C, clamp_value, noise_multiplier):
    class DPFusedFunction(Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            output, buffer, args = layernorm_fwd_fused(input, weight, bias, eps)
            ctx.save_for_backward(input, weight, bias, buffer['mean'], buffer['rstd'])
            ctx.args = args
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, mean, rstd = ctx.saved_tensors
            buffer = {
                'mean': mean,
                'rstd': rstd,
            }
            args = {
                'BLOCK_SIZE': ctx.args['BLOCK_SIZE'],
                'num_warps': ctx.args['num_warps'],
                'eps': eps,
            }
            clip_args = torch.Tensor([C, clamp_value]).to(grad_output.device)
            dx, dw_, db_, args = layernorm_bwd_clip_fused(grad_output, input, weight, bias, clip_args, buffer, args)
            dw, db = layernorm_dwdb_fused(weight, bias, dw_, db_, args)
            
            # Check if the grad shape is correct
            if dx is not None and dx.shape != input.shape:
                raise RuntimeError(f"grad_input shape {dx.shape} is not equal to input shape {input.shape}")
            if dw is not None and dw.shape != weight.shape:
                raise RuntimeError(f"grad_weight shape {dw.shape} is not equal to weight shape {weight.shape}")
            if db is not None and db.shape != bias.shape:
                raise RuntimeError(f"grad_bias shape {db.shape} is not equal to bias shape {bias.shape}")

            return dx, dw, db
    return DPFusedFunction.apply
        

