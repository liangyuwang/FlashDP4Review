import torch
import triton
import triton.language as tl

from ..core import bmtm_clip
from ..core.clip_fn import get_clip_factor_torch
from .utils import RuntimeAutoTuner, to_tl_type, supported_acc_dtypes


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _mm_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    M, 
    N, 
    K, 
    acc_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    mid = tl.program_id(0)
    nid = tl.program_id(1)

    # Starting row + BLOCK_SIZE_M more rows
    a_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Starting col + BLOCK_SIZE_N more columns
    b_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + a_rows[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + b_cols[None, :]

    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=acc_dtype)
    for k in range(K//BLOCK_SIZE_K):
        a = tl.load(a_ptrs).to(acc_dtype)
        b = tl.load(b_ptrs).to(acc_dtype)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    # C's block's offsets
    c_ptrs = a_rows[:, None] * N + b_cols[None, :]
    tl.store(c_ptr+ c_ptrs, accumulator.to(out_dtype))


def mm_fused(tensor1: torch.Tensor, tensor2: torch.Tensor):
    out_dtype = tensor1.dtype
    if out_dtype in supported_acc_dtypes.keys():
        acc_dtype = supported_acc_dtypes[out_dtype][0]
    else:
        raise KeyError("The type of input mat is not supported.")
    if tensor1.dim() < 2 or tensor2.dim() < 2:
        raise KeyError("The dimensions of input tensors must be at least 2.")
    if tensor1.shape[-1] != tensor2.shape[0]:
        raise ValueError("tensor1 last dimension must match tensor2 first dimension")
    if tensor1.dim() > 2: 
        tensor1_original_shape = tensor1.shape
        tensor1 = tensor1.reshape(-1, tensor1.shape[-1])
    else:
        tensor1_original_shape = tensor1.shape
    if tensor2.dim() > 2: 
        tensor2_original_shape = tensor2.shape
        tensor2 = tensor2.reshape(tensor2.shape[0], -1)
    else:
        tensor2_original_shape = tensor2.shape
    tl_acc_dtype = to_tl_type(acc_dtype)
    tl_out_dtype = to_tl_type(out_dtype)
    M, K, N = tensor1.shape[0], tensor1.shape[1], tensor2.shape[1]
    out = torch.zeros([M, N], device=tensor1.device, dtype=acc_dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    _mm_kernel[grid](tensor1, tensor2, out, M, N, K, tl_acc_dtype, tl_out_dtype)
    out = out.reshape(tensor1_original_shape[:-1] + tensor2_original_shape[1:])
    return out


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


def linear_weight_flashdp(input: torch.Tensor, grad_output: torch.Tensor, C: float, clamp_value: float, runtime_tuner: RuntimeAutoTuner):
    clip_args = torch.Tensor([C, clamp_value]).to(grad_output.device)
    tuned_func = runtime_tuner.choose_function(
        [bmtm_clip, _weight_grad_block_clip],
        grad_output, input.to(grad_output.dtype), clip_args=clip_args
    )
    return tuned_func(grad_output, input.to(grad_output.dtype), clip_args=clip_args)


def linear_bias_grad_clip(grad_output: torch.Tensor, C: float, clamp_value: float):
    clip_args = torch.Tensor([C, clamp_value]).to(grad_output.device)
    grad_bias_flatten = grad_output.view(grad_output.shape[0], -1)
    grad_bias_norm = torch.norm(grad_bias_flatten, p=2, dim=-1, keepdim=True)
    clip_factor = get_clip_factor_torch(grad_bias_norm, clip_args)
    grad_bias_flatten.mul_(clip_factor)
    clipped_grad_bias = grad_bias_flatten.view(-1, grad_output.shape[-1]).mean(0)
    return clipped_grad_bias

def linear_bias_flashdp(grad_output: torch.Tensor, C: float, clamp_value: float):
    raise NotImplementedError

