import os
import json
import torch
import triton
import triton.language as tl

from .clip_fn import get_clip_factor_triton
from .utils import to_tl_type, supported_acc_dtypes


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),

        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),

        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['norm_ptr'],
)
@triton.jit
def _mtm_prenorm_kernel(
    a_ptr, b_ptr, c_ptr, norm_ptr,
    M, N, K,
    stride_ak, stride_am,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    acc_dtype: tl.constexpr,
    data_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_k[None, :] * stride_ak + offs_am[:, None] * stride_am)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(data_dtype)

    norm_register = accumulator * accumulator       # (BLOCK_SIZE_M, BLOCK_SIZE_N) vector
    norm_register = tl.sum(norm_register, axis=0)   # (BLOCK_SIZE_N) vector
    norm_register = tl.sum(norm_register, axis=0)   # (1) scalar
    norm_ptrs = tl.arange(0, 1)
    tl.atomic_add(norm_ptr+ norm_ptrs, norm_register)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def _mat_clip_kernel(
    c_ptr, 
    norm_ptr,
    M, 
    N, 
    clip_args_ptr: tl.pointer_type,
    acc_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr
):
    mid = tl.program_id(0)
    nid = tl.program_id(1)
    
    # Compute the clip factor
    norm_ptrs = tl.arange(0, 1)
    norm = tl.sqrt(tl.load(norm_ptr + norm_ptrs).to(acc_dtype))
    clip_factor = get_clip_factor_triton(norm, clip_args_ptr)

    # Load the c
    c_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_rows[:, None] * N + c_cols[None, :]
    c = tl.load(c_ptr + c_ptrs).to(acc_dtype)

    # Clip the c
    c = c * clip_factor[:, None]
    tl.store(c_ptr + c_ptrs, c.to(out_dtype))


@torch.jit.script
def _mat_clip(mat, mat_norm, clip_args):
    mat = mat * torch.clamp(clip_args[0] / (torch.sqrt(mat_norm)+1e-10), max=clip_args[1])
    return mat

def mtm_clip(mat1, mat2, M, N, K, grid1=None, grid2=None, clip_args=None):
    out_dtype = mat1.dtype
    if out_dtype in supported_acc_dtypes.keys():
        acc_dtype = supported_acc_dtypes[out_dtype][0]
    else:
        raise KeyError("The type of input mat is not supported.")
    tl_acc_dtype = to_tl_type(acc_dtype)
    tl_out_dtype = to_tl_type(out_dtype)
    out = torch.empty([M, N], device=mat1.device, dtype=out_dtype)
    out_norm = torch.empty([1], device=mat1.device, dtype=acc_dtype)
    if grid1 is None:
        grid1 = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    _mtm_prenorm_kernel[grid1](
        mat1, mat2, out, out_norm, 
        M, N, K, 
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        tl_acc_dtype, tl_out_dtype)
    if grid2 is None:
        grid2 = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    _mat_clip_kernel[grid2](out, out_norm, M, N, clip_args, tl_acc_dtype, tl_out_dtype)
    del out_norm
    return out

def bmtm_clip_triton(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor,
    clip_args: torch.Tensor = None
):
    Batch, M, K, N = tensor1.shape[0], tensor1.shape[2], tensor1.shape[1], tensor2.shape[2]
    for block_b in range(Batch):
        if block_b == 0:
            out = mtm_clip(tensor1[block_b], tensor2[block_b], M, N, K, clip_args=clip_args)
        else:
            out.add_(mtm_clip(tensor1[block_b], tensor2[block_b], M, N, K, clip_args=clip_args))
    return out

