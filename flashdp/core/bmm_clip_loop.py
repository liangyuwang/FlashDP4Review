import os
import torch
import triton
import triton.language as tl

from .clip_fn import get_clip_factor_triton
from .utils import to_tl_type, supported_acc_dtypes


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
    reset_to_zero=['norm_ptr'],
)
@triton.jit
def _mm_prenorm_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    norm_ptr,
    M, 
    N, 
    K, 
    mat1_start_ptr, 
    mat2_start_ptr, 
    acc_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    mid = tl.program_id(0)
    nid = tl.program_id(1)

    # Compute the norm_ptrs
    norm_ptrs = tl.arange(0, 1)

    # Starting from the batch_index * matrix_size
    a_ptr += mat1_start_ptr
    b_ptr += mat2_start_ptr

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

    norm_register = accumulator * accumulator       # (BLOCK_SIZE_M, BLOCK_SIZE_N) vector
    norm_register = tl.sum(norm_register, axis=0)   # (BLOCK_SIZE_N) vector
    norm_register = tl.sum(norm_register, axis=0)   # (1) scalar
    tl.atomic_add(norm_ptr+ norm_ptrs, norm_register)

    # C's block's offsets
    c_ptrs = a_rows[:, None] * N + b_cols[None, :]
    tl.store(c_ptr+ c_ptrs, accumulator.to(out_dtype))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
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
    # if mid == 0 and nid == 0:
    #     tl.store(norm_ptr + norm_ptrs, tl.zeros_like(norm)) # Reset the norm

    # Load the c
    c_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_rows[:, None] * N + c_cols[None, :]
    c = tl.load(c_ptr + c_ptrs).to(acc_dtype)

    # Clip the c
    c = c * clip_factor[:, None]
    tl.store(c_ptr + c_ptrs, c.to(out_dtype))


def mm_clip(tensor1, tensor2, M, N, K, index_b, grid1=None, grid2=None, clip_args=None):
    out_dtype = tensor1.dtype
    if out_dtype in supported_acc_dtypes.keys():
        acc_dtype = supported_acc_dtypes[out_dtype][0]
    else:
        raise KeyError("The type of input mat is not supported.")
    tl_acc_dtype = to_tl_type(acc_dtype)
    tl_out_dtype = to_tl_type(out_dtype)
    out_norm = torch.zeros([1], device=tensor1.device, dtype=out_dtype)
    out = torch.zeros([M, N], device=tensor1.device, dtype=acc_dtype)
    mat1_start_ptr = index_b * M * K
    mat2_start_ptr = index_b * K * N
    if grid1 is None:
        grid1 = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    _mm_prenorm_kernel[grid1](tensor1, tensor2, out, out_norm, M, N, K, mat1_start_ptr, mat2_start_ptr, tl_acc_dtype, tl_out_dtype)
    if grid2 is None:
        grid2 = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    _mat_clip_kernel[grid2](out, out_norm, M, N, clip_args, tl_acc_dtype, tl_out_dtype)
    del out_norm
    return out

def bmm_clip_triton(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor,
    clip_args: torch.Tensor = None
):
    Batch, M, K, N = tensor1.shape[0], tensor1.shape[1], tensor1.shape[2], tensor2.shape[2]
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    for index_b in range(Batch):
        if index_b == 0:
            out = mm_clip(tensor1, tensor2, M, N, K, index_b, grid1=grid, grid2=grid, clip_args=clip_args)
        else:
            out.add_(mm_clip(tensor1, tensor2, M, N, K, index_b, grid1=grid, grid2=grid, clip_args=clip_args))
    return out

