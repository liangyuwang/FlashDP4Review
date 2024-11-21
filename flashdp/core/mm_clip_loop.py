import os
import torch
import triton
import triton.language as tl

from .clip_fn import get_clip_factor_triton


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 1}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 1}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 1}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 1}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 1}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
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
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    mid = tl.program_id(0)
    nid = tl.program_id(1)

    # # Compute the norm_ptrs
    norm_ptrs = tl.arange(0, BLOCK_SIZE_K)

    # Compute the A and B pointers
    a_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    b_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + a_rows[:, None] * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + b_cols[None, :]

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    ab = a[:, :, None] * b[None, :, :]  # (BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
    
    norm_register = tl.sum(ab * ab, axis=0)         # (BLOCK_SIZE_K, BLOCK_SIZE_N) vector
    norm_register = tl.sum(norm_register, axis=1)   # (BLOCK_SIZE_K) vector
    tl.atomic_add(norm_ptr+ norm_ptrs, norm_register)

    # C's block's offsets
    c_ptrs = (a_rows[:, None, None] * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :, None]) * N + b_cols[None, None, :]
    tl.store(c_ptr+ c_ptrs, ab)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 1}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 1}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 1}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 1}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 1}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _mat_clip_kernel(
    ab_ptr, 
    norm_ptr,
    M, 
    N, 
    K, 
    clip_args_ptr: tl.pointer_type,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    mid = tl.program_id(0)
    nid = tl.program_id(1)
    
    # Compute the clip factor
    norm_ptrs = tl.arange(0, BLOCK_SIZE_K)
    norm = tl.sqrt(tl.load(norm_ptr + norm_ptrs))
    clip_factor = get_clip_factor_triton(norm, clip_args_ptr)
    if mid == 0 and nid == 0:
        tl.store(norm_ptr + norm_ptrs, tl.zeros_like(norm)) # Reset the norm

    # Load the ab
    ab_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ab_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ab_ptrs = (ab_rows[:, None, None] * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :, None]) * N + ab_cols[None, None, :]
    ab = tl.load(ab_ptr + ab_ptrs)

    # Clip the ab
    ab = ab * clip_factor[None, :, None]
    tl.store(ab_ptr + ab_ptrs, ab)



def _outer_clip(mat1, mat2, M, N, BLOCK_SIZE_K=1, grid1=None, grid2=None, clip_args=None):
    out = torch.zeros([M, BLOCK_SIZE_K, N], device=mat1.device, dtype=mat1.dtype)
    out_norm = torch.zeros([BLOCK_SIZE_K], device=mat1.device, dtype=mat1.dtype)
    if grid1 is None:
        grid1 = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    _mm_prenorm_kernel[grid1](mat1, mat2, out, out_norm, M, N, BLOCK_SIZE_K)
    if grid2 is None:
        grid2 = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    _mat_clip_kernel[grid2](out, out_norm, M, N, BLOCK_SIZE_K, clip_args)
    del out_norm
    return out


def mm_clip_triton(
    mat1: torch.Tensor, 
    mat2: torch.Tensor, 
    clip_args: torch.Tensor = None
):
    M, K, N = mat1.shape[0], mat1.shape[1], mat2.shape[1]
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    for index_k in range(K):
        if index_k == 0:
            out = _outer_clip(mat1[:, index_k], mat2[index_k, :], M, N, BLOCK_SIZE_K=1, grid1=grid, grid2=grid, clip_args=clip_args)
        else:
            out.add_(_outer_clip(mat1[:, index_k], mat2[index_k, :], M, N, BLOCK_SIZE_K=1, grid1=grid, grid2=grid, clip_args=clip_args))
    return out.sum(dim=1)

