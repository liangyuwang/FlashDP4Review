import torch
import triton
import triton.language as tl

from .clip_fn import get_clip_factor_triton
from .utils import to_tl_type, supported_acc_dtypes


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['norm_ptr', 'counter_ptr'],
)
@triton.jit
def _mm_clip_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    norm_ptr, 
    counter_ptr, 
    M, 
    N, 
    K, 
    clip_args_ptr: tl.pointer_type,
    acc_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    mid = tl.program_id(0)
    nid = tl.program_id(1)

    num_blocks = (M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N)
    block_id = mid * (N // BLOCK_SIZE_N) + nid

    # Compute the norm's pointers
    norm_ptrs = norm_ptr + tl.arange(0, BLOCK_SIZE_K)

    # Compute the A and B pointers
    a_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    b_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + a_rows[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + b_cols[None, :]

    c_accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(K//BLOCK_SIZE_K):
        a = tl.load(a_ptrs).to(acc_dtype)
        b = tl.load(b_ptrs).to(acc_dtype)

        ab = a[:, :, None] * b[None, :, :]  # (BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)

        norm_register = tl.sum(ab * ab, axis=0)         # (BLOCK_SIZE_K, BLOCK_SIZE_N) vector
        norm_register = tl.sum(norm_register, axis=1)   # (BLOCK_SIZE_K) vector

        # Core step: we need to sum the norm_register in all blocks by offloading the norm_register to global memory
        # because we need to use the whole norm for matrix (M, BLOCK_SIZE_K, N) to compute the clip_factor
        # set the condition that only the first num_blocks blocks will offload the norm_register to global memory
        if block_id < num_blocks:
            tl.atomic_add(norm_ptrs, norm_register)
            tl.atomic_add(counter_ptr, 1)
        while tl.sum(tl.load(norm_ptrs))>0 and tl.load(counter_ptr) <= num_blocks:   # waiting for all block to finish, and then load from the HBM
            pass
        norm_accumulator = tl.load(norm_ptrs).to(acc_dtype)   # (BLOCK_SIZE_K) vector
        
        clip_factor = get_clip_factor_triton(norm_accumulator, clip_args_ptr)[None, :, None]

        c_accumulator += tl.sum(clip_factor * ab, axis=1)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

        # Reset the counter and the norm_accumulator
        if mid == 0 and nid == 0:
            tl.store(counter_ptr, 0)
            tl.store(norm_ptrs, tl.zeros_like(norm_accumulator))

    c = c_accumulator.to(out_dtype)

    # C's block's offsets
    c_ptrs = a_rows[:, None] * N + b_cols[None, :]
    tl.store(c_ptr + c_ptrs, c)

BLOCK_SIZE_K = 8

def mm_clip_triton(mat1: torch.Tensor, mat2: torch.Tensor, max_norm: float = 1.0):
    M, K, N = mat1.shape[0], mat1.shape[1], mat2.shape[1]
    out = torch.empty([M, N], device=mat1.device, dtype=mat1.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    norm_accumulator = torch.zeros(BLOCK_SIZE_K, device=mat1.device, dtype=mat1.dtype)
    counter = torch.zeros(1, device=mat1.device, dtype=mat1.dtype)
    _mm_clip_kernel[grid](mat1, mat2, out, norm_accumulator, counter, M, N, K, max_norm)
    return out
