import torch
import triton
import triton.language as tl

from .clip_fn import get_clip_factor_triton, get_clip_factor_torch
from .utils import to_tl_type, supported_acc_dtypes


# modified from https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    tl_acc_dtype: tl.constexpr,   # accumulation dtype
    tl_out_dtype: tl.constexpr,   # output dtype
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl_acc_dtype)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl_acc_dtype)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl_acc_dtype)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl_acc_dtype)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl_acc_dtype)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y.to(tl_out_dtype), mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused_clip(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             tl_acc_dtype: tl.constexpr,   # accumulation dtype
                             tl_out_dtype: tl.constexpr,   # output dtype
                             clip_args_ptr: tl.pointer_type,    # pointer to the clip arguments
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl_acc_dtype)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl_acc_dtype)
    w = tl.load(W + cols, mask=mask).to(tl_acc_dtype)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx.to(tl_out_dtype), mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw_buffer = tl.load(DW, mask=mask)
        partial_dw_buffer_norm = tl.sqrt(partial_dw_buffer * partial_dw_buffer)
        clip_factor = get_clip_factor_triton(partial_dw_buffer_norm, clip_args_ptr)
        partial_dw_buffer = partial_dw_buffer * clip_factor
        partial_dw += partial_dw_buffer
        partial_db_buffer = tl.load(DB, mask=mask)
        partial_db_buffer_norm = tl.sqrt(partial_db_buffer * partial_db_buffer)
        clip_factor = get_clip_factor_triton(partial_db_buffer_norm, clip_args_ptr)
        partial_db_buffer = partial_db_buffer * clip_factor
        partial_db += partial_db_buffer
    tl.store(DW, partial_dw.to(tl_acc_dtype), mask=mask)
    tl.store(DB, partial_db.to(tl_acc_dtype), mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         tl_acc_dtype: tl.constexpr,   # accumulation dtype
                         tl_out_dtype: tl.constexpr,   # output dtype
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl_acc_dtype)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl_acc_dtype)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(tl_out_dtype), mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db.to(tl_out_dtype), mask=cols < N)


def layernorm_dw_clip(input: torch.Tensor, grad_output: torch.Tensor, weight: torch.Tensor, C: torch.Tensor, clamp_value: torch.Tensor):
    # Sum gradients over all dimensions except the normalized dimensions
    axes = list(i for i in range(grad_output.dim()) if grad_output.size(i) != weight.size(i))
    axes.pop(0) # pop the batch dimension
    grad_weight_preclip = (grad_output * input).sum(dim=axes, keepdim=True)    # (N, *)

    # Clip the gradient in-place
    axes = list(range(grad_weight_preclip.dim()))
    axes.pop(0) # pop the batch dimension
    grad_weight_norm = torch.norm(grad_weight_preclip, p=2, dim=axes, keepdim=True)
    clip_args = torch.Tensor([C, clamp_value]).to(grad_output.device)
    clip_factor = get_clip_factor_torch(grad_weight_norm, clip_args)
    grad_weight_preclip.mul_(clip_factor)
    
    # Accumulating the clipped gradients
    clipped_grad_weight = grad_weight_preclip.mean(0, keepdim=True)    # (1, *)

    return clipped_grad_weight


def layernorm_db_clip(grad_output: torch.Tensor, bias: torch.Tensor, C: torch.Tensor, clamp_value: torch.Tensor):
    axes = list(i for i in range(grad_output.dim()) if grad_output.size(i) != bias.size(i))
    axes.pop(0) # pop the batch dimension
    grad_bias_preclip = grad_output.sum(dim=axes, keepdim=True)

    axes = list(range(grad_bias_preclip.dim()))
    axes.pop(0) # pop the batch dimension
    grad_bias_norm = torch.norm(grad_bias_preclip, p=2, dim=axes, keepdim=True)
    clip_args = torch.Tensor([C, clamp_value]).to(grad_output.device)
    clip_factor = get_clip_factor_torch(grad_bias_norm, clip_args)
    grad_bias_preclip.mul_(clip_factor)

    # Accumulating the clipped gradients
    clipped_grad_bias = grad_bias_preclip.mean(0, keepdim=True)

    return clipped_grad_bias


def layernorm_fwd_fused_triton(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    # Determine the output dtype and accumulation dtype
    out_dtype = input.dtype
    if out_dtype in supported_acc_dtypes.keys():
        acc_dtype = supported_acc_dtypes[out_dtype][0]
    else:
        raise KeyError("In fused mode, the type of input is not supported. Set use_fused=False to use the default implementation.")
    tl_acc_dtype = to_tl_type(acc_dtype)
    tl_out_dtype = to_tl_type(out_dtype)
    # allocate output
    y = torch.empty_like(input)
    # reshape input data into 2D tensor
    x_arg = input.reshape(-1, input.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=acc_dtype, device=input.device)
    rstd = torch.empty((M, ), dtype=acc_dtype, device=input.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    _layer_norm_fwd_fused[(M, )](  #
        x_arg, y, weight, bias, mean, rstd,  #
        x_arg.stride(0), N, eps,  #
        tl_acc_dtype, tl_out_dtype,  #
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
    buffer = {
        'mean': mean,
        'rstd': rstd,
    }
    args = {
        'BLOCK_SIZE': BLOCK_SIZE,
        'num_warps': num_warps,
        'eps': eps,
    }
    return y, buffer, args

def layernorm_bwd_clip_fused_triton(
        grad_output: torch.Tensor, 
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: torch.Tensor, 
        clip_args: torch.Tensor,
        buffer: dict, args: dict):
    # Determine the output dtype and accumulation dtype
    out_dtype = grad_output.dtype
    if out_dtype in supported_acc_dtypes.keys():
        acc_dtype = supported_acc_dtypes[out_dtype][0]
    else:
        raise KeyError("In fused mode, the type of input is not supported. Set use_fused=False to use the default implementation.")
    tl_acc_dtype = to_tl_type(acc_dtype)
    tl_out_dtype = to_tl_type(out_dtype)
    # x, w, b, m, v = ctx.saved_tensors
    m, v = buffer['mean'], buffer['rstd']
    BLOCK_SIZE, num_warps = args['BLOCK_SIZE'], args['num_warps']
    # heuristics for amount of parallel reduction stream for DW/DB
    N = weight.shape[0]
    GROUP_SIZE_M = 64
    if N <= 8192: GROUP_SIZE_M = 96
    if N <= 4096: GROUP_SIZE_M = 128
    if N <= 1024: GROUP_SIZE_M = 256
    # allocate output
    locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=weight.device)
    dw_ = torch.zeros((GROUP_SIZE_M, N), dtype=acc_dtype, device=weight.device)
    db_ = torch.zeros((GROUP_SIZE_M, N), dtype=acc_dtype, device=bias.device)
    dx = torch.empty_like(grad_output)
    # enqueue kernel using forward pass heuristics
    # also compute partial sums for DW and DB
    x_arg = input.reshape(-1, input.shape[-1])
    M, N = x_arg.shape
    _layer_norm_bwd_dx_fused_clip[(M, )](  #
        dx, grad_output, dw_, db_, input, weight, m, v, locks,  #
        x_arg.stride(0), N,  #
        tl_acc_dtype, tl_out_dtype,  #
        clip_args,  #
        BLOCK_SIZE_N=BLOCK_SIZE,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
        num_warps=num_warps)
    args = {
        'GROUP_SIZE_M': GROUP_SIZE_M,
        'M': M,
    }
    return dx, dw_, db_, args


def layernorm_dwdb_fused_triton(
    weight: torch.Tensor, 
    bias: torch.Tensor,
    _dw: torch.Tensor,
    _db: torch.Tensor,
    args: dict = None,
):
    # Determine the output dtype and accumulation dtype
    out_dtype = weight.dtype
    if out_dtype in supported_acc_dtypes.keys():
        acc_dtype = supported_acc_dtypes[out_dtype][0]
    else:
        raise KeyError("In fused mode, the type of input is not supported. Set use_fused=False to use the default implementation.")
    tl_acc_dtype = to_tl_type(acc_dtype)
    tl_out_dtype = to_tl_type(out_dtype)
    N = weight.shape[0]
    GROUP_SIZE_M, M = args['GROUP_SIZE_M'], args['M']
    dw = torch.empty((N, ), dtype=weight.dtype, device=weight.device)
    db = torch.empty((N, ), dtype=weight.dtype, device=bias.device)
    grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
    # accumulate partial sums in separate kernel
    _layer_norm_bwd_dwdb[grid](
        _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
        tl_acc_dtype, tl_out_dtype,  #
        BLOCK_SIZE_M=32,  #
        BLOCK_SIZE_N=128, num_ctas=1)
    return dw, db

