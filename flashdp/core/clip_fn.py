import torch
import triton
import triton.language as tl


@torch.jit.script
def get_clip_factor_torch(norm, clip_args):
    return torch.clamp(clip_args[0] / (norm + 1e-10), max=clip_args[1])


@triton.jit
def tl_clamp(x, max):
    return tl.where(x < max, x, max)


@triton.jit
def get_clip_factor_triton(
    norm, 
    clip_args_ptr
):
    """
    clip_args_ptr: ptr of max_norm
    clip_args_ptr+1: ptr of clamp_value
    """
    return tl_clamp(tl.load(clip_args_ptr) / (norm + 1e-10), tl.load(clip_args_ptr+1))

