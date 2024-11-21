import torch
import torch.nn as nn
import triton.language as tl


class DPParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, requires_dp=True):
        t = nn.Parameter.__new__(cls, data, requires_grad)
        t.requires_dp = requires_dp
        return t


def to_tl_type(ty):
    return getattr(tl, str(ty).split(".")[-1])

supported_acc_dtypes = {
    torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
    torch.float32: (torch.float32, ), torch.int8: (torch.int32, )
}
