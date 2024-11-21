import time
import torch
import triton.language as tl

def to_tl_type(ty):
    return getattr(tl, str(ty).split(".")[-1])


supported_acc_dtypes = {
    torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
    torch.float32: (torch.float32, ), torch.int8: (torch.int32, )
}


class RuntimeAutoTuner:
    def __init__(self, enable=True, warmup_iterations=10, measure_iterations=100, log=False) -> None:
        self.enable = enable
        self.if_final_tune = False
        self.chosen_func = None
        self.warmup_iterations = warmup_iterations
        self.measure_iterations = measure_iterations
        self.log = log

    def choose_function(self, funcs_list, *args, **kwargs):
        if not self.enable:
            return funcs_list[0]
        if self.chosen_func is not None and self.if_final_tune:
            return self.chosen_func
        time_list = []
        for func in funcs_list:
            for _ in range(self.warmup_iterations):
                func(*args, **kwargs)
            time_list.append(self._measure_time(func, *args, **kwargs))
        self.chosen_func = funcs_list[time_list.index(min(time_list))]
        if self.log:
            print(f"Chosen function: {self.chosen_func}")
        return self.chosen_func

    def final_tune(self):
        self.if_final_tune = True

    def _measure_time(self, func, *args, **kwargs):
        start = time.time()
        for _ in range(self.measure_iterations):
            func(*args, **kwargs)
        end = time.time()
        return (end - start) / self.measure_iterations
