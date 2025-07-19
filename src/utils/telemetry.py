import torch
from collections import defaultdict


class Telemetry:
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self._sum   = defaultdict(lambda: 0.0)  # holds running sums
        self._count = defaultdict(lambda: 0.0)  # holds running counts

    def reset(self):
        self._sum.clear()
        self._count.clear()

    def add(self, key, value):
        assert torch.is_tensor(value), "value must be a torch.Tensor"
        self._sum[key]   += value.detach().clone()
        self._count[key] += 1

    def get(self, key):
        # Note: Counts are identical across ranks
        #     Otherwise "out" might be biased
        out = self._sum[key] / self._count[key]
        out = self.accelerator.reduce(out, reduction="mean").cpu()
        return out
    
    def keys(self):
        return self._sum.keys()
