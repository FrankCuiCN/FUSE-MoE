import time
import torch
from collections import defaultdict

class Context:
    def __init__(self, label, owner, enabled=True):
        self.label = label
        self.owner = owner
        self.enabled = enabled

    def __enter__(self):
        if self.owner.enabled and self.enabled:
            self.owner.allocate_result(self.label)
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.owner.enabled and self.enabled:
            torch.cuda.synchronize()
            latency = time.perf_counter() - self.start_time
            self.owner.update_result(self.label, latency)


class Profiler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.results = defaultdict(lambda: {"count": 0, "total_latency": 0.0})

    def __call__(self, label="Region", enabled=True):
        return Context(label, self, enabled=enabled)
    
    def allocate_result(self, label):
        entry = self.results[label]

    def update_result(self, label, latency):
        entry = self.results[label]
        entry["count"] += 1
        entry["total_latency"] += latency

    def reset(self):
        self.results.clear()

    def print_summary(self):
        if self.enabled:
            for idx, (label, entry) in enumerate(self.results.items()):
                print("[{}] {:20s}: {:.4f} seconds ({} runs)".format(
                    idx, label, entry["total_latency"], entry["count"]
                ))
