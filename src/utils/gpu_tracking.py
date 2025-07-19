import threading
import time
import statistics as stats
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    NVML_TEMPERATURE_GPU,
)

_SAMPLER = None  # single global sampler (simple)
class _Sampler(threading.Thread):
    def __init__(self, idx=0, interval=0.05):
        super().__init__(daemon=True)
        self.idx = idx
        self.interval = interval
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(self.idx)
        try:
            while not self._halt.is_set():
                util = nvmlDeviceGetUtilizationRates(h)
                mem = nvmlDeviceGetMemoryInfo(h)
                temp = nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU)
                self.samples.append(
                    {
                        "t": time.perf_counter(),
                        "gpu_util": util.gpu,
                        "mem_util": util.memory,
                        "mem_used": mem.used,
                        "temp": temp,
                    }
                )
                time.sleep(self.interval)
        finally:
            nvmlShutdown()

    def stop(self):
        self._halt.set()
        self.join()


def start_gpu_tracking(device_idx: int = 0, interval: float = 0.05):
    """Begin background polling of GPU stats."""
    global _SAMPLER
    if _SAMPLER is not None:
        raise RuntimeError("Tracking already running")
    _SAMPLER = _Sampler(device_idx, interval)
    _SAMPLER.start()


def stop_gpu_tracking(ensure_sync: bool = True) -> dict:
    """Stop polling and return summary statistics."""
    global _SAMPLER
    if _SAMPLER is None:
        raise RuntimeError("No tracker running")

    # Flush CUDA kernels so we measure only completed work
    if ensure_sync:
        import torch

        torch.cuda.synchronize()

    _SAMPLER.stop()
    data = _SAMPLER.samples
    _SAMPLER = None

    # Collapse raw samples to something easy to log
    to_mb = lambda b: b / 2**20
    gpu_util = [s["gpu_util"] for s in data]
    mem_used = [s["mem_used"] for s in data]
    temp = [s["temp"] for s in data]
    elapsed_s = data[-1]["t"] - data[0]["t"]

    return {
        "duration_s": elapsed_s,
        "gpu_util_avg_%": stats.mean(gpu_util),
        "gpu_util_peak_%": max(gpu_util),
        "mem_used_peak_MB": to_mb(max(mem_used)),
        "mem_used_avg_MB": to_mb(stats.mean(mem_used)),
        "temp_peak_C": max(temp),
        "num_samples": len(data),
    }
