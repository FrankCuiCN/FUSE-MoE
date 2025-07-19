import torch
import accelerate


def synchronize():
    # Note: It seems "wait_for_everyone()" does not wait for CUDA operations
    #     Therefore, explicitly calling "torch.cuda.synchronize" beforehand
    torch.cuda.synchronize()
    accelerate.utils.wait_for_everyone()
