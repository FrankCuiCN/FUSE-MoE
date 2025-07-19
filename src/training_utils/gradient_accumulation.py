import torch
from contextlib import nullcontext


def gradient_accumulation(config, inputs, targets, model, accelerator):
    # Make loss_avg on-device to avoid ".item()" call in accumulation loop
    # See: https://github.com/karpathy/nanoGPT/pull/207#issuecomment-1506348629
    loss_avg     = torch.zeros(1, dtype=torch.float32, device=accelerator.device)
    accu_steps   = config.accu_steps
    device       = accelerator.device
    profiler     = config.runtime["profiler"]
    use_autocast = config.perf_use_autocast
    # Chunk data
    inputs  = torch.chunk(inputs,  accu_steps, dim=0)
    targets = torch.chunk(targets, accu_steps, dim=0)
    for idx in range(accu_steps):
        # Ask: Should we use non_blocking?
        inputs_current  =  inputs[idx].to(device=device, non_blocking=True)
        targets_current = targets[idx].to(device=device, non_blocking=True)
        # Note: No sync except for the last step
        # Ask: Should we apply no_sync to the forward pass? Or backward only?
        ctx_manager = model.no_sync if idx < (accu_steps - 1) else nullcontext
        with ctx_manager():
            # Forward pass
            with profiler("forward pass"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    # Note: Defer division by accu_steps for numerical stability
                    loss, telemetry_local = model(inputs_current, targets_current)
            # Backward pass
            with profiler("backward pass"):
                accelerator.backward(loss)
        # Update loss_avg
        loss_avg += telemetry_local["loss_lm"]
        # Update telemetry
        # Debug: Try to unify the telemetry strategy for grad_accu, validate, and other metrics
        config.runtime["telemetry"].add("load_balancing_loss", telemetry_local["loss_lb"])
    # Normalize and synchronize loss_avg
    loss_avg = accelerator.gather(loss_avg / accu_steps).mean().item()
    # Normalize the gradients by accu_steps
    with profiler("normalize grad"):
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= accu_steps
    return loss_avg
