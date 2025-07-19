import torch
from tqdm import tqdm

# Important assumptions for unbiased loss estimation:
#     (1) All batches have the same batch size
#     (2) All nodes have the same number of iterations
# Debug: Make changes to alleviate these assumptions in a future version
def validate(early_stopping, payload):
    config, model, dataloader_val, accelerator, len_dataloader_val = payload
    loss_avg = torch.zeros(1, dtype=torch.float32, device=accelerator.device)
    device   = accelerator.device
    use_autocast = config.perf_use_autocast
    token_count_per_node = 0
    token_limit_per_node = 5_242_880 // accelerator.num_processes
    # Set the model to evaluation mode
    model.eval()
    for idx_iter, (inputs, targets) in enumerate(tqdm(
        dataloader_val,
        desc="Validating model",
        total=len_dataloader_val,
        disable=not accelerator.is_main_process,
    )):
        # Ask: Should we use non_blocking?
        inputs  =  inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        # Forward pass
        # Ask: Should we use model.no_sync() here?
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16, enabled=use_autocast):
                loss, telemetry_local = model(inputs, targets)
                loss_avg += telemetry_local["loss_lm"]
        # Early stopping
        token_count_per_node += inputs.numel()
        if early_stopping:
            if token_count_per_node >= token_limit_per_node:
                break
    # Normalize and synchronize loss_avg
    loss_avg = accelerator.gather(loss_avg / (idx_iter + 1)).mean().item()
    return loss_avg
