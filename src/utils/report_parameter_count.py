def report_parameter_count(module, verbose=False):
    total_trainable = 0
    total_non_embed = 0
    for name, p in module.named_parameters():
        if p.requires_grad:  # Trainable params
            total_trainable += p.numel()
            if ("wte" not in name) and ("wpe" not in name):  # Non-embed params
                total_non_embed += p.numel()
        if verbose:
            print(name, p.numel(), p.requires_grad)
    print("total_trainable:", f"{total_trainable:,}")
    print("total_non_embed:", f"{total_non_embed:,}")
    return total_trainable, total_non_embed
