import torch
import torch.nn.functional as F


def prepare_packing(expert_assign, num_expert, block_size):
    """
    expert_assign: (batch_size, num_head, num_token)
    num_expert:    constant
    block_size:    constant
    """
    with torch.no_grad():
        # Set up variables
        batch_size = expert_assign.size(0)
        num_head   = expert_assign.size(1)
        num_token  = expert_assign.size(2)
        # Get expert_bincount
        # Note: This approach materializes a (batch_size, num_head, num_token, num_expert) tensor
        # (batch_size, num_head, num_expert)
        expert_bincount = F.one_hot(expert_assign, num_classes=num_expert).sum(dim=-2)
        # Get padding_needed
        # (batch_size, num_head, num_expert)
        padding_needed = (-expert_bincount) % block_size
        # Update padding_size to match padding_size across the batch_size and num_head dimensions
        # Note: Instead of pad to the last expert, we can potentially use this step to promote load balancing
        padding_size_all = padding_needed.sum(dim=-1)
        padding_size     = padding_size_all.max()
        padding_needed[:, :, -1] += padding_size - padding_size_all
        # Get offsets
        # (batch_size, num_head, 1)
        offsets = num_expert * torch.arange(batch_size * num_head, device=expert_assign.device).view(batch_size, num_head, 1)
        # Apply offsets to expert_assign
        # (batch_size, num_head, num_token)
        expert_assign = expert_assign + offsets
        # Flatten expert_assign
        # (batch_size * num_head * num_token,)
        expert_assign = expert_assign.view(batch_size * num_head * num_token)
        # Materialize padding_tensor
        # Consider: Find a more parallelizable repeat_interleave
        # (batch_size * num_head * padding_size,)
        padding_tensor = torch.arange(batch_size * num_head * num_expert, device=expert_assign.device)
        padding_tensor = torch.repeat_interleave(padding_tensor, padding_needed.view(batch_size * num_head * num_expert))
        # Apply padding to expert_assign
        # (batch_size * num_head * num_token + batch_size * num_head * padding_size)
        expert_assign = torch.concat((expert_assign, padding_tensor), dim=0)
        # Sort expert_assign and get mapping
        # Note: stable=True for determinism
        # Note: stable=False is functionally the same
        # (batch_size * num_head * num_token + batch_size * num_head * padding_size)
        expert_assign, mapping = torch.sort(expert_assign, stable=True)
        # Unflatten expert_assign and remove offsets
        # (batch_size, num_head, num_token + padding_size)
        expert_assign = expert_assign.view(batch_size, num_head, num_token + padding_size)
        expert_assign = expert_assign - offsets
        # Get mapping_inv
        # (batch_size * num_head * num_token + batch_size * num_head * padding_size,)
        mapping_inv = torch.empty_like(mapping)
        mapping_inv[mapping] = torch.arange(mapping.size(0), device=mapping.device)
        # Get num_block_q
        # Note: Should be provably divisible
        num_block_q = (num_token + padding_size) // block_size
        # Get block_level_expert_assign
        # (batch_size, num_head, num_block_q)
        block_level_expert_assign = expert_assign.view(batch_size, num_head, num_block_q, block_size)[..., 0]
        return mapping, mapping_inv, padding_size, block_level_expert_assign, expert_assign, expert_bincount
