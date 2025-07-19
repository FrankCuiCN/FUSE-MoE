import torch


def get_block_mask_info(block_level_expert_assign, num_expert, expert_size, block_size):
    """
    block_level_expert_assign: (batch_size, num_head, num_block_q)
    Note: Potential improvement: kv_indices could be sparse
    """
    # Set up variables
    batch_size = block_level_expert_assign.size(0)
    num_head = block_level_expert_assign.size(1)
    num_block_q = block_level_expert_assign.size(2)
    num_block_per_expert = expert_size // block_size
    num_block_k = num_expert * num_block_per_expert
    device = block_level_expert_assign.device
    # Get kv_num_blocks
    kv_num_blocks = torch.zeros(
        size=(batch_size, num_head, num_block_q),
        dtype=torch.int32,  # int32 for triton compatibility
        device=device,
    )
    # Get kv_indices
    kv_indices = torch.zeros(
        size=(batch_size, num_head, num_block_q, num_block_k),
        dtype=torch.int32,  # int32 for triton compatibility
        device=device,
    )
    # Get full_kv_num_blocks
    full_kv_num_blocks = num_block_per_expert * torch.ones(
        size=(batch_size, num_head, num_block_q),
        dtype=torch.int32,  # int32 for triton compatibility
        device=device,
    )
    # Get full_kv_indices
    # (num_block_k,)
    full_kv_indices = torch.arange(
        num_block_k,
        dtype=torch.int32,  # int32 for triton compatibility
        device=device,
    )
    # (1, 1, 1, num_block_k)
    full_kv_indices = full_kv_indices.view(1, 1, 1, num_block_k)
    # (batch_size, num_head, 1, num_block_k)
    full_kv_indices = full_kv_indices.expand(batch_size, num_head, -1, -1)
    # (batch_size, num_head, num_block_q, 1)
    offsets = num_block_per_expert * block_level_expert_assign.view(batch_size, num_head, num_block_q, 1).to(torch.int32)
    # (batch_size, num_head, num_block_q, num_block_k)
    full_kv_indices = full_kv_indices + offsets
    return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices
