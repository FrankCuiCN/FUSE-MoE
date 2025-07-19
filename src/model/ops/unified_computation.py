import torch


def causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def get_block_mask_info(num_token, block_size, device):
    # Get attributes
    num_block_q = num_token // block_size
    num_block_k = num_block_q
    # Define kv_num_blocks
    kv_num_blocks = torch.ones(num_block_q, dtype=torch.int32, device=device)
    kv_num_blocks = kv_num_blocks[None, None]
    # Define kv_indices
    kv_indices = torch.arange(num_block_q, dtype=torch.int32, device=device)
    kv_indices = kv_indices.view(1, 1, -1, 1).repeat(1, 1, 1, num_block_k)
    # Define full_kv_num_blocks
    full_kv_num_blocks = torch.arange(num_block_q, dtype=torch.int32, device=device)
    full_kv_num_blocks = full_kv_num_blocks[None, None]
    # Define full_kv_indices
    full_kv_indices = torch.arange(num_block_q, dtype=torch.int32, device=device)
    full_kv_indices = full_kv_indices.view(1, 1, 1, -1).repeat(1, 1, num_block_q, 1)
    return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices


def stitch_block_masks(set1, set2):
    # kv_num_blocks_1:      (1, num_head, num_block_q_1)
    # kv_indices_1:         (1, num_head, num_block_q_1, num_block_k_1)
    # full_kv_num_blocks_1: (1, num_head, num_block_q_1)
    # full_kv_indices_1:    (1, num_head, num_block_q_1, num_block_k_1)
    # kv_num_blocks_2:      (1, num_head, num_block_q_2)
    # kv_indices_2:         (1, num_head, num_block_q_2, num_block_k_2)
    # full_kv_num_blocks_2: (1, num_head, num_block_q_2)
    # full_kv_indices_2:    (1, num_head, num_block_q_2, num_block_k_2)
    # Get attributes
    kv_num_blocks_1, kv_indices_1, full_kv_num_blocks_1, full_kv_indices_1 = set1
    kv_num_blocks_2, kv_indices_2, full_kv_num_blocks_2, full_kv_indices_2 = set2
    num_block_q_1 = kv_indices_1.size(2)
    num_block_k_1 = kv_indices_1.size(3)
    num_block_q_2 = kv_indices_2.size(2)
    num_block_k_2 = kv_indices_2.size(3)
    num_block_q   = num_block_q_1 + num_block_q_2
    num_block_k   = num_block_k_1 + num_block_k_2
    device = kv_num_blocks_1.device
    # Workaround: Match num_head
    # Assumption: kv_indices_1.size(1) is 1 and kv_indices_2.size(1) is not
    # Debug: Find a better strategy
    num_head = kv_indices_2.size(1)
    kv_num_blocks_1      = kv_num_blocks_1.expand(-1, num_head, -1)
    kv_indices_1         = kv_indices_1.expand(-1, num_head, -1, -1)
    full_kv_num_blocks_1 = full_kv_num_blocks_1.expand(-1, num_head, -1)
    full_kv_indices_1    = full_kv_indices_1.expand(-1, num_head, -1, -1)
    # Stage 1: stitch kv_num_blocks
    kv_num_blocks = torch.cat([kv_num_blocks_1, kv_num_blocks_2], dim=-1)
    # Stage 2: stitch kv_indices
    kv_indices = torch.zeros(1, num_head, num_block_q, num_block_k, device=device, dtype=torch.int32)
    kv_indices[..., :num_block_q_1, :num_block_k_1] = kv_indices_1
    kv_indices[..., num_block_q_1:, :num_block_k_2] = kv_indices_2 + num_block_k_1
    # Stage 3: stitch full_kv_num_blocks
    full_kv_num_blocks = torch.cat([full_kv_num_blocks_1, full_kv_num_blocks_2], dim=-1)
    # Stage 4: stitch full_kv_indices
    full_kv_indices = torch.zeros(1, num_head, num_block_q, num_block_k, device=device, dtype=torch.int32)
    full_kv_indices[..., :num_block_q_1, :num_block_k_1] = full_kv_indices_1
    full_kv_indices[..., num_block_q_1:, :num_block_k_2] = full_kv_indices_2 + num_block_k_1
    return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices

def get_block_mask_info_attn(batch_size, num_head, num_token, block_size, device):
    set_ori = get_block_mask_info(num_token, block_size, device)
    set_out = set_ori
    for idx in range(batch_size * num_head - 1):
        set_out = stitch_block_masks(set_out, set_ori)
    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = set_out
    return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices

def slice_block_mask_info_attn(block_mask_info_attn, batch_size, num_token, block_size):
    """
    Trim a pre-computed causal-mask (block_mask_info_attn) so that it matches
    the current batch_size ≤ B_max and num_token ≤ T_max.
    """
    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = block_mask_info_attn

    assert num_token % block_size == 0, "num_token must be multiple of block_size"
    blocks_per_seq = num_token // block_size
    total_blocks   = batch_size * blocks_per_seq

    kv_num_blocks      = kv_num_blocks[..., :total_blocks].clone()
    kv_indices         = kv_indices[..., :total_blocks, :total_blocks].clone()
    full_kv_num_blocks = full_kv_num_blocks[..., :total_blocks].clone()
    full_kv_indices    = full_kv_indices[..., :total_blocks, :total_blocks].clone()
    return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices
