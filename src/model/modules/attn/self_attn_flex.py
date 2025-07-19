import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from model.modules.rope.rope import RoPE
from config.config_template import ConfigTemplate
from model.ops.unified_computation import causal_mask_mod

flex_attention = torch.compile(flex_attention)
from_kv_blocks = torch.compile(BlockMask.from_kv_blocks)


class SelfAttnFlex(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.emb_size  = config.emb_size
        self.num_head  = config.attn_num_head
        self.head_size = config.attn_head_size
        self.num_block = config.num_block
        self.use_rope  = True  # Consider: Put this in ConfigTemplate
        self.block_size = config.perf_flex_attn_block_size
        self.block_mask_info_attn = config.runtime["block_mask_info_attn"]
        self.profiler = config.runtime["profiler"]
        # Validate attributes
        assert self.emb_size == self.num_head * self.head_size
        # Define layers
        self.fc_1 = nn.Linear(self.emb_size, self.emb_size * 3, bias=False)
        self.fc_2 = nn.Linear(self.emb_size, self.emb_size,     bias=False)
        self.rope = RoPE(config) if self.use_rope else None
        # Initialize parameters
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_2.weight, mean=0.0, std=0.02 / math.sqrt(2.0 * self.num_block))
        # Register weight decay parameters
        self.params_decay = list()
        self.params_decay.append(self.fc_1.weight)
        self.params_decay.append(self.fc_2.weight)

    def forward(self, x):
        """
        in  shape: (batch_size, num_token, emb_size)
        out shape: (batch_size, num_token, emb_size)
        """
        # ----- #
        # Stage: Get attributes
        # ----- #
        batch_size, num_token, emb_size = x.shape
        num_head   = self.num_head
        head_size  = self.head_size
        block_size = self.block_size
        # ----- #
        
        # ----- #
        # Stage: Get q, k, and v
        # ----- #
        # (batch_size, num_token, emb_size)
        q, k, v = self.fc_1(x).chunk(3, dim=2)
        # (batch_size, num_token, num_head, head_size)
        q = q.view(batch_size, num_token, num_head, head_size)
        k = k.view(batch_size, num_token, num_head, head_size)
        v = v.view(batch_size, num_token, num_head, head_size)
        # (batch_size, num_token, num_head, head_size)
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)
        # (num_head, batch_size, num_token, head_size)
        q = q.permute(2, 0, 1, 3).contiguous()
        k = k.permute(2, 0, 1, 3).contiguous()
        v = v.permute(2, 0, 1, 3).contiguous()
        # (1, num_head, batch_size * num_token, head_size)
        q = q.view(1, num_head, batch_size * num_token, head_size)
        k = k.view(1, num_head, batch_size * num_token, head_size)
        v = v.view(1, num_head, batch_size * num_token, head_size)
        # ----- #
        
        # ----- #
        # Stage: Self-Attention
        # ----- #
        # Get block_mask
        kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = self.block_mask_info_attn
        block_mask = from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            BLOCK_SIZE=block_size,
            mask_mod=causal_mask_mod,
        )
        # (1, num_head, batch_size * num_token, head_size)
        x = flex_attention(
            query=q,
            key=k,
            value=v,
            scale=1.0 / math.sqrt(head_size),
            block_mask=block_mask,
        )
        # ----- #
        
        # ----- #
        # Stage: Re-assemble head outputs
        # ----- #
        # (num_head, batch_size, num_token, head_size)
        x = x.view(num_head, batch_size, num_token, head_size)
        # (batch_size, num_token, num_head, head_size)
        x = x.permute(1, 2, 0, 3).contiguous()
        x = x.view(batch_size, num_token, emb_size)
        # Output projection
        x = self.fc_2(x)
        # ----- #
        return x
