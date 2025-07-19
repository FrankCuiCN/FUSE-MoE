import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config_template import ConfigTemplate
from model.modules.rope.rope import RoPE


class SelfAttn(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.emb_size  = config.emb_size
        self.num_head  = config.attn_num_head
        self.head_size = config.attn_head_size
        self.num_block = config.num_block
        self.use_rope  = True  # Consider: Put this in ConfigTemplate
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
        num_head, head_size = self.num_head, self.head_size
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
        # (batch_size, num_head, num_token, head_size)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        # ----- #
        
        # ----- #
        # Stage: Self-Attention
        # ----- #
        # Compute attention scores
        attn = q @ k.transpose(2, 3)
        attn = attn / math.sqrt(head_size)
        # Perform causal masking
        # (1, 1, num_token, num_token)
        mask = torch.triu(torch.ones(
            num_token, num_token, dtype=torch.bool, device=attn.device
        ), diagonal=1)[None, None]
        attn = attn.masked_fill(mask, float("-inf"))
        # Compute attention weights
        attn = F.softmax(attn, dim=3)
        # Apply attention weights to v
        x = attn @ v
        # ----- #
        
        # ----- #
        # Stage: Re-assemble head outputs
        # ----- #
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_token, emb_size)
        # Output projection
        x = self.fc_2(x)
        # ----- #
        return x
