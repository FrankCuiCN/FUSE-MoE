import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from config.config_template import ConfigTemplate
from model.ops.packing.packing import packing
from model.ops.packing.unpacking import unpacking
from model.ops.packing.prepare_packing import prepare_packing
from model.ops.packing.get_block_mask_info import get_block_mask_info

flex_attention = torch.compile(flex_attention)
from_kv_blocks = torch.compile(BlockMask.from_kv_blocks)


class SMoE(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.emb_size            = config.emb_size
        self.num_head            = config.ffwd_num_head
        self.head_size           = config.ffwd_head_size
        self.num_expert          = config.ffwd_num_expert
        self.num_expert_active   = config.ffwd_num_expert_active
        self.expert_size         = config.ffwd_expert_size
        self.num_block           = config.num_block
        self.block_size          = config.perf_flex_attn_block_size
        self.use_bf16_weights    = False
        if config.ffwd_activation == "gelu":
            self.use_gelu = True
            self.use_reversal_trick = True
        elif config.ffwd_activation == "relu":
            self.use_gelu = False
            self.use_reversal_trick = True
        elif config.ffwd_activation == "softmax":
            self.use_gelu = False
            self.use_reversal_trick = False
        else:
            raise Exception("Unexpected ffwd_activation")
        self.router_activ_name   = config.ffwd_gating_function
        self.profiler            = config.runtime["profiler"]
        # Validate attributes
        assert self.expert_size % self.block_size == 0
        assert self.emb_size == self.num_head * self.head_size
        # Load balancing loss
        # Note: register_buffer is checkpoint-friendly
        # Debug: Not sure if pytorch buffers are synchronized across nodes in DDP
        #   Also, not sure if we should synchronize expert_load_ema
        self.register_buffer("expert_load_ema", None)
        self.register_buffer("loss_lb", None)
        self.lbl_alpha = 0.01
        self.lbl_ema_factor = 0.99
        # Define score_mod
        if config.ffwd_activation == "gelu":
            def score_mod(score, batch, head, q_idx, k_idx):
                return torch.log1p(F.gelu(score))
        elif config.ffwd_activation == "relu":
            def score_mod(score, batch, head, q_idx, k_idx):
                return torch.log(F.relu(score))
        elif config.ffwd_activation == "softmax":
            score_mod = None
        else:
            raise Exception("Unexpected ffwd_activation")
        self.score_mod = score_mod
        # Define layers
        self.router = nn.Linear(self.emb_size, self.num_expert, bias=False)
        self.fc_mh  = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.fc_mg  = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.k_ffwd = nn.Parameter(0.02 * torch.randn(
            size=(1, self.num_head, self.num_expert * self.expert_size, self.head_size),
            dtype=torch.bfloat16 if self.use_bf16_weights else torch.float32,
        ))
        self.v_ffwd = nn.Parameter(0.02 * torch.randn(
            size=(1, self.num_head, self.num_expert * self.expert_size, self.head_size),
            dtype=torch.bfloat16 if self.use_bf16_weights else torch.float32,
        ))
        if self.router_activ_name == "sigmoid":
            self.router_activ = F.sigmoid
        elif self.router_activ_name == "softplus":
            self.router_activ = F.softplus
        else:
            raise Exception("Unexpected router_activ_name")
        # Initialize parameters
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_mh.weight,  mean=0.0, std=0.02)
        nn.init.normal_(self.fc_mg.weight,  mean=0.0, std=0.02 / math.sqrt(2.0 * self.num_block))
        # Register weight decay parameters
        self.params_decay = list()
        self.params_decay.append(self.router.weight)
        self.params_decay.append(self.fc_mh.weight)
        self.params_decay.append(self.fc_mg.weight)
        self.params_decay.append(self.k_ffwd)
        self.params_decay.append(self.v_ffwd)
    
    def update_expert_load_ema(self, expert_load):
        # expert_load: (num_expert,)
        if self.expert_load_ema is None:
            self.expert_load_ema = torch.ones(self.num_expert, dtype=torch.float32, device=expert_load.device) / self.num_expert
        self.expert_load_ema = self.lbl_ema_factor * self.expert_load_ema + (1.0 - self.lbl_ema_factor) * expert_load
        return self.expert_load_ema

    def forward(self, x):
        """
        in shape:  (batch_size, num_token, emb_size)
        out shape: (batch_size, num_token, emb_size)
        """
        # ----- #
        # Get attributes
        # ----- #
        batch_size        = x.size(0)
        num_token         = x.size(1)
        emb_size          = x.size(2)
        num_head          = self.num_head
        head_size         = self.head_size
        num_expert        = self.num_expert
        expert_size       = self.expert_size
        num_expert_active = self.num_expert_active
        block_size        = self.block_size
        k_ffwd            = self.k_ffwd
        v_ffwd            = self.v_ffwd
        score_mod         = self.score_mod
        # ----- #
        
        # ----- #
        # Routing
        # ----- #
        # (batch_size, num_token, num_expert)
        router_logits = self.router(x)
        # (batch_size, num_token, num_expert_active)
        router_logits_topk, expert_assign = torch.topk(
            router_logits, num_expert_active, dim=-1,
            largest=True, sorted=False,  # Note: We don't need sorted=True
        )
        # ----- #
        
        # ----- #
        # Multi-head layer
        # ----- #
        # (batch_size, num_token, emb_size)
        q_ffwd = self.fc_mh(x)
        # (batch_size * num_token, num_head, head_size)
        q_ffwd = q_ffwd.view(batch_size * num_token, num_head, head_size)
        # (num_head, batch_size * num_token, head_size)
        q_ffwd = q_ffwd.transpose(0, 1).contiguous()
        # (1, num_head, batch_size * num_token, head_size)
        q_ffwd = q_ffwd.view(1, num_head, batch_size * num_token, head_size)
        # ----- #
        
        # ----- #
        # Duplicate and Flatten Out
        # ----- #
        # (1, num_head, batch_size * num_token, 1, head_size)
        q_ffwd = q_ffwd.view(1, num_head, batch_size * num_token, 1, head_size)
        # (1, num_head, batch_size * num_token, num_expert_active, head_size)
        q_ffwd = q_ffwd.expand(-1, -1, -1, num_expert_active, -1).contiguous()
        # (1, num_head, batch_size * num_token * num_expert_active, head_size)
        q_ffwd = q_ffwd.view(1, num_head, batch_size * num_token * num_expert_active, head_size)
        # (1, 1, batch_size * num_token * num_expert_active)
        expert_assign = expert_assign.view(1, 1, batch_size * num_token * num_expert_active)
        # ----- #
        
        # ----- #
        # Packing
        # ----- #
        # In:
        #   - expert_assign: (1, 1, batch_size * num_token * num_expert_active)
        #   - num_expert:    constant
        #   - block_size:    constant
        # Out:
        #   - expert_assign:             (1, 1, batch_size * num_token * num_expert_active + padding_size)
        #   - expert_bincount:           (1, 1, num_expert)
        #   - block_level_expert_assign: (1, 1, num_block_q)
        mapping, mapping_inv, padding_size, block_level_expert_assign, expert_assign, expert_bincount = prepare_packing(expert_assign, num_expert, block_size)
        # (1, num_head, batch_size * num_token * num_expert_active + padding_size, head_size)
        q_ffwd = packing(q_ffwd, padding_size, mapping)
        # ----- #
        
        # ----- #
        # Construct block_mask
        # ----- #
        # (1, num_head, num_block_q)
        block_level_expert_assign = block_level_expert_assign.expand(-1, num_head, -1)
        kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = get_block_mask_info(block_level_expert_assign, num_expert, expert_size, block_size)
        block_mask = from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            BLOCK_SIZE=block_size,
            mask_mod=None,
        )
        # ----- #
        
        # ----- #
        # Load Balancing Loss
        # ----- #
        if self.training:
            # Get expert_load
            with torch.no_grad():
                # (num_expert,)
                expert_load = expert_bincount.view(num_expert)
                expert_load = expert_load / (num_token * batch_size * num_expert_active)
                # Note: We decide to disable EMA on expert_load
                """
                if self.training:
                    expert_load = self.update_expert_load_ema(expert_load)
                """
            # Get expert_prob
            # (batch_size, num_token, num_expert)
            expert_prob = router_logits.softmax(dim=-1)
            # (num_expert,)
            expert_prob = expert_prob.mean(dim=(0, 1))
            # Update loss_lb
            self.loss_lb = self.lbl_alpha * num_expert * (expert_load * expert_prob).sum()
        # ----- #
        
        # ----- #
        # Flex Attention
        # ----- #
        Q = q_ffwd
        K = k_ffwd if self.use_bf16_weights else k_ffwd.to(torch.bfloat16)
        V = v_ffwd if self.use_bf16_weights else v_ffwd.to(torch.bfloat16)
        # Note: Remember to explicitly set the scale=1.0, otherwise it defaults to 1.0 / math.sqrt(head_size)
        # q:   (1, num_head, batch_size * num_token * num_expert_active + padding_size, head_size)
        # k:   (1, num_head, num_expert * expert_size, head_size)
        # v:   (1, num_head, num_expert * expert_size, head_size)
        # o:   (1, num_head, batch_size * num_token * num_expert_active + padding_size, head_size)
        # lse: (1, num_head, batch_size * num_token * num_expert_active + padding_size)
        q_ffwd, lse = flex_attention(
            query=Q,
            key=K,
            value=V,
            scale=1.0 if self.use_reversal_trick else 1.0 / math.sqrt(self.head_size),
            block_mask=block_mask,
            score_mod=score_mod if self.use_reversal_trick else None,
            return_lse=True,
        )
        if self.use_reversal_trick:
            # (1, num_head, batch_size * num_token * num_expert_active + padding_size, head_size)
            q_ffwd = q_ffwd * lse[:, :, :, None].exp()
            if self.use_gelu:
                # (batch_size * num_token * num_expert_active + padding_size,)
                expert_assign = expert_assign.view(batch_size * num_token * num_expert_active + padding_size)
                q_ffwd = q_ffwd - self.v_ffwd.view(1, num_head, num_expert, expert_size, head_size).sum(dim=-2)[:, :, expert_assign]
        # ----- #
        
        # ----- #
        # Unpacking
        # ----- #
        # (1, num_head, batch_size * num_token * num_expert_active, head_size)
        q_ffwd = unpacking(q_ffwd, padding_size, mapping_inv)
        # ----- #
        
        # ----- #
        # Top-k averaging
        # ----- #
        # (1, num_head, batch_size * num_token, num_expert_active, head_size)
        q_ffwd = q_ffwd.view(1, num_head, batch_size * num_token, num_expert_active, head_size)
        # (1, 1, batch_size * num_token, num_expert_active, 1)
        router_logits_topk = router_logits_topk.view(1, 1, batch_size * num_token, num_expert_active, 1)
        # (1, num_head, batch_size * num_token, num_expert_active, head_size)
        q_ffwd = q_ffwd * self.router_activ(router_logits_topk)
        # (1, num_head, batch_size * num_token, head_size)
        q_ffwd = q_ffwd.sum(dim=-2)
        # ----- #
        
        # ----- #
        # Merge layer
        # ----- #
        # (1, batch_size * num_token, num_head, head_size)
        q_ffwd = q_ffwd.transpose(1, 2).contiguous()
        # (batch_size, num_token, emb_size)
        q_ffwd = q_ffwd.view(batch_size, num_token, emb_size)
        # (batch_size, num_token, emb_size)
        q_ffwd = self.fc_mg(q_ffwd)
        # ----- #
        return q_ffwd
