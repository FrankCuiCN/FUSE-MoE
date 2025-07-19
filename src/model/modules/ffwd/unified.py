import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from config.config_template import ConfigTemplate
from model.modules.rope.rope import RoPE
from model.ops.packing.packing import packing
from model.ops.packing.unpacking import unpacking
from model.ops.packing.prepare_packing import prepare_packing
from model.ops.packing.get_block_mask_info import get_block_mask_info
from model.ops.unified_computation import causal_mask_mod
from model.ops.unified_computation import stitch_block_masks

flex_attention = torch.compile(flex_attention)
from_kv_blocks = torch.compile(BlockMask.from_kv_blocks)


class Unified(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.emb_size             = config.emb_size
        self.num_head             = config.ffwd_num_head
        self.head_size            = config.ffwd_head_size
        self.num_expert           = config.ffwd_num_expert
        self.num_expert_active    = config.ffwd_num_expert_active
        self.expert_size          = config.ffwd_expert_size
        self.num_block            = config.num_block
        self.block_size           = config.perf_flex_attn_block_size
        self.block_mask_info_attn = config.runtime["block_mask_info_attn"]
        self.profiler             = config.runtime["profiler"]
        self.use_rope             = True  # Consider: Put this in ConfigTemplate
        self.context_window       = config.context_window
        self.batch_size_fwd       = config.batch_size_fwd
        if config.ffwd_activation == "gelu":
            self.use_gelu = True
        elif config.ffwd_activation == "relu":
            self.use_gelu = False
        else:
            raise Exception("Unexpected ffwd_activation")
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
        # Debug: We are assuming the batch_size == batch_size_fwd
        attn_threshold = config.batch_size_fwd * config.context_window
        attn_scale = 1.0 / math.sqrt(self.head_size)
        if self.use_gelu:
            def score_mod(score, batch, head, q_idx, k_idx):
                # noinspection PyTypeChecker
                return torch.where(q_idx < attn_threshold, score * attn_scale, torch.log1p(F.gelu(score)))
        else:
            def score_mod(score, batch, head, q_idx, k_idx):
                # noinspection PyTypeChecker
                return torch.where(q_idx < attn_threshold, score * attn_scale, torch.log(F.relu(score)))
        self.score_mod = score_mod
        # Define layers
        self.fc_in_fused  = nn.Linear(self.emb_size, 4 * self.emb_size + self.num_expert, bias=False)
        self.fc_out_fused = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.k_ffwd = nn.Parameter(0.02 * torch.randn(
            size=(1, self.num_head, self.num_expert * self.expert_size, self.head_size),
            dtype=torch.float32,
        ))
        self.v_ffwd = nn.Parameter(0.02 * torch.randn(
            size=(1, self.num_head, self.num_expert * self.expert_size, self.head_size),
            dtype=torch.float32,
        ))
        self.router_activ = F.sigmoid
        self.rope = RoPE(config) if self.use_rope else None
        # Initialize parameters
        nn.init.normal_(self.fc_in_fused.weight,  mean=0.0, std=0.02)
        nn.init.normal_(self.fc_out_fused.weight,  mean=0.0, std=0.02 / math.sqrt(1.0 * self.num_block))
        # Register weight decay parameters
        self.params_decay = list()
        self.params_decay.append(self.fc_in_fused.weight)
        self.params_decay.append(self.fc_out_fused.weight)
        self.params_decay.append(self.k_ffwd)
        self.params_decay.append(self.v_ffwd)

    def forward(self, x):
        """
        in shape:  (batch_size, num_token, emb_size)
        out shape: (batch_size, num_token, emb_size)
        """
        # ----- #
        # Token padding workaround; Test-time only
        # ----- #
        assert x.size(0) <= self.batch_size_fwd
        assert x.size(1) <= self.context_window
        ORI_DIM_0, ORI_DIM_1 = x.size(0), x.size(1)
        FLAG0 = ORI_DIM_0 < self.batch_size_fwd
        FLAG1 = ORI_DIM_1 < self.context_window
        if FLAG1:  # pad tokens (dim-1)
            assert not self.training
            pad1 = self.context_window - ORI_DIM_1
            x = torch.cat([x,
                           torch.zeros(ORI_DIM_0, pad1, x.size(2),
                                       device=x.device, dtype=x.dtype)], dim=1)
        if FLAG0:  # pad batch (dim-0)
            assert not self.training
            pad0 = self.batch_size_fwd - ORI_DIM_0
            x = torch.cat([x,
                           torch.zeros(pad0, self.context_window, x.size(2),
                                       device=x.device, dtype=x.dtype)], dim=0)
        # ----- #
        
        
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
        block_mask_info_attn = self.block_mask_info_attn
        # ----- #
        
        # ----- #
        # Fused input mapping
        # ----- #
        # (batch_size, num_token, 4 * emb_size + num_expert)
        q_ffwd, q_attn, k_attn, v_attn, router_logits = self.fc_in_fused(x).split(
            [emb_size, emb_size, emb_size, emb_size, num_expert], dim=-1
        )
        # ----- #
        
        # ----- #
        # Routing
        # ----- #
        # (batch_size, num_token, num_expert)
        # router_logits
        # (batch_size, num_token, num_expert_active)
        router_logits_topk, expert_assign = torch.topk(
            router_logits, num_expert_active, dim=-1,
            largest=True, sorted=False,  # Note: We don't need sorted=True
        )
        # ----- #
        
        # ----- #
        # Multi-head layer (Attention)
        # ----- #
        # (batch_size, num_token, emb_size)
        # q_attn, k_attn, v_attn = self.fc_mh_attn(x).chunk(3, dim=-1)
        # (batch_size, num_token, num_head, head_size)
        q_attn = q_attn.view(batch_size, num_token, num_head, head_size)
        k_attn = k_attn.view(batch_size, num_token, num_head, head_size)
        v_attn = v_attn.view(batch_size, num_token, num_head, head_size)
        # (batch_size, num_token, num_head, head_size)
        if self.use_rope:
            q_attn = self.rope(q_attn)
            k_attn = self.rope(k_attn)
        # (num_head, batch_size, num_token, head_size)
        q_attn = q_attn.permute(2, 0, 1, 3).contiguous()
        k_attn = k_attn.permute(2, 0, 1, 3).contiguous()
        v_attn = v_attn.permute(2, 0, 1, 3).contiguous()
        # (1, num_head, batch_size * num_token, head_size)
        q_attn = q_attn.view(1, num_head, batch_size * num_token, head_size)
        k_attn = k_attn.view(1, num_head, batch_size * num_token, head_size)
        v_attn = v_attn.view(1, num_head, batch_size * num_token, head_size)
        # ----- #
        
        # ----- #
        # Multi-head layer (Feedforward)
        # ----- #
        # (batch_size, num_token, emb_size)
        # q_ffwd = self.fc_mh(x)
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
        block_mask_info_ffwd = get_block_mask_info(block_level_expert_assign, num_expert, expert_size, block_size)
        kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = stitch_block_masks(block_mask_info_attn, block_mask_info_ffwd)
        block_mask = from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            BLOCK_SIZE=block_size,
            mask_mod=causal_mask_mod,
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
        Q = torch.concat((q_attn, q_ffwd), dim=2)
        K = torch.concat((k_attn, k_ffwd), dim=2).to(torch.bfloat16)
        V = torch.concat((v_attn, v_ffwd), dim=2).to(torch.bfloat16)
        # Note: Remember to explicitly set the scale=1.0, otherwise it defaults to 1.0 / math.sqrt(head_size)
        # q:   (1, num_head, batch_size * num_token + batch_size * num_token * num_expert_active + padding_size, head_size)
        # k:   (1, num_head, batch_size * num_token + num_expert * expert_size, head_size)
        # v:   (1, num_head, batch_size * num_token + num_expert * expert_size, head_size)
        # o:   (1, num_head, batch_size * num_token + batch_size * num_token * num_expert_active + padding_size, head_size)
        # lse: (1, num_head, batch_size * num_token * num_expert_active + padding_size)
        x, lse = flex_attention(
            query=Q,
            key=K,
            value=V,
            scale=1.0,
            block_mask=block_mask,
            score_mod=score_mod,
            return_lse=True,
        )
        # Make selection
        q_attn = x[:, :, :batch_size * num_token]
        q_ffwd = x[:, :, batch_size * num_token:]
        lse    = lse[:, :, batch_size * num_token:]
        # The reversal trick
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
        # Fused fc_out
        # ----- #
        # (1, batch_size * num_token, num_head, head_size)
        q_attn = q_attn.transpose(1, 2).contiguous()
        q_ffwd = q_ffwd.transpose(1, 2).contiguous()
        # (batch_size, num_token, emb_size)
        q_attn = q_attn.view(batch_size, num_token, emb_size)
        q_ffwd = q_ffwd.view(batch_size, num_token, emb_size)
        # (batch_size, num_token, emb_size + emb_size)
        x = torch.concat((q_attn, q_ffwd), dim=-1)
        # (batch_size, num_token, emb_size)
        x = self.fc_out_fused(x)
        # ----- #
        

        # ----- #
        # Token padding workaround; Test-time only
        # ----- #
        if FLAG0 or FLAG1:
            x = x[:ORI_DIM_0, :ORI_DIM_1, :]
        # ----- #
        return x
