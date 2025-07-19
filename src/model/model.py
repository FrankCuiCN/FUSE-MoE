import torch
import torch.nn as nn
from collections import defaultdict
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from config.config_template import ConfigTemplate
from model.block import Block
from model.modules.norm.build_norm import build_norm


class Model(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.vocab_size      = config.vocab_size
        self.num_class       = config.num_class
        self.context_window  = config.context_window
        self.emb_size        = config.emb_size
        self.num_block       = config.num_block
        self.use_liger       = config.perf_use_liger
        self.use_wpe         = False
        self.profiler        = config.runtime["profiler"]
        self.telemetry_local = defaultdict(lambda: 0.0)
        # Define layers (pre-processing stage)
        self.wte = nn.Embedding(self.vocab_size,     self.emb_size)
        self.wpe = nn.Embedding(self.context_window, self.emb_size) if self.use_wpe else None
        # Define layers (transformation stage)
        self.block_all = nn.ModuleList()
        for _ in range(self.num_block):
            self.block_all.append(Block(config))
        # Define layers (post-processing stage)
        self.norm_cls = build_norm(config)
        self.fc_cls   = nn.Linear(self.emb_size, self.num_class, bias=False)
        self.ce       = nn.CrossEntropyLoss(ignore_index=-1)
        self.liger    = LigerFusedLinearCrossEntropyLoss(ignore_index=-1)
        self.liger    = torch.compiler.disable(self.liger, recursive=True)  # BUG: Compling liger seems to cause numerical instability
                                                                            # Workaround: Do not torch.compile liger kernel
        # Initialize parameters
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        if self.use_wpe:
            nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_cls.weight, mean=0.0, std=0.02)
        # Register weight decay parameters
        self.params_decay = list()
        self.params_decay.append(self.wte.weight)
        self.params_decay.append(self.fc_cls.weight)
        if self.use_wpe:
            self.params_decay.append(self.wpe.weight)

    def forward(self, x, y=None):
        # ----- #
        # Stage: Pre-Processing
        # ----- #
        # (batch_size, num_token, emb_size)
        x = self.wte(x)
        # (batch_size, num_token, emb_size)
        if self.use_wpe:
            x = x + self.wpe(torch.arange(0, x.size(1), dtype=torch.long, device=x.device))
        # ----- #
        
        # ----- #
        # Stage: Transformation
        # ----- #
        loss_lb = torch.tensor(0.0, dtype=torch.float32, device=x.device)  # Ask: Which dtype should we use?
        for block in self.block_all:
            x = block(x)
            if hasattr(block.ffwd, "loss_lb") and self.training:
                loss_lb += block.ffwd.loss_lb
        # ----- #
        
        # ----- #
        # Stage: Post-Processing
        # ----- #
        x = self.norm_cls(x)
        if y is not None:
            # Training mode
            if self.use_liger:
                loss_lm = self.liger(self.fc_cls.weight, x.view(-1, self.emb_size), y.view(-1))
            else:
                logits = self.fc_cls(x)
                loss_lm = self.ce(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss_lm + loss_lb
            # Update telemetry_local
            self.telemetry_local["loss_lm"] = loss_lm.detach().clone()
            self.telemetry_local["loss_lb"] = loss_lb.detach().clone()
            return loss, self.telemetry_local
        else:
            # Inference mode
            logits = self.fc_cls(x)
            # Debug: also return telemetry_local, but ensure codebase-wide consistency
            return logits
        # ----- #
    
    def inference(self, x, max_new_tokens=50, eos_token_id=None):
        # Naive implementation; No KV Cache
        # Setup for inference
        generated = x
        with torch.no_grad():
            # Gen tokens up to the max number specified
            for _ in range(max_new_tokens):
                # Get the current length of the generated content
                # If the amount generated is less the context window, generate
                #     a new token and append after the EOT-striped inputs
                # Else, slide a window and append gen token to the end of x
                current_length = generated.shape[1]
                if current_length < self.context_window:
                    pad_length = self.context_window - current_length
                    pad_tokens = torch.full(
                        (generated.shape[0], pad_length),
                        eos_token_id,
                        dtype=generated.dtype,
                        device=generated.device,
                    )
                    input_ids = torch.cat([generated, pad_tokens], dim=1)
                    logits = self.forward(input_ids, y=None)
                    new_token_logits = logits[:, current_length - 1, :]
                else:
                    input_ids = generated[:, -self.context_window :]
                    logits = self.forward(input_ids, y=None)
                    new_token_logits = logits[:, -1, :]
                # Select the token with the highest probability and append
                next_token = torch.argmax(
                    new_token_logits, dim=-1, keepdim=True
                )
                generated = torch.cat([generated, next_token], dim=1)
                # If the newly generated token is EOT, break the loop.
                if eos_token_id is not None:
                    if (next_token == eos_token_id).all():
                        break
        # Return the generate text
        return generated
