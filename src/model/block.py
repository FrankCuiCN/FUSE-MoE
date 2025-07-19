import torch.nn as nn
from config.config_template import ConfigTemplate
from model.modules.ffwd.build_ffwd import build_ffwd
from model.modules.attn.build_attn import build_attn
from model.modules.norm.build_norm import build_norm


class Block(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.profiler  = config.runtime["profiler"]
        self.ffwd_name = config.ffwd_name
        # Define layers
        self.attn      = None if self.ffwd_name == "Unified" else build_attn(config)
        self.norm_attn = None if self.ffwd_name == "Unified" else build_norm(config)
        self.ffwd      = build_ffwd(config)
        self.norm_ffwd = build_norm(config)

    def forward(self, x):
        """
        in  shape: (batch_size, num_token, emb_size)
        out shape: (batch_size, num_token, emb_size)
        """
        if self.ffwd_name == "Unified":
            x = x + self.ffwd(self.norm_ffwd(x))
        elif self.ffwd_name == "Unified-Naive":
            # BUG: in this case, the param init is still scaled by 1 / math.sqrt(2 * num_block)
            x = x + self.attn(self.norm_attn(x)) + self.ffwd(self.norm_ffwd(x))
        else:
            x = x + self.attn(self.norm_attn(x))
            x = x + self.ffwd(self.norm_ffwd(x))
        return x
