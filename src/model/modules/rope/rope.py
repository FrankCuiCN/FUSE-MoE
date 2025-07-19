import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings
from config.config_template import ConfigTemplate

# Debug: Ask: Is this RoPE implementation safe BF16-safe?
# See: https://github.com/huggingface/transformers/pull/29285
class RoPE(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.head_size      = config.attn_head_size
        self.context_window = config.context_window
        # Define layers
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_size,
            max_seq_len=self.context_window,
            base=10000,  # Consider: Add this into ConfigTemplate
        )

    def forward(self, x):
        """
        in  shape: (batch_size, num_token, num_head, head_size)
        out shape: (batch_size, num_token, num_head, head_size)
        """
        x = self.rope(x)
        return x
