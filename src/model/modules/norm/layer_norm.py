import torch
import torch.nn as nn
from config.config_template import ConfigTemplate

class LayerNorm(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.emb_size   = config.emb_size
        self.use_affine = config.norm_use_affine
        self.use_bias   = config.norm_use_bias
        self.eps        = config.norm_eps
        # Define layers
        self.weight = nn.Parameter(torch.ones(self.emb_size)) if self.use_affine else None
        self.bias   = nn.Parameter(torch.zeros(self.emb_size)) if self.use_affine else None

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Debug: should we put eps inside or outside sqrt?
        x_norm = (x - mean) / (torch.sqrt(var) + self.eps)
        if self.weight is not None:
            x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
        return x_norm
