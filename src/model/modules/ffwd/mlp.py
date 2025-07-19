import math
import torch.nn as nn
from config.config_template import ConfigTemplate


class MLP(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Initialize attributes
        self.emb_size  = config.emb_size
        self.hid_size  = config.ffwd_hid_size
        self.use_bias  = config.ffwd_use_bias
        self.num_block = config.num_block
        # Define layers
        self.fc_in  = nn.Linear(self.emb_size, self.hid_size, bias=self.use_bias)
        self.fc_out = nn.Linear(self.hid_size, self.emb_size, bias=self.use_bias)
        self.activ  = nn.GELU()
        # Initialize parameters
        nn.init.normal_(self.fc_in.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02 / math.sqrt(2.0 * self.num_block))
        if self.use_bias:
            nn.init.zeros_(self.fc_in.bias)
            nn.init.zeros_(self.fc_out.bias)
        # Register weight decay parameters
        self.params_decay = list()
        self.params_decay.append(self.fc_in.weight)
        self.params_decay.append(self.fc_out.weight)

    def forward(self, x):
        """
        in  shape: (batch_size, num_token, emb_size)
        out shape: (batch_size, num_token, emb_size)
        """
        x = self.fc_in(x)
        x = self.activ(x)
        x = self.fc_out(x)
        return x
