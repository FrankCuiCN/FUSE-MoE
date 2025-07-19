import torch
import random
import numpy as np
from config.config_template import ConfigTemplate


# Consider: Setting different random seed on different nodes to ensure IID
def set_random_seeds(config: ConfigTemplate):
    random.seed(config.repro_random_seed_value)
    np.random.seed(config.repro_random_seed_value + 1)
    torch.manual_seed(config.repro_random_seed_value + 2)
