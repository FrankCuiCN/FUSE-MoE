from model.modules.ffwd.mlp import MLP
from model.modules.ffwd.smoe import SMoE
from model.modules.ffwd.unified import Unified
from config.config_template import ConfigTemplate


def build_ffwd(config: ConfigTemplate):
    if config.ffwd_name == "MLP":
        return MLP(config)
    elif config.ffwd_name == "SMoE":
        return SMoE(config)
    elif config.ffwd_name == "Unified":
        return Unified(config)
    else:
        raise Exception("Unexpected ffwd_name")
