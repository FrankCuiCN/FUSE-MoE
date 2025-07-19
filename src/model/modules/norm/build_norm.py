from config.config_template import ConfigTemplate
from .layer_norm import LayerNorm
from .rms_norm import RMSNorm


def build_norm(config: ConfigTemplate):
    if config.norm_name == "LayerNorm":
        return LayerNorm(config)
    elif config.norm_name == "RMSNorm":
        return RMSNorm(config)
    else:
        raise Exception("Unexpected norm_name")
