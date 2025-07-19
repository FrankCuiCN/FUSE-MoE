from config.config_template import ConfigTemplate
from model.modules.attn.self_attn import SelfAttn
from model.modules.attn.self_attn_flex import SelfAttnFlex
from model.modules.attn.self_attn_fused import SelfAttnFused


def build_attn(config: ConfigTemplate):
    if config.attn_name == "SelfAttn":
        return SelfAttn(config)
    elif config.attn_name == "SelfAttnFlex":
        return SelfAttnFlex(config)
    elif config.attn_name == "SelfAttnFused":
        return SelfAttnFused(config)
    else:
        raise Exception("Unexpected attn_name")
