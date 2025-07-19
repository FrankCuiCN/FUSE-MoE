import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Tuple, Dict, Any


class MinimalGPTConfig(PretrainedConfig):
    model_type = "minimal-gpt"
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size


class MinimalGPTWrapper(PreTrainedModel):
    config_class = MinimalGPTConfig
    main_input_name = "input_ids"
    def __init__(self, model: nn.Module, config: MinimalGPTConfig):
        super().__init__(config)
        self.inner_model = model
        # Make sure parameters are registered so `.to()` etc. work as expected
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> CausalLMOutput:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = self.inner_model(input_ids, y=None)
        return CausalLMOutput(logits=logits)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # The compiled model does not use past_key_values; feed whole context
        return {"input_ids": input_ids}

    def _init_weights(self, *args, **kwargs):
        """Required by `PreTrainedModel` but irrelevant here since the wrapped
        model already has its own weights."""
        pass
