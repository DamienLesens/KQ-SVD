#custom Mistral model which stores queries in the cache in addition of keys and values
#works in pair with custom_cache_query
#adapted from llama_model_query.py for Mistral architecture

from transformers.models.mistral.modeling_mistral import (
    MistralModel,
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralConfig,
    apply_rotary_pos_emb,
    MistralMLP,
    MistralRMSNorm,
    MistralRotaryEmbedding
)
import torch
import torch.nn as nn
import math
from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sliding_window_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    sliding_window: int = 4096,
    **kwargs,
):
    """
    Sliding window attention implementation for Mistral models.
    Each token can only attend to the previous sliding_window tokens.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # Apply grouped query attention (GQA)
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    # Create sliding window mask
    if seq_len > 1:
        # Create causal mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        
        # Apply sliding window constraint
        if sliding_window is not None and sliding_window > 0:
            for i in range(seq_len):
                start_pos = max(0, i - sliding_window + 1)
                if start_pos > 0:
                    causal_mask[i, :start_pos] = float("-inf")
        
        causal_mask = causal_mask.to(attn_weights.device, dtype=attn_weights.dtype)
        attn_weights = attn_weights + causal_mask
    
    # Apply additional attention mask if provided
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output, attn_weights


class MistralAttentionQuery(MistralAttention):
    """
    Mistral attention module with query caching and sliding window attention.
    """
    
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.sliding_window = getattr(config, 'sliding_window', 4096)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "query": query_states}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Use sliding window attention for Mistral
        attention_interface: Callable = sliding_window_attention_forward
        self.config._attn_implementation = "eager"
        
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights


class MistralDecoderLayerQuery(MistralDecoderLayer):
    """
    Mistral decoder layer with query-aware attention.
    """
    
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MistralAttentionQuery(config=config, layer_idx=layer_idx)


class MistralModelQuery(MistralModel):
    """
    Mistral model with query caching support.
    """
    
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MistralDecoderLayerQuery(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class MistralForCausalLMQuery(MistralForCausalLM):
    """
    Mistral causal LM with query caching support.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModelQuery(config)