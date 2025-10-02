#custom Llama model which stores queries in the cache in addition of keys and values
#works in pair with custom_cache_query

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaConfig,
    apply_rotary_pos_emb,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding
)
import torch
from torch import nn
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
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape #ok
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # in input query, key and value have shape (batch, h, T, d)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # key_states.transpose(2, 3) is shape (batch, h, d, Tkv), query is (batch, h, Tq, d)
    # the first to arguments are considered as batch
    # attn_weights is (batch, h, Tq, Tkv)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    if attention_mask is None:
        attn_bias = torch.zeros(attn_weights.shape, dtype=attn_weights.dtype, device=attn_weights.device)
        temp_mask = torch.ones(attn_weights.shape, dtype=torch.bool).tril(diagonal=0).to(attn_weights.device) #sets elements on the upper triang to False
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf")) #upper triang is set to -inf, lower is still 0
        attn_bias.to(attn_weights.dtype)
        attn_weights = attn_weights + attn_bias


    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)# dropout=0 by default
    attn_output = torch.matmul(attn_weights, value_states)# value_states is (batch, h, Tkv, d)
    #so here attn_output is (batch, h, Tq, d)
    attn_output = attn_output.transpose(1, 2).contiguous()
    #and now it is (batch, Tq, h, d)

    return attn_output, attn_weights


class LlamaAttentionQuery(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        input_shape = hidden_states.shape[:-1] # keep (batch,T)
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "query": query_states}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        #choice of the attention function
        attention_interface: Callable = eager_attention_forward
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
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights
    
class LlamaDecoderLayerQuery(LlamaDecoderLayer):

     def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config,layer_idx)

        self.self_attn = LlamaAttentionQuery(config=config, layer_idx=layer_idx)

class LlamaModelQuery(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayerQuery(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class LlamaForCausalLMQuery(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelQuery(config)