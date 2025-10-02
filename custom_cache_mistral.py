#cache class for storing queries in addition of keys and values for Mistral models
#works in pair with mistral_model_query
#adapted from custom_cache_query.py with sliding window support

import torch
from torch import nn
from transformers.cache_utils import Cache
from typing import Callable, List, Optional, Tuple, Union, Any, Dict
from transformers.utils.deprecation import deprecate_kwarg


class CustomCacheMistral(Cache):
    """
    Cache implementation for Mistral models with sliding window attention support.
    Stores queries, keys, and values with efficient sliding window management.
    """

    def __init__(self, sliding_window: int = 4096) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.sliding_window = sliding_window
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.query_cache: List[torch.Tensor] = []
        
        # For sliding window efficiency - track cache positions
        self._cache_positions: List[int] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.query_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration.
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.query_cache[layer_idx])

    def __len__(self):
        """
        Returns the number of layers in the cache.
        """
        return len(self.key_cache)

    def _apply_sliding_window(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply sliding window constraint to cache tensors.
        Keep only the last sliding_window tokens.
        """
        if tensor.size(-2) > self.sliding_window:
            # Keep only the last sliding_window tokens
            return tensor[..., -self.sliding_window:, :]
        return tensor

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Also stores query states if provided in cache_kwargs.
        
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass, including query states.

        Return:
            A tuple containing the updated key and value states.
        """
        # Extract query states from cache_kwargs if available
        query_states = None
        if cache_kwargs is not None:
            query_states = cache_kwargs.get("query", None)

        # Initialize cache for new layers
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            if query_states is not None:
                self.query_cache.append(query_states)
            else:
                # Initialize with zeros if no query provided
                self.query_cache.append(torch.zeros_like(key_states))
        else:
            # Concatenate new states with existing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            
            if query_states is not None:
                self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)

            # Apply sliding window constraint
            self.key_cache[layer_idx] = self._apply_sliding_window(self.key_cache[layer_idx])
            self.value_cache[layer_idx] = self._apply_sliding_window(self.value_cache[layer_idx])
            self.query_cache[layer_idx] = self._apply_sliding_window(self.query_cache[layer_idx])

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum length supported by the cache (sliding window size)."""
        return self.sliding_window

    def reset(self):
        """Resets the cache contents."""
        self._seen_tokens = 0
        self.key_cache.clear()
        self.value_cache.clear()
        self.query_cache.clear()
        self._cache_positions.clear()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.query_cache[layer_idx].device
            self.query_cache[layer_idx] = self.query_cache[layer_idx].index_select(0, beam_idx.to(device))

    def get_query_states(self, layer_idx: int) -> torch.Tensor:
        """
        Returns the cached query states for a specific layer.
        
        Args:
            layer_idx (int): The layer index to retrieve query states from.
            
        Returns:
            torch.Tensor: The cached query states for the specified layer.
        """
        if layer_idx < len(self.query_cache):
            return self.query_cache[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self.query_cache)} layers, attempted to access layer with index {layer_idx}")

    def crop(self, maximum_length: int):
        """
        Crops the cache to a maximum length.
        
        Args:
            maximum_length (int): The maximum length to keep in the cache.
        """
        # Ensure we don't exceed sliding window
        maximum_length = min(maximum_length, self.sliding_window)
        
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].size(-2) > maximum_length:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][..., -maximum_length:, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][..., -maximum_length:, :]
                self.query_cache[layer_idx] = self.query_cache[layer_idx][..., -maximum_length:, :]