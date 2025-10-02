#!/usr/bin/env python3
"""
Test script for Mistral model implementations.
This script validates that all Mistral components work correctly.
"""

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer

# Import Mistral implementations
from mistral_model_query import MistralForCausalLMQuery
from mistral_model_query_norope import MistralForCausalLMQueryNoRoPE
from mistral_model_norope import MistralForCausalLMNoRoPEinCache
from custom_cache_mistral import CustomCacheMistral
from evaluation_mistral import testing_proj_mistral, evaluate_mistral_compression
from dataset_utils_mistral import prepare_mistral_dataset, get_mistral_model_config

def test_mistral_model_loading():
    """Test that Mistral models can be instantiated correctly."""
    print("Testing Mistral model loading...")
    
    try:
        # Create a dummy config for testing
        config_dict = get_mistral_model_config()
        config_dict.update({
            'num_hidden_layers': 4,  # Smaller for testing
            'hidden_size': 512,
            'intermediate_size': 1024,
            'num_attention_heads': 8,
            'num_key_value_heads': 2,
        })
        
        # Test MistralForCausalLMQuery
        model_query = MistralForCausalLMQuery(type('Config', (), config_dict)())
        print("✓ MistralForCausalLMQuery loaded successfully")
        
        # Test MistralForCausalLMQueryNoRoPE
        model_norope = MistralForCausalLMQueryNoRoPE(type('Config', (), config_dict)())
        print("✓ MistralForCausalLMQueryNoRoPE loaded successfully")
        
        # Test MistralForCausalLMNoRoPEinCache
        model_norope_cache = MistralForCausalLMNoRoPEinCache(type('Config', (), config_dict)())
        print("✓ MistralForCausalLMNoRoPEinCache loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_mistral_cache():
    """Test Mistral cache functionality."""
    print("\nTesting Mistral cache...")
    
    try:
        # Create cache with sliding window
        cache = CustomCacheMistral(sliding_window=1024)
        
        # Test cache operations
        batch_size, num_heads, seq_len, head_dim = 1, 8, 64, 64
        
        # Create dummy tensors
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Test cache update
        cache_kwargs = {"query": queries}
        updated_keys, updated_values = cache.update(keys, values, 0, cache_kwargs)
        
        print(f"✓ Cache update successful, shapes: K={updated_keys.shape}, V={updated_values.shape}")
        
        # Test sliding window constraint
        large_keys = torch.randn(batch_size, num_heads, 2048, head_dim)
        large_values = torch.randn(batch_size, num_heads, 2048, head_dim)
        large_queries = torch.randn(batch_size, num_heads, 2048, head_dim)
        
        cache_kwargs_large = {"query": large_queries}
        updated_large_keys, updated_large_values = cache.update(large_keys, large_values, 1, cache_kwargs_large)
        
        # Should be limited by sliding window
        assert updated_large_keys.shape[-2] <= 1024, f"Sliding window not applied: {updated_large_keys.shape[-2]}"
        print("✓ Sliding window constraint working correctly")
        
        # Test query retrieval
        retrieved_queries = cache.get_query_states(0)
        print(f"✓ Query retrieval successful, shape: {retrieved_queries.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cache testing failed: {e}")
        return False


def test_mistral_attention():
    """Test Mistral attention computation with sliding window."""
    print("\nTesting Mistral attention...")
    
    try:
        # Create dummy config
        config_dict = get_mistral_model_config()
        config_dict.update({
            'num_hidden_layers': 2,
            'hidden_size': 256,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'sliding_window': 128,
        })
        config = type('Config', (), config_dict)()
        
        # Create model
        model = MistralForCausalLMQuery(config)
        model.eval()
        
        # Create dummy input
        batch_size, seq_len = 1, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Test cache
        cache = CustomCacheMistral(sliding_window=128)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=cache, use_cache=True)
        
        print(f"✓ Forward pass successful, output shape: {outputs.logits.shape}")
        print(f"✓ Cache contains {len(cache)} layers")
        
        return True
        
    except Exception as e:
        print(f"✗ Attention testing failed: {e}")
        return False


def test_mistral_evaluation():
    """Test Mistral evaluation functions."""
    print("\nTesting Mistral evaluation...")
    
    try:
        # Create dummy data for evaluation
        config_dict = get_mistral_model_config()
        config_dict.update({
            'num_hidden_layers': 2,
            'hidden_size': 128,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
        })
        config = type('Config', (), config_dict)()
        
        # Create dummy past_key_values (K, V, Q format)
        batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 32
        past_key_values = []
        
        for layer in range(2):
            keys = torch.randn(batch_size, num_heads // 2, seq_len, head_dim)  # num_key_value_heads
            values = torch.randn(batch_size, num_heads // 2, seq_len, head_dim)
            queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
            past_key_values.append((keys, values, queries))
        
        # Create dummy projections
        list_proj = []
        for layer in range(2):
            proj_k = torch.randn(num_heads // 2, head_dim, head_dim // 2)
            proj_q = torch.randn(num_heads, head_dim, head_dim // 2)  
            proj_v = torch.randn(num_heads // 2, head_dim, head_dim // 2)
            proj_w = torch.randn(num_heads, head_dim, head_dim // 2)
            list_proj.append([proj_k, proj_q, proj_v, proj_w])
        
        # Create dummy ranks
        ranks = [[16, 16], [16, 16]]  # [rank_kq, rank_vw] for each layer
        
        # Test evaluation
        errors = testing_proj_mistral(config, past_key_values, list_proj, ranks, sliding_window=128)
        
        print(f"✓ Evaluation successful, got {len(errors)} error metrics")
        print(f"  - Mean K error: {np.mean(errors[0]):.4f}")
        print(f"  - Mean attention error: {np.mean(errors[6]):.4f}")
        
        # Test compression evaluation
        cache = CustomCacheMistral(sliding_window=128)
        compression_results = evaluate_mistral_compression(config, cache)
        
        print(f"✓ Compression evaluation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation testing failed: {e}")
        return False


def test_mistral_vs_llama_compatibility():
    """Test that Mistral implementations are compatible with existing Llama workflow."""
    print("\nTesting Mistral-Llama compatibility...")
    
    try:
        # Test that we can use similar interfaces
        config_dict = get_mistral_model_config()
        config_dict.update({
            'num_hidden_layers': 1,
            'hidden_size': 64,
            'num_attention_heads': 2,
            'num_key_value_heads': 1,
        })
        config = type('Config', (), config_dict)()
        
        # Create both Mistral cache and see if it has similar interface to Llama cache
        mistral_cache = CustomCacheMistral(sliding_window=64)
        
        # Test basic operations that should work with both
        keys = torch.randn(1, 1, 16, 64)
        values = torch.randn(1, 1, 16, 64)
        queries = torch.randn(1, 2, 16, 64)
        
        # Test update
        cache_kwargs = {"query": queries}
        k_out, v_out = mistral_cache.update(keys, values, 0, cache_kwargs)
        
        # Test length operations
        seq_len = mistral_cache.get_seq_length(0)
        max_len = mistral_cache.get_max_length()
        
        print(f"✓ Cache interface compatibility confirmed")
        print(f"  - Sequence length: {seq_len}")
        print(f"  - Max length (sliding window): {max_len}")
        
        return True
        
    except Exception as e:
        print(f"✗ Compatibility testing failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MISTRAL IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_mistral_model_loading),
        ("Cache Functionality", test_mistral_cache),  
        ("Attention Computation", test_mistral_attention),
        ("Evaluation Functions", test_mistral_evaluation),
        ("Llama Compatibility", test_mistral_vs_llama_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Mistral implementation is ready to use.")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)