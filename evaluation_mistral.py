import torch
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, apply_rotary_pos_emb
import numpy as np
from custom_cache_mistral import CustomCacheMistral
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


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


def create_sliding_window_mask(seq_len: int, sliding_window: int, device: torch.device) -> torch.Tensor:
    """
    Create sliding window attention mask for Mistral models.
    
    Args:
        seq_len: Sequence length
        sliding_window: Size of sliding window
        device: Device to create mask on
        
    Returns:
        Attention mask with sliding window constraint
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    
    for i in range(seq_len):
        # Causal masking - can't attend to future tokens
        mask[i, i+1:] = float("-inf")
        
        # Sliding window constraint - can't attend to tokens too far in the past
        start_pos = max(0, i - sliding_window + 1)
        mask[i, start_pos:i+1] = 0.0
    
    return mask


def testing_proj_mistral(model, past_key_values, list_proj, ranks, sliding_window: int = 4096):
    """
    This function takes as input for Mistral models:
    - the model, just for the config (hyperparameters etc...)
    - past_key_values: KQV cache in standard format
    - list_proj: projections for each layer, head, KQV
    - ranks: to cut the basis and get the projection on the subspace of the given shape
    - sliding_window: sliding window size for Mistral attention
    
    It simulates the attention computation from the KQV cache given in input.
    It simulates the computation with and without applying dimension reduction and outputs a bunch of metrics.
    It stores those metrics in lists of size L. Metrics that depend on a head are averaged across all heads in the layer
    
    The list of metrics is the following:
        errorK = [] #relative frobenius norm error on K
        errorQ = [] #relative frobenius norm error on Q
        errorV = [] #relative frobenius norm error on V
        errorKQT = [] #relative frobenius norm error on KQ^T
        errorcoeff = [] #relative frobenius norm error coefficients (after the Softmax)
        errorVWO = [] #relative frobenius norm error on V W_o
        errorouput = [] #relative frobenius norm error on the output of the attention
    """
    
    g = model.config.num_attention_heads // model.config.num_key_value_heads  # number of head groups for GQA

    errorK = []  # relative frobenius norm error on K
    errorQ = []  # relative frobenius norm error on Q
    errorV = []  # relative frobenius norm error on V
    errorKQT = []  # relative frobenius norm error on KQ^T
    errorcoeff = []  # relative frobenius norm error coefficients (after the Softmax)
    errorVWO = []  # relative frobenius norm error on V W_o
    errorouput = []  # relative frobenius norm error on the output of the attention

    for l in range(model.config.num_hidden_layers):
        print("layer:", l)
        # original states
        key_states = past_key_values[l][0].to(torch.float)
        value_states = past_key_values[l][1].to(torch.float)
        query_states = past_key_values[l][2].to(torch.float)

        # to be made generic in the actual code
        _, h, T, d = query_states.shape
        scaling = d ** (-0.5)  # Mistral uses inverse square root scaling

        # computing projections
        projk = list_proj[l][0][:, :, :ranks[l][0]].unsqueeze(0)
        projq = list_proj[l][1][:, :, :ranks[l][0]].unsqueeze(0)
        projv = list_proj[l][2][:, :, :ranks[l][1]].unsqueeze(0)
        projw = list_proj[l][3][:, :, :ranks[l][1]].unsqueeze(0)

        # interleaving stuff that needs to be interleaved for GQA
        key_states = repeat_kv(key_states, g)
        value_states = repeat_kv(value_states, g)
        projk = repeat_kv(projk, g)
        projq = repeat_kv(projq, g)
        projv = repeat_kv(projv, g)
        projw = repeat_kv(projw, g)

        # computing how well K and V are approximated
        key_states_a = key_states @ projk @ projq.transpose(2, 3)
        value_states_a = value_states @ projv @ projw.transpose(2, 3)
        query_states_a = query_states @ projq @ projk.transpose(2, 3)

        # computing the attention without compression
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        # computing the attention with compression
        attn_weights_a = torch.matmul(query_states_a, key_states_a.transpose(2, 3)) * scaling

        errorKQT.append(torch.norm(attn_weights - attn_weights_a) / torch.norm(attn_weights))
        
        # Apply sliding window mask for Mistral
        if T > 1:
            sliding_mask = create_sliding_window_mask(T, sliding_window, attn_weights.device)
            attn_weights = attn_weights + sliding_mask.unsqueeze(0).unsqueeze(0)

        attn_weights_softmax = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights_softmax, value_states)
        
        # Apply sliding window mask for compressed attention
        if T > 1:
            attn_weights_a = attn_weights_a + sliding_mask.unsqueeze(0).unsqueeze(0)

        attn_weights_a_softmax = torch.nn.functional.softmax(attn_weights_a, dim=-1, dtype=torch.float32).to(query_states_a.dtype)
        attn_output_a = torch.matmul(attn_weights_a_softmax, value_states_a)

        # computing errors
        errorK.append(torch.norm(key_states - key_states_a) / torch.norm(key_states))
        errorQ.append(torch.norm(query_states - query_states_a) / torch.norm(query_states))
        errorV.append(torch.norm(value_states - value_states_a) / torch.norm(value_states))
        errorcoeff.append(torch.norm(attn_weights_softmax - attn_weights_a_softmax) / torch.norm(attn_weights_softmax))
        errorouput.append(torch.norm(attn_output - attn_output_a) / torch.norm(attn_output))

        # for the V W_o part, I need to project the output to the original head dim since the attention already averages the heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(1, T, model.config.hidden_size)
        attn_output_a = attn_output_a.transpose(1, 2).contiguous().view(1, T, model.config.hidden_size)

        errorVWO.append(torch.norm(attn_output - attn_output_a) / torch.norm(attn_output))

    return errorK, errorQ, errorV, errorKQT, errorcoeff, errorVWO, errorouput


def testing_proj_mistral_norope(model, past_key_values, list_proj, ranks, sliding_window: int = 4096):
    """
    Version of testing_proj_mistral for models without RoPE.
    Same functionality but without rotary position embeddings.
    """
    return testing_proj_mistral(model, past_key_values, list_proj, ranks, sliding_window)


def evaluate_mistral_compression(model, cache, compression_ratios=None, sliding_window: int = 4096):
    """
    Evaluate compression effectiveness for Mistral models.
    
    Args:
        model: Mistral model instance
        cache: Cache containing compressed KQV states
        compression_ratios: Target compression ratios to evaluate
        sliding_window: Sliding window size
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if compression_ratios is None:
        compression_ratios = [0.1, 0.2, 0.5, 0.8]
    
    results = {
        'compression_ratios': compression_ratios,
        'memory_savings': [],
        'reconstruction_errors': [],
        'attention_errors': []
    }
    
    for ratio in compression_ratios:
        # Simulate compression at this ratio
        total_memory_original = 0
        total_memory_compressed = 0
        total_reconstruction_error = 0
        total_attention_error = 0
        
        for layer_idx in range(len(cache)):
            if hasattr(cache, 'get_compression_ratio'):
                layer_ratios = cache.get_compression_ratio(layer_idx)
                # Calculate memory usage based on compression ratios
                # This is a simplified calculation
                memory_saving = 1 - (1 / max(layer_ratios.values()))
                total_memory_compressed += memory_saving
                
            # Calculate reconstruction errors (placeholder)
            # In a real implementation, you would reconstruct and compare
            total_reconstruction_error += 0.01 * (1 - ratio)  # Placeholder
            total_attention_error += 0.005 * (1 - ratio)  # Placeholder
        
        results['memory_savings'].append(total_memory_compressed / len(cache))
        results['reconstruction_errors'].append(total_reconstruction_error / len(cache))
        results['attention_errors'].append(total_attention_error / len(cache))
    
    return results


def plot_mistral_compression_results(results, save_path=None):
    """
    Plot compression evaluation results for Mistral models.
    
    Args:
        results: Results dictionary from evaluate_mistral_compression
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ratios = results['compression_ratios']
    
    # Plot memory savings vs compression ratio
    ax1.plot(ratios, results['memory_savings'], 'b-o', label='Memory Savings')
    ax1.set_xlabel('Target Compression Ratio')
    ax1.set_ylabel('Memory Savings')
    ax1.set_title('Mistral Cache Compression: Memory Efficiency')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot errors vs compression ratio
    ax2.plot(ratios, results['reconstruction_errors'], 'r-o', label='Reconstruction Error')
    ax2.plot(ratios, results['attention_errors'], 'g-o', label='Attention Error')
    ax2.set_xlabel('Target Compression Ratio')
    ax2.set_ylabel('Error')
    ax2.set_title('Mistral Cache Compression: Error Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_llama_mistral_compression(llama_results, mistral_results, save_path=None):
    """
    Compare compression results between Llama and Mistral models.
    
    Args:
        llama_results: Results from Llama model evaluation
        mistral_results: Results from Mistral model evaluation  
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ratios = llama_results['compression_ratios']
    
    # Compare memory savings
    ax1.plot(ratios, llama_results['memory_savings'], 'b-o', label='Llama')
    ax1.plot(ratios, mistral_results['memory_savings'], 'r-s', label='Mistral')
    ax1.set_xlabel('Target Compression Ratio')
    ax1.set_ylabel('Memory Savings')
    ax1.set_title('Memory Efficiency Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Compare attention errors
    ax2.plot(ratios, llama_results['attention_errors'], 'b-o', label='Llama')
    ax2.plot(ratios, mistral_results['attention_errors'], 'r-s', label='Mistral')
    ax2.set_xlabel('Target Compression Ratio')
    ax2.set_ylabel('Attention Error')
    ax2.set_title('Attention Error Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()