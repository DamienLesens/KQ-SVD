from datasets import load_dataset
from transformers import AutoTokenizer
import random
import torch


def get_c4_mistral(c4location,nsamples_train, nsamples_test, seed, seqlen, model):
    """
    This function gets samples from the C4 data set for Mistral models, using the train/test split of the data set
    It passes them through the tokenizer, so it returns tokens
    Parameters:
        nsamples_train: number of samples for training
        nsamples_test: number of samples for testing
        seed: random seed
        seqlen: length of each sample
        model: path to the Mistral model, just used to get the tokenizer
    """
    print("get_c4_mistral")
    traindata = load_dataset(
        c4location, data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        c4location, data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    # Mistral models use fast tokenizers
    if 'mistral' in model.lower():
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    else:
        # Fallback for other model types
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    # Ensure we have a pad token for Mistral
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_train):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    random.seed(0)
    valenc = []
    for _ in range(nsamples_test):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])

    return trainloader, valenc, tokenizer


def get_wikitext2_mistral(nsamples, seed, seqlen, model):
    """
    Get WikiText-2 dataset samples for Mistral models.
    
    Parameters:
        nsamples: number of samples to extract
        seed: random seed
        seqlen: length of each sample
        model: path to the Mistral model
    """
    print("get_wikitext2_mistral")
    
    # Load WikiText-2 dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    # Load Mistral tokenizer
    if 'mistral' in model.lower():
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    random.seed(seed)
    samples = []
    
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]['text']
            if len(text.strip()) == 0:  # Skip empty texts
                continue
            
            enc = tokenizer(text, return_tensors='pt')
            if enc.input_ids.shape[1] > seqlen:
                break
        
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        samples.append(inp)

    return samples, tokenizer


def prepare_mistral_dataset(location, dataset_name, model_name, nsamples_train=100, nsamples_test=20, 
                           seed=42, seqlen=2048):
    """
    Prepare dataset for Mistral model training/evaluation.
    
    Parameters:
        dataset_name: Name of dataset ('c4', 'wikitext2', etc.)
        model_name: Mistral model name or path
        nsamples_train: Number of training samples
        nsamples_test: Number of test samples
        seed: Random seed
        seqlen: Sequence length
        
    Returns:
        Tuple of (train_data, test_data, tokenizer)
    """
    if dataset_name.lower() == 'c4':
        return get_c4_mistral(location, nsamples_train, nsamples_test, seed, seqlen, model_name)
    elif dataset_name.lower() == 'wikitext2':
        train_data = get_wikitext2_mistral(nsamples_train, seed, seqlen, model_name)
        test_data = get_wikitext2_mistral(nsamples_test, seed + 1000, seqlen, model_name)
        return train_data[0], test_data[0], train_data[1]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_mistral_attention_mask(input_ids, sliding_window=4096):
    """
    Create attention mask for Mistral models with sliding window constraint.
    
    Parameters:
        input_ids: Input token IDs
        sliding_window: Size of sliding window
        
    Returns:
        Attention mask tensor
    """
    batch_size, seq_len = input_ids.shape
    
    # Create causal mask
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    
    # Apply sliding window constraint
    for i in range(seq_len):
        start_pos = max(0, i - sliding_window + 1)
        if start_pos > 0:
            mask[i, :start_pos] = float('-inf')
    
    # Expand for batch and attention heads
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    
    return mask


def tokenize_for_mistral(texts, tokenizer, max_length=2048, padding=True, truncation=True):
    """
    Tokenize texts specifically for Mistral models.
    
    Parameters:
        texts: List of text strings
        tokenizer: Mistral tokenizer
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        
    Returns:
        Tokenized inputs
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors='pt'
    )


def get_mistral_model_config():
    """
    Get default configuration parameters for Mistral models.
    
    Returns:
        Dictionary with configuration parameters
    """
    return {
        'sliding_window': 4096,
        'num_attention_heads': 32,
        'num_key_value_heads': 8,  # For GQA
        'hidden_size': 4096,
        'intermediate_size': 14336,
        'num_hidden_layers': 32,
        'max_position_embeddings': 32768,
        'vocab_size': 32000,
        'rope_theta': 10000.0,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
    }


def validate_mistral_inputs(input_ids, attention_mask=None, sliding_window=4096):
    """
    Validate inputs for Mistral model processing.
    
    Parameters:
        input_ids: Input token IDs
        attention_mask: Optional attention mask
        sliding_window: Sliding window size
        
    Returns:
        Boolean indicating if inputs are valid
    """
    if input_ids is None:
        return False
    
    batch_size, seq_len = input_ids.shape
    
    # Check if sequence length is reasonable for sliding window
    if seq_len > sliding_window * 2:
        print(f"Warning: Sequence length {seq_len} is much larger than sliding window {sliding_window}")
    
    # Check if attention mask matches input dimensions
    if attention_mask is not None:
        if attention_mask.shape[-2:] != (seq_len, seq_len):
            return False
    
    return True

