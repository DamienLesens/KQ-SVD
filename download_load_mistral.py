#!/usr/bin/env python3
import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id  = "mistralai/Mistral-7B-v0.3"
    cache_dir = "/home/mila/r/rakhshab/scratch/models/Mistral-7B-v0.3"

    # 1) grab token (or rely on `huggingface-cli login`)
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
    if not token:
        print("⚠️  No HUGGINGFACE_HUB_TOKEN env var—make sure you've run `huggingface-cli login`.", file=sys.stderr)

    # 2) ensure the folder exists
    os.makedirs(cache_dir, exist_ok=True)

    # 3) download & load tokenizer
    print(f"🔠 Downloading & loading tokenizer from {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_fast=True,
        use_auth_token=token
    )

    # 4) download & load model
    print(f"🧠 Downloading & loading model from {model_id} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_auth_token=token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # 5) quick sanity check
    prompt = "Hello, world!"
    print(f"▶️  Generating for prompt: {prompt!r}")
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    output  = model.generate(**inputs, max_new_tokens=20)
    text    = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n--- Model output ---")
    print(text)

if __name__ == "__main__":
    main()
