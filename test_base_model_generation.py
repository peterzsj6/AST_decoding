#!/usr/bin/env python3
"""
Simple test script to load the base model and generate code directly.
This helps diagnose if the issue is with the model or the generation process.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model path
model_path = "/data/home/zhangsj/deepseek_qwen1.5_distill"
print(f"Loading model from: {model_path}")

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=None,  # Don't use auto device_map
    trust_remote_code=True,
)
if device == "cuda":
    model = model.to(device)
model.eval()

print("Model loaded successfully!\n")

# Test prompt from HumanEval/0
test_prompt = """from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""

# print("=" * 80)
# print("TEST PROMPT:")
# print("=" * 80)
# print(test_prompt)
# print("=" * 80)
# print("\nGenerating completion...\n")

# Tokenize
inputs = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=False)
if device == "cuda":
    inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"Input token length: {inputs['input_ids'].shape[1]}")
print(f"Max new tokens: 512\n")

# Load generation config from model directory
import json
generation_config_path = os.path.join(model_path, "generation_config.json")
generation_kwargs = {}
if os.path.exists(generation_config_path):
    print(f"\nLoading generation config from: {generation_config_path}")
    with open(generation_config_path, 'r') as f:
        gen_config = json.load(f)
    print(f"Generation config: {gen_config}")
    # Use config values from generation_config.json
    generation_kwargs.update(gen_config)
    # Override max_new_tokens for our test
    generation_kwargs['max_new_tokens'] = 2048
    # Ensure we have required token IDs
    if 'pad_token_id' not in generation_kwargs:
        generation_kwargs['pad_token_id'] = tokenizer.pad_token_id or tokenizer.eos_token_id
    if 'eos_token_id' not in generation_kwargs:
        generation_kwargs['eos_token_id'] = tokenizer.eos_token_id
    print(f"Using generation config: do_sample={generation_kwargs.get('do_sample')}, "
          f"temperature={generation_kwargs.get('temperature')}, top_p={generation_kwargs.get('top_p')}")
else:
    print("\nNo generation_config.json found, using defaults")
    generation_kwargs = {
        'max_new_tokens': 512,
        'do_sample': False,
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'use_cache': True,
    }

# Generate
print(f"\nGeneration kwargs: {generation_kwargs}\n")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        **generation_kwargs
    )

# Decode
full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
generated_text = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:], 
    skip_special_tokens=False
)

# print("=" * 80)
# print("FULL OUTPUT (prompt + completion):")
# print("=" * 80)
print(full_text)
# print("=" * 80)
# print("\n" + "=" * 80)
# print("GENERATED COMPLETION ONLY:")
# print("=" * 80)
# print(generated_text)
# print("=" * 80)

# Check for repetition
# print("\n" + "=" * 80)
# print("REPETITION ANALYSIS:")
# print("=" * 80)
# lines = generated_text.split('\n')
# if len(lines) > 5:
#     last_5_lines = lines[-5:]
#     unique_lines = set(last_5_lines)
#     print(f"Last 5 lines unique count: {len(unique_lines)}/{len(last_5_lines)}")
#     if len(unique_lines) < 3:
#         print("WARNING: High repetition detected in last 5 lines!")
#         for i, line in enumerate(last_5_lines):
#             print(f"  Line {i+1}: {line[:80]}")

# Check for TODO comments
# todo_count = generated_text.count("# TODO")
# print(f"\nNumber of '# TODO' occurrences: {todo_count}")
# if todo_count > 5:
#     print("WARNING: Excessive TODO comments detected!")

print("\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)

