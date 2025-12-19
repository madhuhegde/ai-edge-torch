#!/usr/bin/env python3
"""Inspect the amd_llama_attention_decode.tflite model"""

import numpy as np
import tensorflow as tf

model_path = "amd_llama_attention_decode.tflite"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 60)
print("TFLite Model Input/Output Details")
print("=" * 60)

print("\nINPUTS:")
for i, detail in enumerate(input_details):
    print(f"\nInput {i}:")
    print(f"  Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Dtype: {detail['dtype']}")
    print(f"  Size: {np.prod(detail['shape'])} elements")
    print(f"  Bytes: {np.prod(detail['shape']) * 4} bytes (assuming float32)")

print("\n" + "=" * 60)
print("\nOUTPUTS:")
for i, detail in enumerate(output_details):
    print(f"\nOutput {i}:")
    print(f"  Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Dtype: {detail['dtype']}")
    print(f"  Size: {np.prod(detail['shape'])} elements")
    print(f"  Bytes: {np.prod(detail['shape']) * 4} bytes (assuming float32)")

print("\n" + "=" * 60)
print("\nModel Parameters:")
print(f"  Num heads: 2")
print(f"  Head dim: 64")
print(f"  KV length: 512")
print("=" * 60)

# Calculate expected sizes
num_heads = 2
head_dim = 64
kv_len = 512

q_size = 1 * 1 * num_heads * head_dim  # [1, 1, num_heads, head_dim]
k_size = 1 * kv_len * num_heads * head_dim  # [1, kv_len, num_heads, head_dim]
v_size = 1 * kv_len * num_heads * head_dim  # [1, kv_len, num_heads, head_dim]
mask_size = 1 * 1 * 1 * kv_len  # [1, 1, 1, kv_len]

print("\nExpected Sizes (based on num_heads=2, head_dim=64, kv_len=512):")
print(f"  Q: [1, 1, {num_heads}, {head_dim}] = {q_size} floats = {q_size*4} bytes")
print(f"  K: [1, {kv_len}, {num_heads}, {head_dim}] = {k_size} floats = {k_size*4} bytes")
print(f"  V: [1, {kv_len}, {num_heads}, {head_dim}] = {v_size} floats = {v_size*4} bytes")
print(f"  Mask: [1, 1, 1, {kv_len}] = {mask_size} floats = {mask_size*4} bytes")
print("=" * 60)

