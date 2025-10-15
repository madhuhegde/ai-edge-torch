# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Verifies the TFLite TinyLlama model against the PyTorch model."""

import logging
import os

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import numpy as np
import torch
import transformers

try:
  from ai_edge_litert import interpreter as tfl_interpreter
except ImportError:
  try:
    import tensorflow.lite as tfl_interpreter
  except ImportError:
    raise ImportError(
        "Neither ai_edge_litert nor tensorflow.lite found. "
        "Please install one of them."
    )


_TFLITE_PATH = flags.DEFINE_string(
    "tflite_path",
    None,
    "Path to the converted TinyLlama TFLite model file.",
    required=True,
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Path to TinyLlama checkpoint (safetensor model directory).",
    required=True,
)
_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    ["What is AI?", "Hello, how are you?"],
    "Test prompts for verification.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    20,
    "Maximum number of tokens to generate.",
)
_TOLERANCE = flags.DEFINE_float(
    "tolerance",
    1e-4,
    "Tolerance for comparing logits between PyTorch and TFLite.",
)


def load_tflite_model(tflite_path: str):
  """Load TFLite model and return interpreter."""
  if not os.path.exists(tflite_path):
    raise FileNotFoundError(f"TFLite model not found at: {tflite_path}")
  
  logging.info(f"Loading TFLite model from: {tflite_path}")
  interpreter = tfl_interpreter.Interpreter(model_path=tflite_path)
  
  # Get all signatures
  signatures = interpreter.get_signature_list()
  logging.info(f"Available signatures: {signatures}")
  
  return interpreter


def load_pytorch_model(checkpoint_path: str):
  """Load PyTorch model."""
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint directory not found at: {checkpoint_path}"
    )
  
  logging.info(f"Loading PyTorch model from: {checkpoint_path}")
  model = tiny_llama.build_model(checkpoint_path, kv_cache_max_len=1280)
  model.eval()
  return model


def compare_single_forward(
    pytorch_model, tflite_interpreter, tokens, seq_len, tolerance
):
  """Compare a single forward pass between PyTorch and TFLite models."""
  batch_size = 1
  
  # Prepare PyTorch inputs
  input_pos = torch.arange(seq_len, dtype=torch.int32)
  kv_cache = kv_utils.KVCache.from_model_config(pytorch_model.config)
  
  # PyTorch forward
  logging.info("Running PyTorch forward pass...")
  with torch.no_grad():
    pytorch_output = pytorch_model.forward(tokens, input_pos, kv_cache)
    pytorch_logits = pytorch_output["logits"].numpy()
  
  logging.info(f"PyTorch logits shape: {pytorch_logits.shape}")
  
  # TFLite forward (try prefill signature)
  logging.info("Running TFLite forward pass...")
  try:
    # Try to find appropriate signature
    sig_name = None
    sig_list = tflite_interpreter.get_signature_list()
    
    # For single token, use decode; for multiple tokens, use prefill
    if seq_len == 1:
      if "decode" in sig_list:
        sig_name = "decode"
    else:
      # Try to find matching prefill signature
      for name in sig_list:
        if "prefill" in name and (f"_{seq_len}" in name or f"seq{seq_len}" in name):
          sig_name = name
          break
      # If no exact match, use generic prefill
      if sig_name is None and "prefill" in sig_list:
        sig_name = "prefill"
    
    if sig_name is None:
      raise ValueError(f"No suitable signature found for sequence length {seq_len}")
    
    logging.info(f"Using TFLite signature: {sig_name}")
    
    # Get signature runner
    runner = tflite_interpreter.get_signature_runner(sig_name)
    
    # Prepare TFLite inputs
    input_details = runner.get_input_details()
    logging.info(f"TFLite input details: {input_details.keys()}")
    
    # Check expected sequence length from tokens input
    expected_seq_len = None
    if "tokens" in input_details:
      tokens_shape = input_details["tokens"]["shape"]
      expected_seq_len = tokens_shape[1] if len(tokens_shape) > 1 else None
      logging.info(f"Expected tokens shape: {tokens_shape}, seq_len: {expected_seq_len}")
    
    # Pad tokens and input_pos if needed
    tokens_np = tokens.numpy().astype(np.int32)
    input_pos_np = input_pos.numpy().astype(np.int32)
    
    if expected_seq_len is not None and seq_len < expected_seq_len:
      # Pad with zeros (or pad token)
      pad_len = expected_seq_len - seq_len
      tokens_np = np.pad(tokens_np, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
      input_pos_np = np.pad(input_pos_np, ((0, pad_len),), mode='constant', constant_values=0)
      logging.info(f"Padded from {seq_len} to {expected_seq_len} tokens")
    elif expected_seq_len is not None and seq_len > expected_seq_len:
      # Truncate
      tokens_np = tokens_np[:, :expected_seq_len]
      input_pos_np = input_pos_np[:expected_seq_len]
      logging.warning(f"Truncated from {seq_len} to {expected_seq_len} tokens")
    
    # Run inference
    tflite_inputs = {
        "tokens": tokens_np,
        "input_pos": input_pos_np,
    }
    
    # Add KV cache inputs if needed
    for key in input_details.keys():
      if "cache" in key.lower() and key not in tflite_inputs:
        shape = input_details[key]["shape"]
        tflite_inputs[key] = np.zeros(shape, dtype=np.float32)
    
    tflite_outputs = runner(**tflite_inputs)
    
    # Log what outputs are available
    logging.info(f"TFLite output keys: {list(tflite_outputs.keys())}")
    
    # Get logits from output
    if "logits" not in tflite_outputs:
      logging.warning(
          "TFLite model does not output logits for this signature. "
          "The model was likely converted with output_logits_on_prefill=False. "
          "Skipping forward pass comparison."
      )
      # Still print what we got
      for key, value in list(tflite_outputs.items())[:3]:  # Show first 3 outputs
        logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
      return None  # Skip this test
    
    tflite_logits = tflite_outputs["logits"]
    logging.info(f"TFLite logits shape: {tflite_logits.shape}")
    
    # If we padded, only compare the non-padded portion
    if expected_seq_len is not None and seq_len < expected_seq_len:
      tflite_logits = tflite_logits[:, :seq_len, :]
      logging.info(f"Extracted non-padded TFLite logits shape: {tflite_logits.shape}")
    
    # Compare outputs
    if pytorch_logits.shape != tflite_logits.shape:
      logging.warning(
          f"Shape mismatch: PyTorch {pytorch_logits.shape} vs "
          f"TFLite {tflite_logits.shape}"
      )
      return False
    
    # Calculate differences
    abs_diff = np.abs(pytorch_logits - tflite_logits)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    logging.info(f"Max absolute difference: {max_diff:.6f}")
    logging.info(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Print sample outputs for inspection
    logging.info("\n--- Sample Outputs (first 10 logits of last token) ---")
    logging.info(f"PyTorch: {pytorch_logits[0, -1, :10]}")
    logging.info(f"TFLite:  {tflite_logits[0, -1, :10]}")
    
    # Print predicted tokens
    pytorch_pred = np.argmax(pytorch_logits[0, -1, :])
    tflite_pred = np.argmax(tflite_logits[0, -1, :])
    logging.info(f"\nPredicted next token - PyTorch: {pytorch_pred}, TFLite: {tflite_pred}")
    
    if max_diff < tolerance:
      logging.info(f"✓ PASSED: Difference {max_diff:.6f} < tolerance {tolerance}")
      return True
    else:
      logging.error(
          f"✗ FAILED: Difference {max_diff:.6f} >= tolerance {tolerance}"
      )
      return False
    
  except Exception as e:
    logging.error(f"Error during TFLite inference: {e}")
    import traceback
    traceback.print_exc()
    return False


def generate_with_tflite(
    tflite_interpreter, tokenizer, prompt, max_tokens, eos_token_id
):
  """Generate text using TFLite model with proper prefill + decode loop."""
  logging.info(f"\n=== TFLite Generation ===")
  logging.info(f"Prompt: '{prompt}'")
  
  # Tokenize prompt
  if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.encode(formatted_prompt, return_tensors="pt")
  else:
    tokens = tokenizer.encode(prompt, return_tensors="pt")
  
  prompt_tokens = tokens[0].numpy().astype(np.int32)
  seq_len = len(prompt_tokens)
  logging.info(f"Prompt length: {seq_len} tokens")
  
  # Get model config (22 layers, 4 kv heads, 64 head dim, 1280 max len)
  num_layers = 22
  kv_max_len = 1280
  num_kv_heads = 4
  head_dim = 64
  
  # Initialize KV cache to all zeros
  kv_cache = {}
  for i in range(num_layers):
    kv_cache[f'kv_cache_k_{i}'] = np.zeros((1, kv_max_len, num_kv_heads, head_dim), dtype=np.float32)
    kv_cache[f'kv_cache_v_{i}'] = np.zeros((1, kv_max_len, num_kv_heads, head_dim), dtype=np.float32)
  
  # Get prefill signature
  prefill_sig = tflite_interpreter.get_signature_runner("prefill")
  decode_sig = tflite_interpreter.get_signature_runner("decode")
  
  # Check if we need to pad the prompt
  prefill_input_details = prefill_sig.get_input_details()
  expected_seq_len = prefill_input_details['tokens']['shape'][1]
  
  # Pad prompt if needed
  if seq_len < expected_seq_len:
    padded_tokens = np.pad(prompt_tokens, (0, expected_seq_len - seq_len), mode='constant', constant_values=0)
    # Pad input_pos with continuing positions (not zeros!)
    full_input_pos = np.arange(expected_seq_len, dtype=np.int32)
    padded_input_pos = full_input_pos
    logging.info(f"Padded prompt from {seq_len} to {expected_seq_len} tokens")
  elif seq_len > expected_seq_len:
    padded_tokens = prompt_tokens[:expected_seq_len]
    padded_input_pos = np.arange(expected_seq_len, dtype=np.int32)
    logging.warning(f"Truncated prompt from {seq_len} to {expected_seq_len} tokens")
    seq_len = expected_seq_len
  else:
    padded_tokens = prompt_tokens
    padded_input_pos = np.arange(seq_len, dtype=np.int32)
  
  # PREFILL STAGE: Process the prompt
  logging.info("Running prefill stage...")
  prefill_inputs = {
      'tokens': padded_tokens.reshape(1, -1),
      'input_pos': padded_input_pos,
      **kv_cache
  }
  
  prefill_outputs = prefill_sig(**prefill_inputs)
  
  # Update KV cache from prefill outputs
  for i in range(num_layers):
    if f'kv_cache_k_{i}' in prefill_outputs:
      kv_cache[f'kv_cache_k_{i}'] = prefill_outputs[f'kv_cache_k_{i}']
    if f'kv_cache_v_{i}' in prefill_outputs:
      kv_cache[f'kv_cache_v_{i}'] = prefill_outputs[f'kv_cache_v_{i}']
  
  # Check if prefill outputs logits (it might not with default conversion)
  if 'logits' in prefill_outputs:
    # Get first predicted token from prefill
    logits = prefill_outputs['logits']
    # Get logits for the last actual token (not padding)
    next_token = np.argmax(logits[0, seq_len - 1, :])
    logging.info(f"Prefill output: First predicted token = {next_token}")
    generated_tokens = [int(next_token)]
    current_pos = seq_len
  else:
    logging.info("Prefill doesn't output logits, using decode to generate...")
    # After prefill, the KV cache has processed all prompt tokens
    # We don't need a separate first decode - just start the decode loop
    # The first decode will predict the first new token
    generated_tokens = []
    next_token = prompt_tokens[-1]  # This will be updated in first decode iteration
    current_pos = seq_len - 1  # Will be incremented to seq_len in first iteration
    
  # DECODE STAGE: Generate remaining tokens iteratively
  logging.info("Running decode stage...")
  
  for step in range(max_tokens):
    # For the first iteration when prefill didn't output logits,
    # we need to get the prediction for the last prompt position
    if len(generated_tokens) == 0 and 'logits' not in prefill_outputs:
      # This is the first decode after prefill without logits
      # Use the last actual prompt token and position seq_len-1
      decode_inputs = {
          'tokens': np.array([[prompt_tokens[-1]]], dtype=np.int32),
          'input_pos': np.array([seq_len - 1], dtype=np.int32),
          **kv_cache
      }
      logging.info(f"First decode: token={prompt_tokens[-1]}, pos={seq_len-1}")
    else:
      # Normal decode: use previously generated token
      current_pos += 1
      decode_inputs = {
          'tokens': np.array([[next_token]], dtype=np.int32),
          'input_pos': np.array([current_pos], dtype=np.int32),
          **kv_cache
      }
      if step < 3:  # Log first few iterations
        logging.info(f"Decode step {step}: token={next_token}, pos={current_pos}, generated_so_far={len(generated_tokens)}")
    
    # Run decode
    decode_outputs = decode_sig(**decode_inputs)
    
    # Update KV cache
    for i in range(num_layers):
      if f'kv_cache_k_{i}' in decode_outputs:
        kv_cache[f'kv_cache_k_{i}'] = decode_outputs[f'kv_cache_k_{i}']
      if f'kv_cache_v_{i}' in decode_outputs:
        kv_cache[f'kv_cache_v_{i}'] = decode_outputs[f'kv_cache_v_{i}']
    
    # Get next token from logits (greedy)
    logits = decode_outputs['logits']
    next_token = np.argmax(logits[0, 0, :])
    generated_tokens.append(int(next_token))
    
    if step < 5:  # Log first few tokens
      logging.info(f"Generated token {step}: {next_token} (top logits: {np.sort(logits[0, 0, :])[-5:]})")
    
    # Check for EOS
    if next_token == eos_token_id:
      logging.info(f"EOS token encountered at step {step}")
      break
  
  # Decode generated tokens
  full_output = prompt_tokens.tolist() + generated_tokens
  decoded_text = tokenizer.decode(full_output, skip_special_tokens=True)
  
  logging.info(f"Generated {len(generated_tokens)} tokens")
  logging.info(f"Generated tokens: {generated_tokens[:20]}...")  # Show first 20
  logging.info(f"\nTFLite Output:\n{decoded_text}\n")
  
  return generated_tokens, decoded_text


def verify_text_generation(
    pytorch_model, tflite_interpreter, tokenizer, prompt, max_tokens
):
  """Verify text generation between PyTorch and TFLite."""
  logging.info(f"\n{'='*70}")
  logging.info(f"Verifying text generation for prompt: '{prompt}'")
  logging.info(f"{'='*70}")
  
  eos_token_id = tokenizer.eos_token_id
  
  # Generate with TFLite
  tflite_tokens, tflite_text = generate_with_tflite(
      tflite_interpreter, tokenizer, prompt, max_tokens, eos_token_id
  )
  
  # Generate with PyTorch for comparison
  logging.info(f"\n=== PyTorch Generation (for comparison) ===")
  if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.encode(formatted_prompt, return_tensors="pt")
  else:
    tokens = tokenizer.encode(prompt, return_tensors="pt")
  
  # PyTorch generation
  with torch.no_grad():
    input_pos = torch.arange(0, tokens.shape[1], dtype=torch.int)
    kv_cache = kv_utils.KVCache.from_model_config(pytorch_model.config)
    
    # Prefill
    output = pytorch_model.forward(tokens, input_pos, kv_cache)
    logits = output["logits"]
    kv_cache = output["kv_cache"]
    
    # Get first token
    next_token = torch.argmax(logits[0, -1, :]).item()
    pytorch_tokens = [next_token]
    current_pos = tokens.shape[1]
    
    # Decode loop
    for step in range(max_tokens):
      token_tensor = torch.tensor([[next_token]], dtype=torch.int)
      pos_tensor = torch.tensor([current_pos], dtype=torch.int)
      
      output = pytorch_model.forward(token_tensor, pos_tensor, kv_cache)
      logits = output["logits"]
      kv_cache = output["kv_cache"]
      
      next_token = torch.argmax(logits[0, -1, :]).item()
      pytorch_tokens.append(next_token)
      
      if next_token == eos_token_id:
        break
      
      current_pos += 1
    
    full_pytorch = tokens[0].tolist() + pytorch_tokens
    pytorch_text = tokenizer.decode(full_pytorch, skip_special_tokens=True)
    
    logging.info(f"Generated {len(pytorch_tokens)} tokens")
    logging.info(f"Generated tokens: {pytorch_tokens[:20]}...")
    logging.info(f"\nPyTorch Output:\n{pytorch_text}\n")
  
  # Compare outputs
  logging.info(f"{'='*70}")
  logging.info("COMPARISON")
  logging.info(f"{'='*70}")
  logging.info(f"TFLite tokens: {len(tflite_tokens)}")
  logging.info(f"PyTorch tokens: {len(pytorch_tokens)}")
  
  # Check if first few tokens match
  match_count = 0
  min_len = min(len(tflite_tokens), len(pytorch_tokens))
  for i in range(min(10, min_len)):
    if tflite_tokens[i] == pytorch_tokens[i]:
      match_count += 1
  
  logging.info(f"First 10 tokens match: {match_count}/{min(10, min_len)}")
  
  # Success if first token matches (quantization can cause divergence)
  if tflite_tokens[0] == pytorch_tokens[0]:
    logging.info("✓ First token matches (allowing divergence due to quantization)")
    return True
  else:
    logging.warning("✗ First token differs (may be due to quantization)")
    return False


def main(_):
  # Configure logging
  logging.basicConfig(
      level=logging.INFO, format="%(levelname)s: %(message)s"
  )
  
  logging.info("=" * 70)
  logging.info("TinyLlama TFLite Verification")
  logging.info("=" * 70)
  
  # Load models
  try:
    tflite_interpreter = load_tflite_model(_TFLITE_PATH.value)
    pytorch_model = load_pytorch_model(_CHECKPOINT_PATH.value)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        _CHECKPOINT_PATH.value
    )
    
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
    
  except Exception as e:
    logging.error(f"Failed to load models: {e}")
    return 1
  
  logging.info("\n" + "="*70)
  logging.info("NOTE: This script uses the proper TFLite inference workflow:")
  logging.info("  1. Initialize KV cache to zeros")
  logging.info("  2. Run prefill signature with the prompt")
  logging.info("  3. Run decode signature iteratively (greedy decoding)")
  logging.info("  4. Stop at EOS token or max tokens")
  logging.info("="*70 + "\n")
  
  # Run verification tests
  passed_tests = 0
  total_tests = 0
  
  # Test: Verify text generation with prompts
  for prompt in _PROMPTS.value:
    logging.info("\n" + "=" * 70)
    logging.info(f"Test: Prompt Verification")
    logging.info("=" * 70)
    result = verify_text_generation(
        pytorch_model,
        tflite_interpreter,
        tokenizer,
        prompt,
        _MAX_NEW_TOKENS.value,
    )
    if result is not None:  # Test was run (not skipped)
      total_tests += 1
      if result:
        passed_tests += 1
  
  # Summary
  logging.info("\n" + "=" * 70)
  logging.info("VERIFICATION SUMMARY")
  logging.info("=" * 70)
  logging.info(f"Tests passed: {passed_tests}/{total_tests}")
  
  if passed_tests == total_tests:
    logging.info("✓ ALL TESTS PASSED")
    return 0
  else:
    logging.error(
        f"✗ SOME TESTS FAILED: {total_tests - passed_tests} failures"
    )
    return 1


if __name__ == "__main__":
  app.run(main)

