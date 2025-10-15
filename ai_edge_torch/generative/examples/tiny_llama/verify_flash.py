#!/usr/bin/env python3
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

"""Verifies the TinyLlama model WITH Flash Attention enabled."""

import logging
import pathlib

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import transformers


_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Path to TinyLlama checkpoint (local path or HuggingFace model ID).",
)
_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "Show me the program to add 2 and 3.",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)


def main(_):
  checkpoint = _CHECKPOINT_PATH.value
  logging.info("Loading the original model from: %s", checkpoint)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(
      checkpoint, trust_remote_code=True
  )

  # Build reauthored model WITH Flash Attention enabled
  logging.info("Building the reauthored model WITH Flash Attention from: %s", checkpoint)
  reauthored_model = tiny_llama.build_model(
      checkpoint,
      use_flash_attention=True  # ENABLE FLASH ATTENTION
  )

  logging.info("Loading the tokenizer from: %s", checkpoint)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
  
  # Ensure tokenizer has pad_token set
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  logging.info("=" * 80)
  logging.info("VERIFYING WITH FLASH ATTENTION ENABLED")
  logging.info("=" * 80)

  verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=verifier.ReauthoredModelWrapper(reauthored_model),
      tokenizer=verifier.TokenizerWrapper(tokenizer),
      generate_prompts=_PROMPTS.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
      atol=1e-04,
  )
  
  logging.info("=" * 80)
  logging.info("FLASH ATTENTION VERIFICATION COMPLETE")
  logging.info("=" * 80)


if __name__ == "__main__":
  app.run(main)

