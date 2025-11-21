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
# Custom RMS Normalization operator using stablehlo.custom_call.

from ai_edge_torch.odml_torch import lowerings
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch


# Use torch.library.custom_op to define a new custom operator.
@torch.library.custom_op("ai_edge_torch::custom_rms_norm", mutates_args=())
def custom_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
  """Custom RMS Normalization operator.

  RMS Normalization normalizes the input tensor by its root mean square
  and scales it by a learnable weight parameter. This is a custom operator
  that lowers to stablehlo.custom_call, which maps to a custom C++ kernel
  in TFLite.

  Formula:
    output = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2, dim=-1) + epsilon)

  Args:
    x: Input tensor of shape [..., hidden_dim]
    weight: Learnable scale parameter of shape [hidden_dim]
    epsilon: Small constant for numerical stability

  Returns:
    Normalized tensor of the same shape as input.

  Example:
    >>> x = torch.randn(2, 128, 768)
    >>> weight = torch.ones(768)
    >>> output = custom_rms_norm(x, weight, 1e-6)
    >>> output.shape
    torch.Size([2, 128, 768])

  Note:
    This custom op uses stablehlo.custom_call which generates a
    STABLEHLO_CUSTOM_CALL operation in TFLite. Requires modified TensorFlow
    with VHLO custom_call fallback handler (see VHLO_CUSTOM_CALL_FIX.md).
  """
  # Validate inputs
  if x.dim() < 1:
    raise ValueError(f"Input tensor must have at least 1 dimension, got {x.dim()}")
  
  if weight.dim() != 1:
    raise ValueError(f"Weight must be 1-dimensional, got {weight.dim()}")
  
  if x.shape[-1] != weight.shape[0]:
    raise ValueError(
        f"Last dimension of input ({x.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )
  
  if epsilon <= 0:
    raise ValueError(f"Epsilon must be positive, got {epsilon}")

  # Compute RMS normalization (PyTorch implementation for eager execution)
  variance = x.pow(2).mean(dim=-1, keepdim=True)
  x_normalized = x * torch.rsqrt(variance + epsilon)
  return x_normalized * weight


# Use register_fake to add a ``FakeTensor`` kernel for the operator
@custom_rms_norm.register_fake
def _(x, weight, epsilon):
  """Fake implementation for shape inference during tracing.

  This function is called by PyTorch's tracing mechanism to determine
  the output shape without executing the actual computation.

  Args:
    x: Input tensor
    weight: Weight tensor
    epsilon: Epsilon value (unused in shape inference)

  Returns:
    A tensor with the same shape as input for shape inference.
  """
  # Validate inputs
  if x.dim() < 1:
    raise ValueError(f"Input tensor must have at least 1 dimension, got {x.dim()}")
  
  if weight.dim() != 1:
    raise ValueError(f"Weight must be 1-dimensional, got {weight.dim()}")
  
  if x.shape[-1] != weight.shape[0]:
    raise ValueError(
        f"Last dimension of input ({x.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )
  
  # Return tensor with same shape as input
  return x.clone()


@lowerings.lower(torch.ops.ai_edge_torch.custom_rms_norm)
def _custom_rms_norm_lower(
    lctx,
    x: ir.Value,
    weight: ir.Value,
    epsilon: float,
):
  """Lower the custom RMS norm op to StableHLO custom_call.

  This function converts the PyTorch custom operator to a StableHLO
  custom_call operation. The TFLite converter (with VHLO fallback handler)
  will preserve this as a STABLEHLO_CUSTOM_CALL operation that links to
  a custom C++ kernel at runtime.

  Args:
    lctx: Lowering context
    x: Input tensor in MLIR IR
    weight: Weight tensor in MLIR IR (scale parameter)
    epsilon: Epsilon value for numerical stability (stored in backend_config)

  Returns:
    StableHLO custom_call operation.

  Note:
    Requires modified TensorFlow with VHLO custom_call fallback handler.
    See VHLO_CUSTOM_CALL_FIX.md for details.

  The generated StableHLO IR looks like:
    %result = stablehlo.custom_call @ai_edge_torch.rms_norm(%x, %weight) {
        backend_config = "",
        api_version = 1
    } : (tensor<...xf32>, tensor<...xf32>) -> tensor<...xf32>
  """
  # Create custom_call operation with proper attribute types
  # The call_target_name "ai_edge_torch.rms_norm" must match the C++ kernel registration
  result = stablehlo.custom_call(
      [x.type],  # result types (same shape as input)
      [x, weight],  # operands
      call_target_name=ir.StringAttr.get("ai_edge_torch.rms_norm"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),  # Could store epsilon here if needed
      api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1),  # API_VERSION_ORIGINAL
  )
  
  return result
