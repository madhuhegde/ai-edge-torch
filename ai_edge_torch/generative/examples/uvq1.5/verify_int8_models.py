#!/usr/bin/env python3
"""
Verify INT8 quantized TFLite models and compare with FLOAT32 versions.

This script:
1. Loads both INT8 and FLOAT32 TFLite models
2. Inspects tensor types and quantization parameters
3. Runs inference with the same input on both versions
4. Compares output differences
"""

import os
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow is required. Install with: pip install tensorflow")
    exit(1)


def analyze_model(model_path):
    """Analyze a TFLite model and print detailed information."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {os.path.basename(model_path)}")
    print(f"{'='*70}")
    
    # Get file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input details
    input_details = interpreter.get_input_details()
    print(f"\nInput tensors: {len(input_details)}")
    for i, inp in enumerate(input_details):
        print(f"  [{i}] {inp['name']}")
        print(f"      Shape: {inp['shape']}")
        print(f"      Dtype: {inp['dtype']}")
        if 'quantization_parameters' in inp:
            qparams = inp['quantization_parameters']
            if qparams['scales'].size > 0:
                print(f"      Quantized: scales={qparams['scales']}, zero_points={qparams['zero_points']}")
    
    # Get output details
    output_details = interpreter.get_output_details()
    print(f"\nOutput tensors: {len(output_details)}")
    for i, out in enumerate(output_details):
        print(f"  [{i}] {out['name']}")
        print(f"      Shape: {out['shape']}")
        print(f"      Dtype: {out['dtype']}")
        if 'quantization_parameters' in out:
            qparams = out['quantization_parameters']
            if qparams['scales'].size > 0:
                print(f"      Quantized: scales={qparams['scales']}, zero_points={qparams['zero_points']}")
    
    # Get all tensor details
    tensor_details = interpreter.get_tensor_details()
    print(f"\nTotal tensors: {len(tensor_details)}")
    
    # Count tensor types
    dtype_counts = {}
    quantized_count = 0
    for tensor in tensor_details:
        dtype = str(tensor['dtype'])
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        if 'quantization_parameters' in tensor:
            qparams = tensor['quantization_parameters']
            if qparams['scales'].size > 0:
                quantized_count += 1
    
    print(f"\nTensor types:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count}")
    print(f"\nQuantized tensors: {quantized_count}/{len(tensor_details)}")
    
    return interpreter, input_details, output_details


def compare_models(float32_path, int8_path, model_name):
    """Compare FLOAT32 and INT8 versions of a model."""
    print(f"\n{'='*70}")
    print(f"Comparing {model_name}: FLOAT32 vs INT8")
    print(f"{'='*70}")
    
    # Analyze both models
    float32_interp, float32_inputs, float32_outputs = analyze_model(float32_path)
    int8_interp, int8_inputs, int8_outputs = analyze_model(int8_path)
    
    # Create sample input
    print(f"\n{'='*70}")
    print("Running inference comparison")
    print(f"{'='*70}")
    
    if model_name == "ContentNet":
        # (1, 3, 256, 256)
        sample_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    elif model_name == "DistortionNet":
        # (9, 3, 360, 640)
        sample_input = np.random.randn(9, 3, 360, 640).astype(np.float32)
    elif model_name == "AggregationNet":
        # Two inputs: content (1, 8, 8, 128) and distortion (1, 24, 24, 128)
        content_input = np.random.randn(1, 8, 8, 128).astype(np.float32)
        distortion_input = np.random.randn(1, 24, 24, 128).astype(np.float32)
    
    # Run FLOAT32 inference
    if model_name == "AggregationNet":
        float32_interp.set_tensor(float32_inputs[0]['index'], content_input)
        float32_interp.set_tensor(float32_inputs[1]['index'], distortion_input)
    else:
        float32_interp.set_tensor(float32_inputs[0]['index'], sample_input)
    
    float32_interp.invoke()
    float32_output = float32_interp.get_tensor(float32_outputs[0]['index'])
    
    # Run INT8 inference
    if model_name == "AggregationNet":
        int8_interp.set_tensor(int8_inputs[0]['index'], content_input)
        int8_interp.set_tensor(int8_inputs[1]['index'], distortion_input)
    else:
        int8_interp.set_tensor(int8_inputs[0]['index'], sample_input)
    
    int8_interp.invoke()
    int8_output = int8_interp.get_tensor(int8_outputs[0]['index'])
    
    # Compare outputs
    print(f"\nFLOAT32 output shape: {float32_output.shape}")
    print(f"INT8 output shape: {int8_output.shape}")
    print(f"\nFLOAT32 output range: [{float32_output.min():.6f}, {float32_output.max():.6f}]")
    print(f"INT8 output range: [{int8_output.min():.6f}, {int8_output.max():.6f}]")
    
    # Calculate differences
    abs_diff = np.abs(float32_output - int8_output)
    rel_diff = abs_diff / (np.abs(float32_output) + 1e-8)
    
    print(f"\nAbsolute difference:")
    print(f"  Mean: {abs_diff.mean():.6f}")
    print(f"  Max: {abs_diff.max():.6f}")
    print(f"  Std: {abs_diff.std():.6f}")
    
    print(f"\nRelative difference (%):")
    print(f"  Mean: {rel_diff.mean() * 100:.4f}%")
    print(f"  Max: {rel_diff.max() * 100:.4f}%")
    print(f"  Std: {rel_diff.std() * 100:.4f}%")
    
    # Calculate size reduction
    float32_size = os.path.getsize(float32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    reduction = (1 - int8_size / float32_size) * 100
    
    print(f"\nModel size:")
    print(f"  FLOAT32: {float32_size:.2f} MB")
    print(f"  INT8: {int8_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")


def main():
    # Model paths
    base_dir = os.path.expanduser("~/work/UVQ/uvq/models/tflite_models/uvq1.5")
    
    models = [
        ("ContentNet", "content_net.tflite", "content_net_int8.tflite"),
        ("DistortionNet", "distortion_net.tflite", "distortion_net_int8.tflite"),
        ("AggregationNet", "aggregation_net.tflite", "aggregation_net_int8.tflite"),
    ]
    
    print("="*70)
    print("UVQ 1.5 TFLite Model Verification: FLOAT32 vs INT8")
    print("="*70)
    
    for model_name, float32_file, int8_file in models:
        float32_path = os.path.join(base_dir, float32_file)
        int8_path = os.path.join(base_dir, int8_file)
        
        if not os.path.exists(float32_path):
            print(f"\n✗ FLOAT32 model not found: {float32_path}")
            continue
        if not os.path.exists(int8_path):
            print(f"\n✗ INT8 model not found: {int8_path}")
            continue
        
        try:
            compare_models(float32_path, int8_path, model_name)
        except Exception as e:
            print(f"\n✗ Error comparing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Verification complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

