#!/usr/bin/env python3
"""Check the input tensor order and shapes in the TFLite model."""

import sys
import tensorflow as tf

def check_model_inputs(model_path):
    """Check input tensor details."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("=" * 80)
    print("TFLite Model Input/Output Details")
    print("=" * 80)
    print(f"Model: {model_path}")
    print()
    
    print(f"Number of inputs: {len(input_details)}")
    for i, detail in enumerate(input_details):
        print(f"\nInput {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Index: {detail['index']}")
        total_elements = 1
        for dim in detail['shape']:
            total_elements *= dim
        print(f"  Total elements: {total_elements}")
        print(f"  Size in bytes: {total_elements * 4} (assuming float32)")
    
    print()
    print(f"Number of outputs: {len(output_details)}")
    for i, detail in enumerate(output_details):
        print(f"\nOutput {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Index: {detail['index']}")
        total_elements = 1
        for dim in detail['shape']:
            total_elements *= dim
        print(f"  Total elements: {total_elements}")
        print(f"  Size in bytes: {total_elements * 4} (assuming float32)")
    
    print()
    print("=" * 80)
    print("Expected Input Order Based on Names:")
    print("=" * 80)
    
    # Try to determine the order based on names
    input_map = {}
    for i, detail in enumerate(input_details):
        name = detail['name'].lower()
        if 'args_0' in name or name.endswith('_0'):
            input_map['Q'] = i
        elif 'args_1' in name or name.endswith('_1'):
            input_map['K'] = i
        elif 'args_2' in name or name.endswith('_2'):
            input_map['V'] = i
        elif 'args_3' in name or name.endswith('_3'):
            input_map['Mask'] = i
    
    if input_map:
        print("Detected mapping:")
        for tensor_name, idx in sorted(input_map.items(), key=lambda x: x[1]):
            detail = input_details[idx]
            print(f"  Input[{idx}] = {tensor_name}: {detail['shape']} ({detail['name']})")
    else:
        print("Could not auto-detect mapping from names")
        print("Manual inspection required:")
        for i, detail in enumerate(input_details):
            print(f"  Input[{i}]: {detail['shape']} - {detail['name']}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_model_inputs.py <model.tflite>")
        sys.exit(1)
    
    check_model_inputs(sys.argv[1])

