#!/usr/bin/env python3
"""Analyze TFLite model operators."""

import sys
import tensorflow as tf
from collections import Counter


def analyze_operators(tflite_path):
    """Analyze operators in a TFLite model."""
    print(f"\nAnalyzing: {tflite_path}")
    print("="*70)
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nInput shape:  {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Get all operators
    with open(tflite_path, 'rb') as f:
        model_content = f.read()
    
    # Use TFLite schema to parse
    try:
        import flatbuffers
        from tensorflow.lite.python import schema_py_generated as schema_fb
        
        model = schema_fb.Model.GetRootAsModel(model_content, 0)
        
        # Get subgraph (usually only one)
        subgraph = model.Subgraphs(0)
        
        # Get operator codes
        op_codes = []
        for i in range(model.OperatorCodesLength()):
            op_code = model.OperatorCodes(i)
            builtin_code = op_code.BuiltinCode()
            # Get operator name from builtin code
            op_name = schema_fb.BuiltinOperator().Name(builtin_code)
            op_codes.append(op_name)
        
        # Count operators
        operator_counts = Counter()
        for i in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(i)
            op_code_idx = operator.OpcodeIndex()
            op_name = op_codes[op_code_idx]
            operator_counts[op_name] += 1
        
        # Print operator statistics
        print(f"\nTotal operators: {sum(operator_counts.values())}")
        print(f"Unique operators: {len(operator_counts)}")
        
        print("\nOperator breakdown:")
        for op_name, count in sorted(operator_counts.items(), key=lambda x: -x[1]):
            print(f"  {op_name:30s}: {count:4d}")
        
        # Check for problematic operators
        print("\n" + "="*70)
        problematic_ops = ['GATHER_ND', 'GATHER', 'SCATTER_ND']
        found_problematic = False
        for op in problematic_ops:
            if op in operator_counts:
                print(f"⚠️  Found {op}: {operator_counts[op]} instances")
                found_problematic = True
        
        if not found_problematic:
            print("✅ No problematic operators (GATHER_ND, GATHER, SCATTER_ND) found!")
        
        # Check tensor dimensions
        print("\n" + "="*70)
        print("Tensor dimension analysis:")
        max_dims = 0
        tensors_by_dims = Counter()
        
        for i in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(i)
            shape_len = tensor.ShapeLength()
            tensors_by_dims[shape_len] += 1
            max_dims = max(max_dims, shape_len)
        
        print(f"\nMax tensor dimensions: {max_dims}D")
        print("\nTensor dimension distribution:")
        for dims in sorted(tensors_by_dims.keys()):
            print(f"  {dims}D tensors: {tensors_by_dims[dims]}")
        
        if max_dims <= 4:
            print("\n✅ All tensors are 4D or less (BSTM HW compatible)!")
        else:
            print(f"\n⚠️  Found {max_dims}D tensors (BSTM HW only supports up to 4D)")
        
    except Exception as e:
        print(f"\n❌ Error analyzing operators: {e}")
        print("   (This is expected if flatbuffers is not available)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_tflite_ops.py <tflite_model_path>")
        sys.exit(1)
    
    analyze_operators(sys.argv[1])

