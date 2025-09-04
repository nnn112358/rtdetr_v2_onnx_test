#!/usr/bin/env python3
import os
import onnx
from onnxconverter_common import float16
import argparse

def quantize_model_to_fp16(model_path, quantized_model_path):
    """
    Convert ONNX model from FP32 to FP16
    """
    print(f"Loading model: {model_path}")
    
    # Load the original ONNX model
    model = onnx.load(model_path)
    
    print("Converting model to FP16...")
    
    # Convert model to FP16
    model_fp16 = float16.convert_float_to_float16(model)
    
    # Save the FP16 model
    onnx.save(model_fp16, quantized_model_path)
    
    print(f"FP16 model saved to: {quantized_model_path}")
    
    # Compare model sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
    compression_ratio = original_size / quantized_size
    
    print(f"\nModel Size Comparison:")
    print(f"Original model (FP32): {original_size:.2f} MB")
    print(f"FP16 model: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to FP16")
    parser.add_argument("--input", default="rtdetr_r18vd/rtdetr_r18vd.onnx", help="Input ONNX model path")
    parser.add_argument("--output", default="rtdetr_r18vd/rtdetr_r18vd_fp16.onnx", help="Output FP16 model path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input model not found: {args.input}")
        return
    
    try:
        quantize_model_to_fp16(args.input, args.output)
        print("\n✓ FP16 conversion completed successfully!")
        
        # Verify the quantized model can be loaded
        print("\nVerifying FP16 model...")
        import onnxruntime as ort
        session = ort.InferenceSession(args.output)
        print("✓ FP16 model loads successfully!")
        
        # Print input/output info
        print(f"✓ Model inputs: {[inp.name for inp in session.get_inputs()]}")
        print(f"✓ Model outputs: {[out.name for out in session.get_outputs()]}")
        
    except Exception as e:
        print(f"\n✗ FP16 conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()