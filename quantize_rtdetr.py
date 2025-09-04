#!/usr/bin/env python3
import os
import sys
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import cv2
from pathlib import Path

class RTDETRCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, target_size=(640, 640), max_samples=100):
        self.image_folder = calibration_image_folder
        self.target_size = target_size
        self.max_samples = max_samples
        self.data_list = []
        self.current_index = 0
        self._prepare_data()

    def _prepare_data(self):
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        if os.path.isfile(self.image_folder):
            # Single image file
            image_files = [self.image_folder]
        else:
            # Directory of images
            for ext in image_extensions:
                image_files.extend(Path(self.image_folder).glob(f'*{ext}'))
                image_files.extend(Path(self.image_folder).glob(f'*{ext.upper()}'))
        
        # Limit number of calibration samples
        if len(image_files) > self.max_samples:
            image_files = image_files[:self.max_samples]
        
        print(f"Found {len(image_files)} calibration images")
        
        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
                
            # Preprocess similar to inference
            original_shape = image.shape[:2]
            h, w = original_shape
            scale = min(self.target_size[0] / h, self.target_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create padded image
            padded = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            y_offset = (self.target_size[0] - new_h) // 2
            x_offset = (self.target_size[1] - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Convert to RGB and normalize
            rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # Convert to CHW format and add batch dimension
            input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
            self.data_list.append({"pixel_values": input_tensor})

    def get_next(self):
        if self.current_index >= len(self.data_list):
            return None
        
        data = self.data_list[self.current_index]
        self.current_index += 1
        return data

def quantize_rtdetr_model(model_path, quantized_model_path, calibration_data_path):
    """
    Quantize RT-DETR ONNX model to INT8 using static quantization
    """
    print(f"Loading model: {model_path}")
    
    # Load the ONNX model to get input name
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    print(f"Model input name: {input_name}")
    
    # Create calibration data reader
    print(f"Creating calibration data reader with data from: {calibration_data_path}")
    calibration_data_reader = RTDETRCalibrationDataReader(calibration_data_path)
    
    # Quantization configuration
    print("Starting static quantization...")
    quantize_static(
        model_input=model_path,
        model_output=quantized_model_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format (better for GPU)
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,  # Per-channel quantization for better accuracy
        reduce_range=False,  # Use full INT8 range
        use_external_data_format=False
    )
    
    print(f"Quantized model saved to: {quantized_model_path}")
    
    # Compare model sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
    compression_ratio = original_size / quantized_size
    
    print(f"\nModel Size Comparison:")
    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")

def quantize_rtdetr_int16(model_path, quantized_model_path, calibration_data_path):
    """
    Quantize RT-DETR ONNX model to INT16 using static quantization
    """
    print(f"Loading model: {model_path}")
    
    # Load the ONNX model to get input name
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    print(f"Model input name: {input_name}")
    
    # Create calibration data reader
    print(f"Creating calibration data reader with data from: {calibration_data_path}")
    calibration_data_reader = RTDETRCalibrationDataReader(calibration_data_path)
    
    # Quantization configuration for INT16
    print("Starting INT16 static quantization...")
    quantize_static(
        model_input=model_path,
        model_output=quantized_model_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format
        activation_type=QuantType.QInt16,  # INT16 for activations
        weight_type=QuantType.QInt16,     # INT16 for weights
        per_channel=True,  # Per-channel quantization for better accuracy
        reduce_range=False,  # Use full INT16 range
        use_external_data_format=False
    )
    
    print(f"INT16 quantized model saved to: {quantized_model_path}")
    
    # Compare model sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
    compression_ratio = original_size / quantized_size
    
    print(f"\nModel Size Comparison:")
    print(f"Original model: {original_size:.2f} MB")
    print(f"INT16 quantized model: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quantize RT-DETR model")
    parser.add_argument("--mode", choices=['int8', 'int16'], default='int8', help="Quantization mode")
    parser.add_argument("--model", default="rtdetr_r18vd/rtdetr_r18vd.onnx", help="Input model path")
    args = parser.parse_args()
    
    # Model paths
    original_model = args.model
    
    # Generate output path based on input model
    model_dir = os.path.dirname(original_model)
    model_name = os.path.basename(original_model).replace('.onnx', '')
    
    if args.mode == 'int8':
        quantized_model = os.path.join(model_dir, f"{model_name}_int8.onnx")
    else:
        quantized_model = os.path.join(model_dir, f"{model_name}_int16.onnx")
    
    # Calibration data (using test image)
    calibration_data = "test.jpg"
    
    if not os.path.exists(original_model):
        print(f"Error: Original model not found: {original_model}")
        return
    
    if not os.path.exists(calibration_data):
        print(f"Error: Calibration data not found: {calibration_data}")
        print("Please ensure test.jpg exists in the current directory")
        return
    
    try:
        if args.mode == 'int8':
            quantize_rtdetr_model(original_model, quantized_model, calibration_data)
            print("\n✓ INT8 Quantization completed successfully!")
        else:
            quantize_rtdetr_int16(original_model, quantized_model, calibration_data)
            print("\n✓ INT16 Quantization completed successfully!")
        
        # Verify the quantized model can be loaded
        print("\nVerifying quantized model...")
        import onnxruntime as ort
        session = ort.InferenceSession(quantized_model)
        print("✓ Quantized model loads successfully!")
        
    except Exception as e:
        print(f"\n✗ Quantization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()