#!/usr/bin/env python3
import os
import subprocess
import time
from pathlib import Path
import csv

def run_inference_on_all_models():
    """Run GPU inference on all ONNX models and record timing results"""
    
    # Find all ONNX models
    onnx_files = [
        "rtdetr_v2_r18vd/rtdetr_v2_r18vd.onnx",
        "rtdetr_v2_r34vd/rtdetr_v2_r34vd.onnx", 
        "rtdetr_v2_r50vd/rtdetr_v2_r50vd.onnx",
        "rtdetr_v2_r101vd/rtdetr_v2_r101vd.onnx",
        "rtdetr_r18vd/rtdetr_r18vd.onnx",
        "rtdetr_r34vd/rtdetr_r34vd.onnx",
        "rtdetr_r50vd/rtdetr_r50vd.onnx",
        "rtdetr_r101vd/rtdetr_r101vd.onnx"
    ]
    
    # Test image
    test_image = "test.jpg"
    
    if not os.path.exists(test_image):
        print(f"Error: Test image {test_image} not found!")
        return
    
    # Clear previous results
    if os.path.exists("inference_times.txt"):
        os.remove("inference_times.txt")
    
    results = []
    
    print("Starting GPU inference on all ONNX models...")
    print("=" * 60)
    
    for model_path in onnx_files:
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found, skipping...")
            continue
            
        print(f"\nRunning inference on: {model_path}")
        
        # Run inference with GPU
        cmd = [
            "python", "rtdetr_onnx_infer.py",
            "--model", model_path,
            "--image", test_image,
            "--device", "gpu",
            "--conf", "0.5"
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            total_time = time.time() - start_time
            
            if result.returncode == 0:
                # Extract inference time from output
                output_lines = result.stdout.split('\n')
                inference_time = None
                
                for line in output_lines:
                    if "Inference time:" in line:
                        # Parse: "Inference time: 0.1234 seconds (123.45 ms)"
                        parts = line.split()
                        if len(parts) >= 3:
                            inference_time = float(parts[2])
                        break
                
                model_name = os.path.basename(model_path).replace('.onnx', '')
                results.append({
                    'model': model_name,
                    'inference_time_seconds': inference_time,
                    'inference_time_ms': inference_time * 1000 if inference_time else None,
                    'total_time_seconds': total_time,
                    'status': 'success'
                })
                
                print(f"  ✓ Success - Inference: {inference_time:.4f}s ({inference_time*1000:.2f}ms)")
                print(f"  ✓ Total time: {total_time:.2f}s")
                
            else:
                print(f"  ✗ Error running inference: {result.stderr}")
                results.append({
                    'model': os.path.basename(model_path).replace('.onnx', ''),
                    'inference_time_seconds': None,
                    'inference_time_ms': None,
                    'total_time_seconds': total_time,
                    'status': 'error',
                    'error': result.stderr.strip()
                })
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout after 60 seconds")
            results.append({
                'model': os.path.basename(model_path).replace('.onnx', ''),
                'inference_time_seconds': None,
                'inference_time_ms': None,
                'total_time_seconds': None,
                'status': 'timeout'
            })
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append({
                'model': os.path.basename(model_path).replace('.onnx', ''),
                'inference_time_seconds': None,
                'inference_time_ms': None,
                'total_time_seconds': None,
                'status': 'exception',
                'error': str(e)
            })
    
    # Save detailed results to CSV
    print(f"\n{'='*60}")
    print("SUMMARY - GPU Inference Results")
    print(f"{'='*60}")
    
    with open('gpu_inference_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['model', 'inference_time_seconds', 'inference_time_ms', 'total_time_seconds', 'status', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        successful_results = []
        
        for result in results:
            writer.writerow(result)
            
            if result['status'] == 'success':
                successful_results.append(result)
                print(f"{result['model']:<20} | {result['inference_time_ms']:.2f} ms | Success")
            else:
                print(f"{result['model']:<20} | N/A        | {result['status'].title()}")
    
    if successful_results:
        print(f"\n{'='*60}")
        print("Performance Analysis")
        print(f"{'='*60}")
        
        # Sort by inference time
        successful_results.sort(key=lambda x: x['inference_time_ms'])
        
        print("Models ranked by inference speed (fastest first):")
        for i, result in enumerate(successful_results, 1):
            print(f"{i}. {result['model']:<20} - {result['inference_time_ms']:.2f} ms")
        
        # Calculate statistics
        inference_times = [r['inference_time_ms'] for r in successful_results]
        avg_time = sum(inference_times) / len(inference_times)
        fastest = min(inference_times)
        slowest = max(inference_times)
        
        print(f"\nStatistics:")
        print(f"  Fastest: {fastest:.2f} ms")
        print(f"  Slowest: {slowest:.2f} ms")
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Speedup (slowest/fastest): {slowest/fastest:.2f}x")
    
    print(f"\nDetailed results saved to: gpu_inference_results.csv")
    print(f"Raw inference times saved to: inference_times.txt")

if __name__ == "__main__":
    run_inference_on_all_models()