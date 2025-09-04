import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
from pathlib import Path

def preprocess_image(image_path, target_size=(640, 640)):
    """Load and preprocess image for ONNX model inference"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_shape = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    h, w = original_shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Convert to RGB and normalize
    rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # Convert to CHW format and add batch dimension
    input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    
    return input_tensor, original_shape, scale, (x_offset, y_offset)

def postprocess_results(logits, pred_boxes, original_shape, scale, offset, conf_threshold=0.5):
    """Convert model outputs to bounding boxes in original image coordinates"""
    # Apply softmax to logits to get class probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    
    # Get max class probability and class index for each detection
    max_probs = np.max(probs, axis=-1)
    class_ids = np.argmax(probs, axis=-1)
    
    # Filter detections by confidence threshold
    valid_mask = max_probs > conf_threshold
    
    if not np.any(valid_mask):
        return [], [], []
    
    filtered_boxes = pred_boxes[valid_mask]
    filtered_probs = max_probs[valid_mask]
    filtered_classes = class_ids[valid_mask]
    
    # Convert normalized coordinates to original image coordinates
    x_offset, y_offset = offset
    h_orig, w_orig = original_shape
    
    boxes = []
    for box in filtered_boxes:
        # Box format: [cx, cy, w, h] normalized
        cx, cy, w, h = box
        
        # Convert to pixel coordinates in padded image
        x1 = (cx - w/2) * 640 - x_offset
        y1 = (cy - h/2) * 640 - y_offset  
        x2 = (cx + w/2) * 640 - x_offset
        y2 = (cy + h/2) * 640 - y_offset
        
        # Scale back to original image size
        x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
        
        # Clamp to image bounds
        x1 = max(0, min(w_orig, x1))
        y1 = max(0, min(h_orig, y1))
        x2 = max(0, min(w_orig, x2))
        y2 = max(0, min(h_orig, y2))
        
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    return boxes, filtered_probs, filtered_classes

def draw_detections(image, boxes, scores, class_ids, class_names=None):
    """Draw bounding boxes and labels on image"""
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"Class {class_id}: {score:.2f}"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image

def get_available_providers():
    """Get available ONNX runtime providers"""
    available_providers = ort.get_available_providers()
    gpu_providers = ['CUDAExecutionProvider', 'ROCMExecutionProvider']
    
    gpu_available = any(provider in available_providers for provider in gpu_providers)
    return available_providers, gpu_available

def main():
    parser = argparse.ArgumentParser(description="ONNX Object Detection Inference")
    parser.add_argument("--model", default="model.onnx", help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save output image (optional)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", choices=['auto', 'cpu', 'gpu'], default='auto', 
                       help="Device to use for inference: auto (GPU if available, else CPU), cpu, or gpu")
    
    args = parser.parse_args()
    
    # Determine execution providers
    available_providers, gpu_available = get_available_providers()
    
    if args.device == 'gpu' and not gpu_available:
        print("Warning: GPU requested but no GPU providers available. Falling back to CPU.")
        providers = ['CPUExecutionProvider']
    elif args.device == 'cpu':
        providers = ['CPUExecutionProvider']
    elif args.device == 'auto':
        if gpu_available:
            # Use GPU providers if available, with CPU as fallback
            providers = [p for p in available_providers if p in ['CUDAExecutionProvider', 'ROCMExecutionProvider']] + ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    else:  # args.device == 'gpu'
        providers = [p for p in available_providers if p in ['CUDAExecutionProvider', 'ROCMExecutionProvider']]
    
    print(f"Available providers: {available_providers}")
    print(f"Using providers: {providers}")
    
    # Load ONNX model
    print(f"Loading ONNX model: {args.model}")
    session = ort.InferenceSession(args.model, providers=providers)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    input_tensor, original_shape, scale, offset = preprocess_image(args.image)
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = time.time() - start_time
    logits, pred_boxes = outputs
    
    print(f"Inference time: {inference_time:.4f} seconds ({inference_time*1000:.2f} ms)")
    
    # Record inference time to file
    with open("inference_times.txt", "a") as f:
        f.write(f"{args.model},{inference_time:.4f},{inference_time*1000:.2f}\n")
    
    # Postprocess results
    boxes, scores, class_ids = postprocess_results(
        logits[0], pred_boxes[0], original_shape, scale, offset, args.conf
    )
    
    # Load original image for visualization
    original_image = cv2.imread(args.image)
    
    if len(boxes) > 0:
        print(f"Found {len(boxes)} detections")
        # Draw detections
        result_image = draw_detections(original_image.copy(), boxes, scores, class_ids)
    else:
        print("No detections found")
        result_image = original_image
    
    # Display result
    cv2.imshow("Detection Results", result_image)
    
    # Save output if specified
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Output saved to: {args.output}")
    
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()