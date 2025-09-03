import torch
import requests
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# モデル読み込み
print("Loading model...")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model.eval()

# サンプル画像で入力サイズ確認
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")

# ONNX エクスポート
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    inputs["pixel_values"],
    "rtdetr_v2.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
#    input_names=["pixel_values"],
#    output_names=["logits", "pred_boxes"],
#    dynamic_axes={
#        "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
#        "logits": {0: "batch_size"},
#        "pred_boxes": {0: "batch_size"}
#    },
    dynamo=True
)

print("✅ ONNX export completed: rtdetr_v2.onnx")



