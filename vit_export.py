import torch
import onnx
import onnxruntime as ort
from PIL import Image
import requests
from transformers import ViTImageProcessor, ViTForImageClassification

# 1) モデル＆前処理
model_id = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_id)
model = ViTForImageClassification.from_pretrained(model_id)
model.eval()

# テスト画像
url = "https://images.unsplash.com/photo-1516117172878-fd2c41f4a759"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 2) 前処理（pixel_values: [B,3,224,224]）
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # torch.Size([1, 3, 224, 224])

# SDPA無効化（上と同様の理由）
torch.backends.cuda.matmul.allow_tf32 = False
try:
    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
except Exception:
    pass

# 3) export
torch.onnx.export(
    model,
    (pixel_values,),
    "vit_hf_224.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes={
        "pixel_values": {0: "batch"},  # バッチ可変
        "logits": {0: "batch"},
    },
)

# 4) 検証
onnx.checker.check_model(onnx.load("vit_hf_224.onnx"))
sess = ort.InferenceSession("vit_hf_224.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"pixel_values": pixel_values.numpy()})
print([x.shape for x in out])  # 例: [(1, 1000)]
