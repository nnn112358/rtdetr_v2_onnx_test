# RT-DETR ONNX 推論ツールキット

リアルタイム DETR (Detection Transformer) の ONNX 推論ツールキットです。物体検出タスクに対応しています。

## 機能

- 複数のRT-DETRモデルバリアント対応 (R18, R34, R50, R101)
- RT-DETR v2モデル対応
- GPU/CPU選択可能な ONNX Runtime 推論
- 設定可能な信頼度閾値
- 画像前処理・後処理
- バッチ推論対応
- INT8/INT16モデル量子化対応

## 必要環境

- Python 3.8+
- numpy
- opencv-python
- onnxruntime-gpu (GPU使用時)
- onnxruntime-tools (量子化時)

## インストール

```bash
uv add numpy opencv-python onnxruntime-gpu onnxruntime-tools
```

## モデルの入手方法

```bash
# 必要なパッケージをインストール
uv add optimum[onnxruntime]

# 全RT-DETRモデルを一括でONNX変換
chmod +x onnx_export_rtdetr.sh
./onnx_export_rtdetr.sh
```

## 使用方法

### 基本的な推論

```bash
# GPU推論 (デフォルト)
uv run python rtdetr_onnx_infer.py --model rtdetr_r18vd/rtdetr_r18vd.onnx --image test.jpg --output result.jpg

# CPU推論
uv run python rtdetr_onnx_infer.py --model rtdetr_r18vd/rtdetr_r18vd.onnx --image test.jpg --device cpu

# 全モデルでGPU推論実行
uv run python batch_inference.py
```

### 利用可能なモデル

| モデル | GPU推論時間 | CPU推論時間 | 特徴 |
|--------|-------------|-------------|------|
| `rtdetr_r18vd/rtdetr_r18vd.onnx` | 241.3ms | - | 最高速 |
| `rtdetr_r34vd/rtdetr_r34vd.onnx` | 259.7ms | - | 高速 |
| `rtdetr_r50vd/rtdetr_r50vd.onnx` | 258.9ms | - | バランス |
| `rtdetr_r101vd/rtdetr_r101vd.onnx` | 309.1ms | - | 高精度 |
| `rtdetr_v2_r18vd/rtdetr_v2_r18vd.onnx` | 264.3ms | 189.2ms | V2最高速 |
| `rtdetr_v2_r34vd/rtdetr_v2_r34vd.onnx` | 261.6ms | - | V2高速 |
| `rtdetr_v2_r50vd/rtdetr_v2_r50vd.onnx` | 271.6ms | - | V2バランス |
| `rtdetr_v2_r101vd/rtdetr_v2_r101vd.onnx` | 305.5ms | - | V2高精度 |

### パラメータ

- `--model`: ONNXモデルファイルのパス
- `--image`: 入力画像のパス
- `--output`: 出力画像のパス (オプション)
- `--conf`: 信頼度閾値 (デフォルト: 0.5)
- `--device`: 実行デバイス (auto/cpu/gpu, デフォルト: auto)

## モデル量子化

### INT8/INT16量子化の実行

```bash
# INT8量子化 (最大圧縮、高速、精度低下あり)
uv run python quantize_rtdetr.py --model rtdetr_r18vd/rtdetr_r18vd.onnx --mode int8

# INT16量子化 (中程度圧縮、低速、精度維持良好)
uv run python quantize_rtdetr.py --model rtdetr_r18vd/rtdetr_r18vd.onnx --mode int16
```

### 量子化モデル性能比較 (rtdetr_r18vd)

| 量子化タイプ | モデルサイズ | 圧縮率 | GPU推論時間 | CPU推論時間 | 特徴 |
|-------------|-------------|--------|-------------|-------------|------|
| **FP32 (元)** | 77.35 MB | 1.00x | 241.3ms | - | 基準 |
| **INT8** | 20.79 MB | 3.72x | 260.5ms | 231.6ms | 最大圧縮 |
| **INT16** | 40.03 MB | 1.93x | 832.6ms | - | 中程度圧縮 |

### 量子化モデル性能比較 (rtdetr_v2_r18vd)

| 量子化タイプ | モデルサイズ | 圧縮率 | GPU推論時間 | CPU推論時間 | 特徴 |
|-------------|-------------|--------|-------------|-------------|------|
| **FP32 (元)** | 77.30 MB | 1.00x | 264.3ms | 189.2ms | 基準 |
| **INT8** | 20.72 MB | 3.73x | 284.8ms | 231.6ms | 最大圧縮 |
| **INT16** | 39.96 MB | 1.93x | 849.9ms | - | 中程度圧縮 |

### 性能記録

推論時間は `inference_times.txt` に以下の形式で記録されます:
```
model_path,inference_time_seconds,inference_time_ms
```

## 使用例

### Python コード例

```python
import cv2
import numpy as np
import onnxruntime as ort

# GPU対応のモデル読み込み
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("rtdetr_r18vd/rtdetr_r18vd.onnx", providers=providers)

# 画像処理と推論実行
# 完全な実装は rtdetr_onnx_infer.py を参照
```

### バッチ処理例

```bash
# 全ONNXモデルでGPU推論を実行し性能を測定
uv run python batch_inference.py
```

## 重要な注意事項

- **GPU推論**: CUDAまたはROCMが利用可能な場合に自動で使用されます
- **INT16量子化**: GPU上では多数のMemcpyノード（1900+）により性能が大幅に低下します
- **CPU推論**: v2_r18vdモデルは CPU で FP32 が最も高速です (189.2ms)
- **量子化**: INT8は最大圧縮率ですが検出精度が低下する可能性があります

## ファイル構成

- `rtdetr_onnx_infer.py` - メイン推論スクリプト
- `quantize_rtdetr.py` - モデル量子化スクリプト  
- `batch_inference.py` - バッチ推論スクリプト
- `inference_times.txt` - 推論時間記録ファイル
- `gpu_inference_results.csv` - 詳細な性能結果