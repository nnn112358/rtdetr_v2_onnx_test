#!/bin/bash
# RT-DETR 系のモデルをまとめて ONNX に変換してリネームするスクリプト

models=(
  rtdetr_v2_r18vd
  rtdetr_v2_r34vd
  rtdetr_v2_r50vd
  rtdetr_v2_r101vd
  rtdetr_r18vd
  rtdetr_r34vd
  rtdetr_r50vd
  rtdetr_r101vd
)

for model in "${models[@]}"; do
  echo "=== Exporting $model to ONNX ==="
  uv run optimum-cli export onnx \
    --model PekingU/$model \
    --task object-detection \
    $model

  if [ -f "$model/model.onnx" ]; then
    mv "$model/model.onnx" "$model/$model.onnx"
    echo "Renamed to $model/$model.onnx"
  else
    echo "⚠️  $model/model.onnx が見つかりませんでした"
  fi
done

echo "✅ 全てのモデルを処理しました"

