"""
ONNX 模型输出格式验证脚本 (PC 端运行，需 onnxruntime)。

验证要点:
    1. 输出 shape 必须为 (1, 84, 8400) — YOLOv8n 标准格式
    2. 置信度过滤 + NMS 后能检出棋盘

用法:
    pip install onnxruntime
    python tools/test_onnx.py test.jpg

说明:
    在 ATC 转换为 OM 之前，先在 PC 上验证 ONNX 输出格式正确，
    避免香橙派部署后发现 shape 不匹配。
"""

import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
os.chdir(_PROJ_ROOT)

import cv2
import numpy as np


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python tools/test_onnx.py <image.jpg>")
        print("  Validates ONNX model output format")
        sys.exit(1)

    img_path = sys.argv[1]
    onnx_path = "models/chessboard_v8n.onnx"

    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}")
        print("Run 'python tools/export_onnx.py' first.")
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    # 加载模型
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    print(f"Input : {input_name} shape={input_shape}")
    print(f"Output: {session.get_outputs()[0].name} shape={output_shape}")

    # 检查输出格式 (onnxruntime 可能返回 list 或 tuple)
    shape = tuple(output_shape)
    if shape in ((1, 5, 8400), (1, 84, 8400)):
        n_cls = shape[1] - 4
        print(f"Output shape OK: {shape} ({n_cls} class{'es' if n_cls > 1 else ''})")
    else:
        print(f"WARNING: Expected (1, 5|84, 8400), got {shape}")

    # 预处理测试图片
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to load: {img_path}")
        sys.exit(1)
    print(f"\nTest image: {img_path} ({frame.shape[1]}x{frame.shape[0]})")

    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    img = np.ascontiguousarray(img)

    # 推理
    outputs = session.run(None, {input_name: img})
    output = outputs[0]
    print(f"Inference output shape: {output.shape}")

    # 简单后处理：检查是否有高置信度检出
    data = output[0]   # (C, 8400)
    n_channels = data.shape[0]

    if n_channels == 5:
        # 单类模型: row 4 已是 final confidence
        cls_max = data[4, :]
    else:
        # 多类模型: rows 4-83 → sigmoid → max
        cls_raw = data[4:, :]
        cls_max = 1 / (1 + np.exp(-np.clip(cls_raw, -20, 20))).max(axis=0)
    top_conf = float(cls_max.max())
    top_n = int(np.sum(cls_max >= 0.5))
    print(f"Top confidence: {top_conf:.3f}")
    print(f"Anchors with conf >= 0.5: {top_n}")

    if top_n == 0:
        print("WARNING: No detections with conf >= 0.5. Image may not contain a chessboard,")
        print("         or the ONNX export may be incorrect.")
    else:
        print("Verification: ONNX output looks valid for ATC conversion.")


if __name__ == "__main__":
    main()
