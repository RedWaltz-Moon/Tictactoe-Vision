"""
YOLOv8n PT → ONNX 导出脚本 (PC 端运行，需 ultralytics)。

用法:
    python tools/export_onnx.py

输出:
    models/chessboard_v8n.onnx  (~6MB, opset=12, simplify=True)

说明:
    ONNX 模型随后传输到香橙派，用 ATC 编译为 OM 模型供 NPU 推理。
    opset=12 对 Ascend 310B 兼容性好；simplify 清理冗余算子减少 ATC 报错风险。
"""

import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
os.chdir(_PROJ_ROOT)

from ultralytics import YOLO


def main():
    pt_path = "models/chessboard_v8n.pt"

    print(f"Loading: {pt_path}")
    model = YOLO(pt_path)

    print("Exporting to ONNX...")
    # opset=12: Ascend 310B ATC 兼容性好
    # simplify=True: 清理冗余算子
    # half=False: 保持 FP32 精度
    model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        half=False,
    )

    onnx_path = "models/chessboard_v8n.onnx"
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Done: {onnx_path} ({size_mb:.1f} MB)")
    else:
        print("Export failed — ONNX file not created.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
