"""
labelme JSON → YOLO 检测格式转换。

labelme 标注: Create Rectangle 画棋盘外框，标签 "chessboard"
输出 YOLO: class x_center y_center width height (归一化坐标)

用法:
    python labelme_to_yolo.py                    # 转换 data/labels/ 下所有 JSON
    python labelme_to_yolo.py --show             # 转换 + 可视化校验
"""

import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)  # 让 tools/ 内脚本能 import 项目根模块
os.chdir(_PROJ_ROOT)            # 统一工作目录为项目根，data/ 路径正确

import json
import glob
import cv2
import numpy as np


LABELME_DIR = "data/labels"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "data/labels"


def rectangle_to_bbox(points: list, img_w: int, img_h: int) -> str:
    """
    labelme rectangle 有 2 个对角点 → YOLO 标准 bbox。
    返回: "0 xc yc w h" (归一化)
    """
    x1, y1 = points[0]
    x2, y2 = points[1]

    # 计算中心 + 宽高 (归一化)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h

    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def polygon_to_bbox(points: list, img_w: int, img_h: int) -> str:
    """
    polygon (4 个角点) → YOLO 标准 bbox (外接矩形)。
    """
    pts = np.array(points, dtype=np.float32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_single(json_path: str, output_dir: str) -> bool:
    """转换单个 labelme JSON → YOLO txt。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth", 0)
    img_h = data.get("imageHeight", 0)

    # 如果 JSON 没有尺寸，从图片文件读取
    if img_w == 0 or img_h == 0:
        img_name = data.get("imagePath", "")
        img_path = os.path.join(IMAGE_DIR, os.path.basename(img_name))
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_h, img_w = img.shape[:2]

    if img_w == 0 or img_h == 0:
        print(f"  SKIP {os.path.basename(json_path)}: no image size")
        return False

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label.lower() not in ("chessboard", "board", "0"):
            continue

        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])

        if shape_type == "rectangle":
            if len(points) != 2:
                print(f"  WARN {os.path.basename(json_path)}: rectangle has {len(points)} points")
                continue
            line = rectangle_to_bbox(points, img_w, img_h)
            lines.append(line)

        elif shape_type == "polygon":
            if len(points) < 4:
                print(f"  WARN {os.path.basename(json_path)}: polygon has {len(points)} points")
                continue
            line = polygon_to_bbox(points, img_w, img_h)
            lines.append(line)

        else:
            print(f"  WARN {os.path.basename(json_path)}: unsupported shape_type={shape_type}")

    if not lines:
        print(f"  WARN {os.path.basename(json_path)}: no valid chessboard shape found")
        return False

    # 写 YOLO txt
    basename = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(output_dir, basename + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return True


def main():
    show = "--show" in sys.argv
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = glob.glob(os.path.join(LABELME_DIR, "*.json"))
    if not json_files:
        print(f"No JSON files found in {LABELME_DIR}/")
        print("Place your labelme .json annotations in data/labels/")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files")
    ok = 0
    for jf in sorted(json_files):
        if convert_single(jf, OUTPUT_DIR):
            ok += 1

    print(f"Converted: {ok}/{len(json_files)}")
    print(f"Output: {OUTPUT_DIR}/")

    if show and ok > 0:
        print("\n--- Visual check ---")
        print("Controls: N=Next  P=Prev  Q=Quit")
        from verify_labels import main as verify_main
        verify_main()


if __name__ == "__main__":
    main()
