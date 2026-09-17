"""
标注可视化校验：在图片上画出 YOLO bbox 标签。

支持格式:
  YOLO detection: class cx cy w h (归一化)
"""

import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)  # 让 tools/ 内脚本能 import 项目根模块
os.chdir(_PROJ_ROOT)            # 统一工作目录为项目根，data/ 路径正确

import cv2
import glob

LABEL_DIR = "data/labels"
IMAGE_DIR = "data/images"


def draw_bbox(img, cx, cy, w, h, color=(0, 255, 0)):
    """画 bbox。"""
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def parse_line(line: str, img_w: int, img_h: int):
    """
    解析 YOLO 标签行。
    标准检测格式: class cx cy w h (归一化)
    OBB 格式:      class x1 y1 x2 y2 x3 y3 x4 y4 (归一化)
    返回: 用于 draw_bbox 的 (cx, cy, w, h) 或 None
    """
    parts = line.strip().split()
    if not parts:
        return None

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    if len(coords) == 4:
        # 标准 bbox: cx cy w h (归一化)
        return class_id, coords[0] * img_w, coords[1] * img_h, coords[2] * img_w, coords[3] * img_h
    elif len(coords) == 8:
        # OBB: 4个角点 → 转 bbox
        xs = [coords[i] * img_w for i in range(0, 8, 2)]
        ys = [coords[i] * img_h for i in range(1, 8, 2)]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        return class_id, cx, cy, w, h
    else:
        return None


def main():
    label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    if not label_files:
        print("No label files found in", LABEL_DIR)
        sys.exit(1)

    print(f"Found {len(label_files)} label files")
    print("Controls: N=Next  P=Prev  Q=Quit")

    idx = 0
    while True:
        label_path = label_files[idx]
        base = os.path.splitext(os.path.basename(label_path))[0]

        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            p = os.path.join(IMAGE_DIR, base + ext)
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            print(f"Image not found for: {base}")
            idx = (idx + 1) % len(label_files)
            continue

        img = cv2.imread(img_path)
        if img is None:
            idx = (idx + 1) % len(label_files)
            continue

        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                result = parse_line(line, w, h)
                if result:
                    _, cx, cy, bw, bh = result
                    draw_bbox(img, cx, cy, bw, bh)

        cv2.putText(img, f"[{idx+1}/{len(label_files)}] {base}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Label Check", img)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("n"), ord(" ")):
            idx = (idx + 1) % len(label_files)
        elif key == ord("p"):
            idx = (idx - 1) % len(label_files)
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
