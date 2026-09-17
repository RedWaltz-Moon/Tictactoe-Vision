"""
YOLOv8n 训练脚本 (标准检测，非 OBB)。
labelme 标注的是轴对齐矩形框，所以用标准 YOLO detection。

用法:
    python train.py prepare   — 划分 train/val
    python train.py train     — 训练
    python train.py all       — 准备 + 训练
"""

import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)  # 让 tools/ 内脚本能 import 项目根模块
os.chdir(_PROJ_ROOT)            # 统一工作目录为项目根，data/ models/ 路径正确

import shutil
from pathlib import Path

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

TRAIN_IMG = IMAGES_DIR / "train"
VAL_IMG = IMAGES_DIR / "val"
TRAIN_LBL = LABELS_DIR / "train"
VAL_LBL = LABELS_DIR / "val"


def prepare_splits(train_ratio: float = 0.85):
    images = sorted(IMAGES_DIR.glob("*.jpg")) + \
              sorted(IMAGES_DIR.glob("*.jpeg")) + \
              sorted(IMAGES_DIR.glob("*.png"))

    if not images:
        print("No images found in data/images/")
        return False

    split = int(len(images) * train_ratio)
    train_imgs = images[:split]
    val_imgs = images[split:]

    for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    def copy_files(file_list, img_dst, lbl_dst):
        for img_path in file_list:
            shutil.copy2(img_path, img_dst / img_path.name)
            lbl_name = img_path.stem + ".txt"
            lbl_path = LABELS_DIR / lbl_name
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_name)
            else:
                print(f"  WARNING: No label for {img_path.name}")

    print(f"Train: {len(train_imgs)} images")
    copy_files(train_imgs, TRAIN_IMG, TRAIN_LBL)
    print(f"Val:   {len(val_imgs)} images")
    copy_files(val_imgs, VAL_IMG, VAL_LBL)
    print("Split done.")
    return True


def train():
    from ultralytics import YOLO

    # 标准检测预训练权重 (不是 OBB)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="chessboard",
        patience=15,
        save=True,
        device="cpu",
        verbose=True,
    )

    best = Path("runs/detect/chessboard/weights/best.pt")
    if best.exists():
        dst = Path("models/chessboard_v8n.pt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"Model saved to: {dst}")

    return results


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python train.py prepare   — split data into train/val")
        print("  python train.py train     — train YOLOv8n")
        print("  python train.py all       — prepare + train")
        return

    cmd = sys.argv[1]
    if cmd in ("prepare", "all"):
        if not prepare_splits():
            return
    if cmd in ("train", "all"):
        train()


if __name__ == "__main__":
    main()
