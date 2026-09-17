"""验证 detector.crop() 裁切效果"""
import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)  # 让 tools/ 内脚本能 import 项目根模块
os.chdir(_PROJ_ROOT)            # 统一工作目录为项目根

import cv2
from detector import BoardDetector

path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
img = cv2.imread(path)
if img is None:
    print(f"Failed to load: {path}")
    sys.exit(1)

det = BoardDetector()
cropped, rect = det.crop(img)

if cropped is None:
    print("FAILED: board not found")
    sys.exit(1)

print(f"Input:    {img.shape[1]}x{img.shape[0]}")
print(f"Cropped:  {cropped.shape[1]}x{cropped.shape[0]}")
print(f"Rect:     {rect}")
print(f"Reduced:  {(1 - cropped.shape[1]*cropped.shape[0] / (img.shape[1]*img.shape[0])) * 100:.0f}%")

out = "test_cropped.jpg"
cv2.imwrite(out, cropped)
print(f"Saved:    {out}")
