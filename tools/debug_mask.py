"""
棋盘识别诊断 — 全格 HSV mask 占比，与 classifier.py 逻辑一致。
用法: python tools/debug_mask.py
"""
import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
os.chdir(_PROJ_ROOT)

import cv2
import numpy as np
from detector import BoardDetector
from config import *

IS_WINDOWS = sys.platform.startswith("win")
detector = BoardDetector()
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cap.read()
cap.release()
if not ret:
    print("Failed to capture frame"); sys.exit(1)

cropped, rect = detector.crop(frame)
if cropped is None:
    print("Crop FAILED"); sys.exit(1)
print(f"Cropped OK: {cropped.shape}, rect={rect}")

h, w = cropped.shape[:2]
ch, cw = h // 3, w // 3


def hsv_mask(hsv, hl, hh, sl, sh, vl, vh):
    """HSV inRange，支持 H 跨边界（红色）。"""
    if hl <= hh:
        return cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))
    else:
        m1 = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([180, sh, vh]))
        m2 = cv2.inRange(hsv, np.array([0, sl, vl]), np.array([hh, sh, vh]))
        return cv2.bitwise_or(m1, m2)


results = []
for row in range(3):
    for col in range(3):
        y1, y2 = row * ch, (row + 1) * ch
        x1, x2 = col * cw, (col + 1) * cw
        cell = cropped[y1:y2, x1:x2]
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        total = ch * cw

        iron_mask = hsv_mask(hsv,
            CLASSIFIER_IRON_H_LOW, CLASSIFIER_IRON_H_HIGH,
            CLASSIFIER_IRON_S_LOW, CLASSIFIER_IRON_S_HIGH,
            CLASSIFIER_IRON_V_LOW, CLASSIFIER_IRON_V_HIGH)
        red_mask = hsv_mask(hsv,
            CLASSIFIER_RED_H_LOW, CLASSIFIER_RED_H_HIGH,
            CLASSIFIER_RED_S_LOW, CLASSIFIER_RED_S_HIGH,
            CLASSIFIER_RED_V_LOW, CLASSIFIER_RED_V_HIGH)

        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circ = 0.0
        if cnts:
            la = cv2.contourArea(max(cnts, key=cv2.contourArea))
            lp = cv2.arcLength(max(cnts, key=cv2.contourArea), True)
            if lp > 0:
                circ = 4 * np.pi * la / (lp * lp)

        results.append({
            "cell": cell, "iron_mask": iron_mask, "red_mask": red_mask,
            "iron_ratio": np.count_nonzero(iron_mask) / total,
            "red_ratio": np.count_nonzero(red_mask) / total,
            "circ": circ,
            "v_mean": float(np.mean(hsv[:, :, 2])),
            "s_mean": float(np.mean(hsv[:, :, 1])),
        })

iron_th = CLASSIFIER_MASK_RATIO_MIN
red_th = CLASSIFIER_MASK_RATIO_MIN

print(f"\n=== Cell Analysis (full-cell, threshold = {iron_th:.0%}) ===")
iron_list = [f"{r['iron_ratio']:.1%}" for r in results]
red_list = [f"{r['red_ratio']:.1%}" for r in results]
print(f"Iron values: {iron_list}")
print(f"Red  values: {red_list}")
print(f"{'Idx':>3} {'V':>4} {'S':>4} {'Circ':>5} {'Iron%':>7} {'Red%':>7}  Verdict")

for idx, r in enumerate(results):
    i_ok = r["iron_ratio"] >= iron_th
    r_ok = r["red_ratio"] >= red_th
    if i_ok and not r_ok:
        v = "IRON"
    elif r_ok and not i_ok:
        v = "RED"
    elif i_ok and r_ok:
        v = "CONFLICT"
    elif r["circ"] < CLASSIFIER_CIRCULARITY_MIN and r["v_mean"] > CLASSIFIER_EMPTY_V_MID:
        v = "EMPTY"
    else:
        v = "EMPTY"
    print(f" [{idx}] {r['v_mean']:4.0f} {r['s_mean']:4.0f} {r['circ']:5.2f} "
          f"{r['iron_ratio']:6.1%}  {r['red_ratio']:6.1%}  {v}")

# 可视化：左原图 | 右 mask(蓝=铁, 红=红)
V = 120
grid = np.zeros((3 * V, 3 * 2 * V, 3), dtype=np.uint8)
for idx, r in enumerate(results):
    row, col = idx // 3, idx % 3
    c_resized = cv2.resize(r["cell"], (V, V))
    m_rgb = np.zeros((r["cell"].shape[0], r["cell"].shape[1], 3), dtype=np.uint8)
    m_rgb[r["iron_mask"] > 0] = (255, 0, 0)   # blue = iron
    m_rgb[r["red_mask"] > 0] = (0, 0, 255)    # red = red
    m_rgb = cv2.resize(m_rgb, (V, V))

    i_ok = r["iron_ratio"] >= iron_th
    r_ok = r["red_ratio"] >= red_th
    if i_ok and r_ok:
        color = (0, 255, 255)
    elif i_ok:
        color = (255, 0, 0)
    elif r_ok:
        color = (0, 0, 255)
    else:
        color = (128, 128, 128)
    cv2.rectangle(c_resized, (0, 0), (V - 1, V - 1), color, 3)
    cv2.rectangle(m_rgb, (0, 0), (V - 1, V - 1), color, 3)

    y0, x0 = row * V, col * 2 * V
    grid[y0:y0 + V, x0:x0 + V] = c_resized
    grid[y0:y0 + V, x0 + V:x0 + 2 * V] = m_rgb
    cv2.putText(grid, f"I:{r['iron_ratio']:.0%} R:{r['red_ratio']:.0%}",
                (x0 + 3, y0 + V - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

cv2.imshow("Debug — Left=Orig Right=Mask | Blue=Iron Red=Red", grid)
print("Press any key to close, or Ctrl+C...")
while True:
    key = cv2.waitKey(100) & 0xFF
    if key != 0xFF:
        break
cv2.destroyAllWindows()
