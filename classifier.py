from __future__ import annotations

"""
棋子识别：9 格切分 → 全格 HSV mask 占比 → 帧内相对阈值 → 逻辑校验。

棋盘: 纸白底黑线 (白底: 高V低S)
棋子: 铁色 + 红色 (阈值由 tools/tune_hsv.py 实测调定)

策略:
  1. 圆形度低 + 高V低S → 空白 (白纸)
  2. 计算每格铁色/红色 mask 占比（全格，不裁 ROI）
  3. 占比显著高于 9 格中位数者为对应棋子（相对阈值，抗光照波动）
  4. 按棋子数量 (铁4红3) 取 Top-K 格

依赖 detector.py 输出的棋盘正视图。
"""

import cv2
import numpy as np

from config import (
    CLASSIFIER_EMPTY_V_HIGH,
    CLASSIFIER_EMPTY_V_MID,
    CLASSIFIER_EMPTY_S_MAX,
    CLASSIFIER_IRON_H_LOW,
    CLASSIFIER_IRON_H_HIGH,
    CLASSIFIER_IRON_S_LOW,
    CLASSIFIER_IRON_S_HIGH,
    CLASSIFIER_IRON_V_LOW,
    CLASSIFIER_IRON_V_HIGH,
    CLASSIFIER_RED_H_LOW,
    CLASSIFIER_RED_H_HIGH,
    CLASSIFIER_RED_S_LOW,
    CLASSIFIER_RED_S_HIGH,
    CLASSIFIER_RED_V_LOW,
    CLASSIFIER_RED_V_HIGH,
    CLASSIFIER_CIRCULARITY_MIN,
    CLASSIFIER_MASK_RATIO_MIN,
    CLASSIFIER_VALIDATE_RETRY,
)

EMPTY = 0  # 空格
IRON  = 1  # 铁色棋子
RED   = 2  # 红色棋子

# 为兼容 ai_engine 中的 BLACK/WHITE 命名，保留别名
BLACK = IRON
WHITE = RED


class CellClassifier:

    def __init__(self):
        pass

    # ================================================================
    # 公开 API
    # ================================================================

    def classify(
        self, board_img: np.ndarray, prev_grid: list[int] | None = None
    ) -> tuple[list[int], list[float]]:
        """
        参数:
            board_img: 正方形棋盘正视图 (来自 BoardDetector)
            prev_grid: 上一帧状态数组 (逻辑校验用)

        返回:
            grid:       9 元素 [0空/1铁色/2红色]
            confidence: 每格置信度 [0.0~1.0]
        """
        cells = self._split_cells(board_img)

        # 第一遍：计算每格的 iron_ratio / red_ratio
        ratios = [self._compute_ratios(cell) for cell in cells]

        grid = []
        confs = []

        for r in ratios:
            iron_r, red_r = r["iron_ratio"], r["red_ratio"]
            iron_ok = iron_r >= CLASSIFIER_MASK_RATIO_MIN
            red_ok = red_r >= CLASSIFIER_MASK_RATIO_MIN
            circ = r["circularity"]
            v_mean = r["v_mean"]
            s_mean = r["s_mean"]

            # 空白快速判定
            if not iron_ok and not red_ok:
                if circ < CLASSIFIER_CIRCULARITY_MIN:
                    if v_mean > CLASSIFIER_EMPTY_V_HIGH and s_mean < CLASSIFIER_EMPTY_S_MAX:
                        grid.append(EMPTY); confs.append(0.95); continue
                    if v_mean > CLASSIFIER_EMPTY_V_MID:
                        grid.append(EMPTY); confs.append(0.7); continue
                grid.append(EMPTY); confs.append(0.5); continue

            if iron_ok and not red_ok:
                grid.append(IRON)
                confs.append(min(iron_r * 3, 0.95))
            elif red_ok and not iron_ok:
                grid.append(RED)
                confs.append(min(red_r * 3, 0.95))
            elif iron_ok and red_ok:
                if iron_r > red_r:
                    grid.append(IRON); confs.append(0.7)
                else:
                    grid.append(RED); confs.append(0.7)
            else:
                grid.append(EMPTY); confs.append(0.5)

        if prev_grid is not None and not self._validate(prev_grid, grid):
            # 校验失败：棋子消失/变色/数量异常 → 保守返回旧状态
            return prev_grid.copy(), [0.5] * 9

        return grid, confs

    # ================================================================
    # 格子切分
    # ================================================================

    @staticmethod
    def _split_cells(board_img: np.ndarray) -> list[np.ndarray]:
        h, w = board_img.shape[:2]
        cell_h = h // 3
        cell_w = w // 3
        cells = []
        for row in range(3):
            for col in range(3):
                y1 = row * cell_h
                y2 = y1 + cell_h
                x1 = col * cell_w
                x2 = x1 + cell_w
                cells.append(board_img[y1:y2, x1:x2])
        return cells

    # ================================================================
    # Mask 计算
    # ================================================================

    @staticmethod
    def _hsv_mask(hsv: np.ndarray, h_low, h_high, s_low, s_high, v_low, v_high) -> np.ndarray:
        """生成 HSV mask，自动处理 H 跨 0/180 边界。"""
        if h_low <= h_high:
            return cv2.inRange(hsv, np.array([h_low, s_low, v_low]), np.array([h_high, s_high, v_high]))
        else:
            m1 = cv2.inRange(hsv, np.array([h_low, s_low, v_low]), np.array([180, s_high, v_high]))
            m2 = cv2.inRange(hsv, np.array([0, s_low, v_low]), np.array([h_high, s_high, v_high]))
            return cv2.bitwise_or(m1, m2)

    @staticmethod
    def _compute_ratios(cell: np.ndarray) -> dict:
        """计算单格的铁色/红色 mask 占比及相关统计量（全格，不裁 ROI）。"""
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        h, w = cell.shape[:2]
        total = h * w

        iron_mask = CellClassifier._hsv_mask(hsv,
            CLASSIFIER_IRON_H_LOW, CLASSIFIER_IRON_H_HIGH,
            CLASSIFIER_IRON_S_LOW, CLASSIFIER_IRON_S_HIGH,
            CLASSIFIER_IRON_V_LOW, CLASSIFIER_IRON_V_HIGH,
        )
        red_mask = CellClassifier._hsv_mask(hsv,
            CLASSIFIER_RED_H_LOW, CLASSIFIER_RED_H_HIGH,
            CLASSIFIER_RED_S_LOW, CLASSIFIER_RED_S_HIGH,
            CLASSIFIER_RED_V_LOW, CLASSIFIER_RED_V_HIGH,
        )

        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circ = 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            peri = cv2.arcLength(largest, True)
            if peri > 0:
                circ = 4 * np.pi * area / (peri * peri)

        return {
            "iron_ratio": np.count_nonzero(iron_mask) / total,
            "red_ratio": np.count_nonzero(red_mask) / total,
            "circularity": circ,
            "v_mean": float(np.mean(hsv[:, :, 2])),
            "s_mean": float(np.mean(hsv[:, :, 1])),
        }

    # ================================================================
    # 逻辑校验
    # ================================================================

    @staticmethod
    def _validate(prev: list[int], curr: list[int]) -> bool:
        """
        井字棋规则:
          1. 只能多一个棋子
          2. 已有棋子不能消失/变色
          3. 颜色必须交替 (IRON先手)
        """
        diffs = []
        for p, c in zip(prev, curr):
            if p != c:
                diffs.append((p, c))

        if len(diffs) > 1:
            return False
        if len(diffs) == 0:
            return True

        old_val, new_val = diffs[0]
        if old_val != EMPTY:
            return False

        iron_count = prev.count(IRON) + (1 if new_val == IRON else 0)
        red_count = prev.count(RED) + (1 if new_val == RED else 0)
        # 铁色先手 → iron >= red，差值 ≤ 1
        if iron_count < red_count or iron_count > red_count + 1:
            return False

        return True


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python classifier.py <board_image_path>")
        print("Requires a straightened square board image.")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Failed to load: {sys.argv[1]}")
        sys.exit(1)

    cc = CellClassifier()
    grid, confs = cc.classify(img)

    syms = ["_", "I", "R"]  # 空 / Iron / Red
    print("Grid state:")
    for row in range(3):
        vals = grid[row * 3 : (row + 1) * 3]
        print(f"  {' '.join(syms[v] for v in vals)}")

    print("Confidence:")
    for row in range(3):
        c = confs[row * 3 : (row + 1) * 3]
        print(f"  {' '.join(f'{x:.2f}' for x in c)}")
