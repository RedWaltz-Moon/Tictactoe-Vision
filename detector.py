from __future__ import annotations

"""
棋盘检测：轮廓等比裁切 → 固定角度旋转校正 → YOLO/CV 精确定位 → 输出正视图。

流程（旋转场景）:
  1. 轮廓检测 → bounding rect 等比裁切（丢弃无关背景，保持宽高比）
  2. 绕中心旋转 BOARD_ROTATION_DEG 度
  3. YOLO 检测棋盘 bbox（或 CV fallback）→ 精确裁出棋盘
  4. 正方形输出

正视场景：BOARD_ROTATION_DEG = 0，跳过步骤 2 即可。
"""

import cv2
import numpy as np
from collections import deque

from config import (
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_MISSING_FRAME_LIMIT,
    BACKEND,
    ASCEND_MODEL_PATH,
    ASCEND_INPUT_SIZE,
    CORNER_SHIFT_THRESHOLD_PX,
    CORNER_SMOOTH_WINDOW,
    DETECT_THRESH_METHOD,
    DETECT_MANUAL_THRESH,
    DETECT_MORPH_KERNEL,
    DETECT_MORPH_CLOSE_ITER,
    DETECT_BLUR_SIZE,
    DETECT_MIN_AREA_RATIO,
    CROP_MARGIN_PX,
    BOARD_ROTATION_DEG,
    ROTATION_ANGLE_THRESH,
)


class BoardDetector:

    def __init__(self, model_path: str | None = None, backend: str | None = None):
        if backend is None:
            backend = BACKEND

        self.model = None

        if backend == "ascend":
            self.model_path: str = model_path or ASCEND_MODEL_PATH
            try:
                from ascend_inference import AscendInference
                self.model = AscendInference(self.model_path, input_size=ASCEND_INPUT_SIZE)
            except Exception as e:
                print(f"[detector] Ascend backend unavailable: {e}")
        else:  # "ultralytics" (PC)
            self.model_path: str = model_path or YOLO_MODEL_PATH
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            except Exception:
                pass

        self._corner_history: deque[np.ndarray] | None = None
        self._missing_count = 0
        self._step1_raw_bbox: tuple | None = None  # 步骤1扩张margin前的原始bbox

    # ================================================================
    # 公开 API
    # ================================================================

    def detect(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        完整检测流水线:
          等比裁切 → 自动判断旋转 → YOLO/CV 定位 → 正方形输出

        返回:
            board_img: 正方形棋盘正视图（只有棋盘，无背景）
            corners:   原始帧上裁切框四角 [左上, 右上, 左下, 右下]
        """
        # ── 第 1 步: 轮廓等比裁切 ──
        bbox = self._find_bounding_rect(frame)
        if bbox is None:
            self._missing_count += 1
            return None, None

        self._missing_count = 0
        x1, y1, x2, y2 = bbox

        corners = np.array(
            [[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32
        )
        corners = self._smooth_corners(corners)

        x1, y1 = max(0, int(corners[0][0])), max(0, int(corners[0][1]))
        x2, y2 = min(frame.shape[1], int(corners[3][0])), min(frame.shape[0], int(corners[3][1]))
        roi = frame[y1:y2, x1:x2]

        # ── 第 2 步: 自动判断是否需要旋转 ──
        rotated = self._is_rotated(roi)
        if rotated:
            roi = self._rotate(roi, BOARD_ROTATION_DEG)

        # ── 第 3 步: 精确定位棋盘 → 裁切 ──
        if rotated:
            # 旋转后: YOLO/CV 重新检测（去掉黑边/背景）
            if self.model is not None:
                refined = self._find_bbox_yolo(roi)
            else:
                refined = self._find_bbox_fallback(roi)
            if refined is not None:
                bx1, by1, bx2, by2 = refined
                roi = roi[by1:by2, bx1:bx2]
        elif self._step1_raw_bbox is not None:
            # 正视: 直接复用步骤1的原始bbox（预margin），映射到ROI坐标
            rx1, ry1, rx2, ry2 = self._step1_raw_bbox
            ox, oy = int(corners[0][0]), int(corners[0][1])
            bx1, by1 = rx1 - ox, ry1 - oy
            bx2, by2 = rx2 - ox, ry2 - oy
            h, w = roi.shape[:2]
            bx1 = max(0, bx1); by1 = max(0, by1)
            bx2 = min(w, bx2); by2 = min(h, by2)
            if bx2 > bx1 and by2 > by1:
                roi = roi[by1:by2, bx1:bx2]

        if roi.size == 0:
            return None, None

        # ── 第 4 步: 正方形输出 ──
        ch, cw = roi.shape[:2]
        side = max(ch, cw)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        square[0:ch, 0:cw] = roi
        return square, corners

    def crop(self, frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
        board_img, corners = self.detect(frame)
        if board_img is not None:
            x1 = int(min(corners[:, 0]))
            y1 = int(min(corners[:, 1]))
            x2 = int(max(corners[:, 0]))
            y2 = int(max(corners[:, 1]))
            return board_img, (x1, y1, x2, y2)
        self._missing_count += 1
        return None, None

    @property
    def missing(self) -> bool:
        return self._missing_count >= YOLO_MISSING_FRAME_LIMIT

    def reset(self) -> None:
        self._corner_history = None
        self._missing_count = 0

    # ================================================================
    # 第 1 步: 轮廓等比裁切
    # ================================================================

    def _find_bounding_rect(self, frame: np.ndarray) -> tuple | None:
        """
        全图轮廓检测 → 最大轮廓 bounding rect + margin → 等比裁切。
        优先 YOLO bbox，无模型时用传统 CV。
        """
        # YOLO 优先
        if self.model is not None:
            results = self.model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0][:4])
                    self._step1_raw_bbox = (x1, y1, x2, y2)
                    return self._expand_rect(frame, x1, y1, x2, y2)

        # Fallback: 传统 CV
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bs = DETECT_BLUR_SIZE if DETECT_BLUR_SIZE % 2 == 1 else DETECT_BLUR_SIZE + 1
        blurred = cv2.GaussianBlur(gray, (bs, bs), 0)

        if DETECT_THRESH_METHOD == 0:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif DETECT_THRESH_METHOD == 1:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, bs | 1, 2)
        else:
            _, thresh = cv2.threshold(blurred, DETECT_MANUAL_THRESH, 255, cv2.THRESH_BINARY_INV)

        mk = max(1, DETECT_MORPH_KERNEL)
        kernel = np.ones((mk, mk), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=DETECT_MORPH_CLOSE_ITER)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) / (fh * fw) < DETECT_MIN_AREA_RATIO:
            return None

        x, y, rw, rh = cv2.boundingRect(largest)
        self._step1_raw_bbox = (x, y, x + rw, y + rh)
        return self._expand_rect(frame, x, y, x + rw, y + rh)

    @staticmethod
    def _expand_rect(frame, x1, y1, x2, y2):
        fh, fw = frame.shape[:2]
        m = CROP_MARGIN_PX
        return (
            max(0, x1 - m),
            max(0, y1 - m),
            min(fw, x2 + m),
            min(fh, y2 + m),
        )

    # ================================================================
    # 第 2 步: 旋转检测 + 旋转
    # ================================================================

    @staticmethod
    def _is_rotated(roi: np.ndarray) -> bool:
        """
        用 minAreaRect 角度判断棋盘是否旋转:
          正视: 棋盘边与 ROI 边平行 → angle ≈ 0°
          45°: 棋盘边与 ROI 边成 45° → angle ≈ 45°

        正方形对称性: 0° 和 90° 等价 → 归一化到 [0, 45] 后判断。
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return False
        rect = cv2.minAreaRect(largest)  # (center, (w, h), angle)
        angle = abs(rect[2])             # angle ∈ [0, 90)
        if angle > 45:
            angle = 90 - angle           # 归一化到 [0, 45]
        return angle > ROTATION_ANGLE_THRESH

    @staticmethod
    def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
        """绕图像中心旋转，扩大画布容纳全部内容。"""
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_a = abs(rot_mat[0, 0])
        sin_a = abs(rot_mat[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        rot_mat[0, 2] += new_w / 2 - center[0]
        rot_mat[1, 2] += new_h / 2 - center[1]
        return cv2.warpAffine(img, rot_mat, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    # ================================================================
    # 第 3 步: YOLO/CV 精确定位棋盘（旋转场景）
    # ================================================================

    def _find_bbox_yolo(self, frame: np.ndarray) -> tuple | None:
        results = self.model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            return None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if boxes is None or len(boxes) == 0:
            return None
        x1, y1, x2, y2 = map(int, boxes[0][:4])
        # 不额外扩 margin，YOLO 已经给出棋盘边界
        fh, fw = frame.shape[:2]
        return (max(0, x1), max(0, y1), min(fw, x2), min(fh, y2))

    def _find_bbox_fallback(self, frame: np.ndarray) -> tuple | None:
        """传统 CV：自适应二值化 → 最大轮廓 → bounding rect。"""
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bs = DETECT_BLUR_SIZE if DETECT_BLUR_SIZE % 2 == 1 else DETECT_BLUR_SIZE + 1
        blurred = cv2.GaussianBlur(gray, (bs, bs), 0)

        if DETECT_THRESH_METHOD == 0:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif DETECT_THRESH_METHOD == 1:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, bs | 1, 2)
        else:
            _, thresh = cv2.threshold(blurred, DETECT_MANUAL_THRESH, 255, cv2.THRESH_BINARY_INV)

        mk = max(1, DETECT_MORPH_KERNEL)
        kernel = np.ones((mk, mk), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=DETECT_MORPH_CLOSE_ITER)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) / (fh * fw) < DETECT_MIN_AREA_RATIO:
            return None

        x, y, rw, rh = cv2.boundingRect(largest)
        return (max(0, x), max(0, y), min(fw, x + rw), min(fh, y + rh))

    # ================================================================
    # 辅助
    # ================================================================

    def _smooth_corners(self, corners: np.ndarray) -> np.ndarray:
        if self._corner_history is None:
            self._corner_history = deque(maxlen=CORNER_SMOOTH_WINDOW)
            for _ in range(CORNER_SMOOTH_WINDOW):
                self._corner_history.append(corners.copy())
        else:
            self._corner_history.append(corners.copy())

        if len(self._corner_history) >= 2:
            prev = self._corner_history[-2]
            shift = np.max(np.linalg.norm(corners - prev, axis=1))
            if shift > CORNER_SHIFT_THRESHOLD_PX:
                return prev.copy()

        return np.mean(np.array(self._corner_history), axis=0).astype(np.float32)


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path>")
        print("Shows the detected board after rotation correction.")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Failed to load: {sys.argv[1]}")
        sys.exit(1)

    det = BoardDetector()
    board, corners = det.detect(img)

    if board is None:
        print("Board not found.")
        sys.exit(1)

    vis = img.copy()
    for i, (x, y) in enumerate(corners.astype(int)):
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(
            vis, f"P{i}", (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    cv2.imshow("Original + BBox", vis)
    cv2.imshow("Rotated Board", board)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
