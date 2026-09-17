"""
棋盘裁切调参工具 — 只做矩形裁切，丢弃无关画面，不改变比例/不透视变换。

用法:
  python tune_crop.py test.jpg    # 静态图片
  python tune_crop.py             # 摄像头实时

显示:  左侧原图+裁切框  |  右侧裁切后画面

可调参数:
  ThreshMethod  — 0=Otsu, 1=Adaptive, 2=Manual
  ManualThresh  — 手动二值化阈值（Method=2时生效）
  MorphKernel   — 形态学核大小
  CloseIter     — 闭操作迭代次数
  BlurSize      — 高斯模糊核
  Margin        — 裁切框外扩像素
  MinArea/100   — 最小轮廓面积比例

操作:  s=打印参数  q=退出
"""

import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)  # 让 tools/ 内脚本能 import 项目根模块
os.chdir(_PROJ_ROOT)            # 统一工作目录为项目根

import cv2
import numpy as np

IS_WINDOWS = sys.platform.startswith("win")
DEV_INDEX = 0
FRAME_W, FRAME_H = 640, 480

WIN = "Crop Tuner | s=Print q=Quit"


def nothing(x):
    pass


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else None
    cap = None
    static_frame = None

    if image_path:
        static_frame = cv2.imread(image_path)
        if static_frame is None:
            print(f"Failed to load: {image_path}")
            return
        print(f"Loaded: {image_path}")
    else:
        cap = cv2.VideoCapture(DEV_INDEX, cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {DEV_INDEX} not available")

    cv2.namedWindow(WIN)
    cv2.createTrackbar("ThreshMethod", WIN, 0, 2, nothing)
    cv2.createTrackbar("ManualThresh", WIN, 127, 255, nothing)
    cv2.createTrackbar("MorphKernel",  WIN, 3, 15, nothing)
    cv2.createTrackbar("CloseIter",    WIN, 2, 5, nothing)
    cv2.createTrackbar("BlurSize",     WIN, 5, 31, nothing)
    cv2.createTrackbar("Margin",       WIN, 20, 200, nothing)
    cv2.createTrackbar("MinArea/100",  WIN, 10, 100, nothing)

    print("Crop Tuner: Left=original+crop rect | Right=cropped")
    print()

    while True:
        if static_frame is not None:
            frame = static_frame.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                continue

        # ---- 读滑条 ----
        tm = cv2.getTrackbarPos("ThreshMethod", WIN)
        mt = cv2.getTrackbarPos("ManualThresh", WIN)
        mk = cv2.getTrackbarPos("MorphKernel",  WIN) or 1
        mi = cv2.getTrackbarPos("CloseIter",    WIN)
        bs = cv2.getTrackbarPos("BlurSize",     WIN)
        margin = cv2.getTrackbarPos("Margin",   WIN)
        mr = cv2.getTrackbarPos("MinArea/100",  WIN) / 100.0
        if bs % 2 == 0:
            bs += 1

        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (bs, bs), 0)

        # ---- 二值化 ----
        if tm == 0:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif tm == 1:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, bs | 1, 2)
        else:
            _, thresh = cv2.threshold(blurred, mt, 255, cv2.THRESH_BINARY_INV)

        # ---- 形态学 ----
        kernel = np.ones((mk, mk), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=mi)

        # ---- 找最大轮廓 → bounding rect ----
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) / (fh * fw) >= mr:
                x, y, rw, rh = cv2.boundingRect(largest)
                # 外扩 margin，不超出画面
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(fw, x + rw + margin)
                y2 = min(fh, y + rh + margin)
                rect = (x1, y1, x2, y2)

        # ---- 裁切 ----
        if rect is not None:
            x1, y1, x2, y2 = rect
            cropped = frame[y1:y2, x1:x2]
        else:
            cropped = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(cropped, "NO BOARD", (5, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # ---- 原图 + 裁切框 ----
        vis = frame.copy()
        if rect is not None:
            x1, y1, x2, y2 = rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---- 拼接 ----
        vis = cv2.resize(vis, (FRAME_W, FRAME_H))
        # 裁切图缩放到同样高度，保持宽高比
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            crop_h = FRAME_H
            crop_w = int(cropped.shape[1] * FRAME_H / cropped.shape[0])
            cropped_disp = cv2.resize(cropped, (crop_w, crop_h))
        else:
            cropped_disp = cropped

        display = np.hstack([vis, cropped_disp])

        info = f"M:{tm} K:{mk} It:{mi} Blur:{bs} Mar:{margin} Area>{mr:.0%} Th:{mt}"
        cv2.putText(display, info, (10, display.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display, f"{fw}x{fh} -> {cropped.shape[1]}x{cropped.shape[0]}",
                    (FRAME_W + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            print(f"\nDETECT_THRESH_METHOD    = {tm}")
            print(f"DETECT_MANUAL_THRESH    = {mt}")
            print(f"DETECT_MORPH_KERNEL     = {mk}")
            print(f"DETECT_MORPH_CLOSE_ITER = {mi}")
            print(f"DETECT_BLUR_SIZE        = {bs}")
            print(f"CROP_MARGIN_PX          = {margin}")
            print(f"DETECT_MIN_AREA_RATIO   = {mr:.2f}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
