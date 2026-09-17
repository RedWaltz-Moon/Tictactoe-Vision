"""
HSV 阈值调参工具 — 用于测定铁色和红色棋子的 HSV 区间。

用法:
  python tools/tune_hsv.py                      # 摄像头实时调节
  python tools/tune_hsv.py image.jpg            # 对静态图片调节
  python tools/tune_hsv.py                      # 按 L 加载图片

操作:
  鼠标点击 → 终端打印该像素的 HSV 值（采样用）
  滑条拖拽 → 实时看二值化 mask（铁色/红色分两个窗口）
  按 1    → 切换到铁色调参模式
  按 2    → 切换到红色调参模式
  按 s    → 终端打印当前阈值 config 代码
  q      → 退出

说明:
  铁色棋子: 低 V（暗），低 S（灰），H 不重要（暗色无彩色色调）
  红色棋子: H 在 0-10 或 160-180，S 中高，V 中
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

# 窗口名
WIN_ORIGINAL = "Original | Click=Sample HSV | 1=Iron 2=Red | s=Print | q=Quit"
WIN_MASK = "Mask Preview"

# ---- 铁色默认阈值 ----
IRON_H_LOW, IRON_H_HIGH = 0, 180    # 铁色不限 H
IRON_S_LOW, IRON_S_HIGH = 0, 60     # 低 S（暗灰）
IRON_V_LOW, IRON_V_HIGH = 0, 100    # 低 V（暗）

# ---- 红色默认阈值 ----
RED_H_LOW, RED_H_HIGH = 0, 10       # 红色第一段
RED_S_LOW, RED_S_HIGH = 50, 255     # 中高 S
RED_V_LOW, RED_V_HIGH = 50, 255     # 中高 V

MODE_IRON = 0
MODE_RED = 1
current_mode = MODE_IRON


def nothing(x):
    pass


def create_trackbars(window, h_low, h_high, s_low, s_high, v_low, v_high):
    """创建 HSV 6 通道滑条"""
    cv2.createTrackbar("H_low",  window, h_low,  180, nothing)
    cv2.createTrackbar("H_high", window, h_high, 180, nothing)
    cv2.createTrackbar("S_low",  window, s_low,  255, nothing)
    cv2.createTrackbar("S_high", window, s_high, 255, nothing)
    cv2.createTrackbar("V_low",  window, v_low,  255, nothing)
    cv2.createTrackbar("V_high", window, v_high, 255, nothing)


def get_trackbar_values(window):
    """读取当前滑条值"""
    hl = cv2.getTrackbarPos("H_low",  window)
    hh = cv2.getTrackbarPos("H_high", window)
    sl = cv2.getTrackbarPos("S_low",  window)
    sh = cv2.getTrackbarPos("S_high", window)
    vl = cv2.getTrackbarPos("V_low",  window)
    vh = cv2.getTrackbarPos("V_high", window)
    return hl, hh, sl, sh, vl, vh


def set_trackbar_values(window, hl, hh, sl, sh, vl, vh):
    """设置滑条值"""
    cv2.setTrackbarPos("H_low",  window, hl)
    cv2.setTrackbarPos("H_high", window, hh)
    cv2.setTrackbarPos("S_low",  window, sl)
    cv2.setTrackbarPos("S_high", window, sh)
    cv2.setTrackbarPos("V_low",  window, vl)
    cv2.setTrackbarPos("V_high", window, vh)


def hsv_mask(hsv, hl, hh, sl, sh, vl, vh):
    """
    生成 HSV mask。
    红色可能跨越 H=0/180 边界，需分段处理。
    """
    lower = np.array([hl, sl, vl])
    upper = np.array([hh, sh, vh])

    if hl <= hh:
        # 正常区间（大多数颜色）
        return cv2.inRange(hsv, lower, upper)
    else:
        # H 跨边界（红色：hl>hh, 如 160-10）
        mask1 = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([180, sh, vh]))
        mask2 = cv2.inRange(hsv, np.array([0, sl, vl]), np.array([hh, sh, vh]))
        return cv2.bitwise_or(mask1, mask2)


# 鼠标回调：点击采样
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param[0]  # param 是列表包装的 frame 引用
        if frame is not None and 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            bgr = frame[y, x]
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
            print(f"  Click @ ({x},{y})  BGR=({bgr[2]},{bgr[1]},{bgr[0]})  HSV=(H:{hsv[0]}, S:{hsv[1]}, V:{hsv[2]})")


def main():
    global current_mode

    # 解析参数
    image_path = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]

    # 初始化相机或加载图片
    cap = None
    static_frame = None
    if image_path:
        static_frame = cv2.imread(image_path)
        if static_frame is None:
            print(f"Failed to load: {image_path}")
            return
        static_frame = cv2.resize(static_frame, (FRAME_W, FRAME_H))
        print(f"Loaded: {image_path}")
        print(f"  Image size: {FRAME_W}x{FRAME_H}")
    else:
        cap = cv2.VideoCapture(DEV_INDEX, cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {DEV_INDEX} not available")

    # 创建窗口
    cv2.namedWindow(WIN_ORIGINAL)
    cv2.namedWindow(WIN_MASK)

    # 铁色模式初始化
    create_trackbars(WIN_MASK, IRON_H_LOW, IRON_H_HIGH, IRON_S_LOW, IRON_S_HIGH,
                     IRON_V_LOW, IRON_V_HIGH)

    # 保存两组阈值
    iron_thresh = {"h_low": IRON_H_LOW, "h_high": IRON_H_HIGH, "s_low": IRON_S_LOW,
                   "s_high": IRON_S_HIGH, "v_low": IRON_V_LOW, "v_high": IRON_V_HIGH}
    red_thresh = {"h_low": RED_H_LOW, "h_high": RED_H_HIGH, "s_low": RED_S_LOW,
                  "s_high": RED_S_HIGH, "v_low": RED_V_LOW, "v_high": RED_V_HIGH}

    # 鼠标回调
    current_frame = [None]  # 用列表包装，方便回调修改
    cv2.setMouseCallback(WIN_ORIGINAL, on_mouse, current_frame)

    print("\n" + "=" * 60)
    print("HSV Threshold Tuner for TicTacToe Pieces")
    print("=" * 60)
    print("  Click on pixel  → print HSV value")
    print("  1               → switch to IRON (iron color) tuning")
    print("  2               → switch to RED tuning")
    print("  s               → print current config code")
    print("  L               → load an image file")
    print("  q               → quit")
    print()

    while True:
        # 获取画面
        if static_frame is not None:
            frame = static_frame.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        current_frame[0] = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 读当前滑条
        hl, hh, sl, sh, vl, vh = get_trackbar_values(WIN_MASK)

        # 保存到当前模式
        cur = iron_thresh if current_mode == MODE_IRON else red_thresh
        cur["h_low"], cur["h_high"] = hl, hh
        cur["s_low"], cur["s_high"] = sl, sh
        cur["v_low"], cur["v_high"] = vl, vh

        # 生成 mask
        mask = hsv_mask(hsv, hl, hh, sl, sh, vl, vh)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 在 mask 上画找到的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(mask_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(mask_bgr, f"{area:.0f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 叠加标注
        mode_name = "IRON (dark)" if current_mode == MODE_IRON else "RED"
        color = (200, 200, 200) if current_mode == MODE_IRON else (0, 0, 255)

        cv2.putText(frame, f"Mode: {mode_name}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(mask_bgr, f"{mode_name} mask", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        display = np.hstack([frame, mask_bgr])
        cv2.imshow(WIN_ORIGINAL, display)

        # WIN_MASK 窗口只放 mask 便于调滑条
        cv2.imshow(WIN_MASK, mask_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("1"):
            current_mode = MODE_IRON
            t = iron_thresh
            set_trackbar_values(WIN_MASK, t["h_low"], t["h_high"], t["s_low"],
                                t["s_high"], t["v_low"], t["v_high"])
            print("\n>>> Switched to IRON mode <<<")

        elif key == ord("2"):
            current_mode = MODE_RED
            t = red_thresh
            set_trackbar_values(WIN_MASK, t["h_low"], t["h_high"], t["s_low"],
                                t["s_high"], t["v_low"], t["v_high"])
            print("\n>>> Switched to RED mode <<<")

        elif key == ord("s"):
            print("\n" + "=" * 60)
            print("Current Thresholds — copy to config.py")
            print("=" * 60)
            print(f"# Iron (dark piece)")
            print(f"CLASSIFIER_IRON_H_LOW  = {iron_thresh['h_low']}")
            print(f"CLASSIFIER_IRON_H_HIGH = {iron_thresh['h_high']}")
            print(f"CLASSIFIER_IRON_S_LOW  = {iron_thresh['s_low']}")
            print(f"CLASSIFIER_IRON_S_HIGH = {iron_thresh['s_high']}")
            print(f"CLASSIFIER_IRON_V_LOW  = {iron_thresh['v_low']}")
            print(f"CLASSIFIER_IRON_V_HIGH = {iron_thresh['v_high']}")
            print(f"\n# Red (red piece)")
            print(f"CLASSIFIER_RED_H_LOW   = {red_thresh['h_low']}")
            print(f"CLASSIFIER_RED_H_HIGH  = {red_thresh['h_high']}")
            print(f"CLASSIFIER_RED_S_LOW   = {red_thresh['s_low']}")
            print(f"CLASSIFIER_RED_S_HIGH  = {red_thresh['s_high']}")
            print(f"CLASSIFIER_RED_V_LOW   = {red_thresh['v_low']}")
            print(f"CLASSIFIER_RED_V_HIGH  = {red_thresh['v_high']}")
            print("=" * 60)

        elif key == ord("l") and static_frame is None:
            # 加载图片
            path = input("Enter image path: ").strip().strip('"')
            if path and os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    static_frame = cv2.resize(img, (FRAME_W, FRAME_H))
                    print(f"Loaded: {path}")
                else:
                    print(f"Failed to load: {path}")
            else:
                print("File not found")

    # 清理
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
