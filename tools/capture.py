"""
数据采集脚本：拍棋盘照片存到 data/images/。
每按一次空格拍一张，ESC 退出，支持断点续采。

棋盘摆放要求覆盖:
  - 0°~45° 旋转角度
  - 不同棋子状态（空盘、开局、中盘、残局、满盘）
  - 不同光照条件（亮/暗/侧光）
  - 目标: 100+ 张
"""

import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)  # 让 tools/ 内脚本能 import 项目根模块
os.chdir(_PROJ_ROOT)            # 统一工作目录为项目根，data/ 路径正确

import cv2
import re
import glob
from datetime import datetime

SAVE_DIR = "data/images"
CAMERA_INDEX = 0


def find_resume_count(save_dir: str) -> int:
    """
    扫描已有图片，解析文件名中的 4 位序号，返回下一张起始编号。
    文件名格式: board_YYYYMMDD_HHMMSS_XXXX.jpg
    """
    pattern = os.path.join(save_dir, "board_*_*.jpg")
    existing = glob.glob(pattern)
    if not existing:
        return 0

    max_idx = -1
    for path in existing:
        name = os.path.basename(path)
        # 匹配末尾 _XXXX.jpg
        m = re.search(r"_(\d{4})\.jpg$", name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx

    return max_idx + 1


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 断点续采：从已有最大序号 +1 开始
    count = find_resume_count(SAVE_DIR)
    prev_files = len(glob.glob(os.path.join(SAVE_DIR, "board_*.jpg")))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Camera {CAMERA_INDEX} not available")
        return

    print("Data Capture Tool")
    print(f"  Existing images: {prev_files}")
    print(f"  Next index:      {count}")
    print("  SPACE = capture  |  ESC = quit  |  D = delete last")
    print(f"  Saving to: {SAVE_DIR}/")
    print()

    last_saved: str | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        # 居中画十字参考线
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
        cv2.putText(frame, f"Total: {prev_files + (count - find_resume_count(SAVE_DIR))} | Next: {count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=Capture  D=Del Last  ESC=Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"board_{ts}_{count:04d}.jpg"
            path = os.path.join(SAVE_DIR, name)
            cv2.imwrite(path, frame)
            print(f"[{count}] Saved: {name}")
            last_saved = path
            count += 1

        elif key == ord("d"):  # 删除最后一张
            if last_saved and os.path.exists(last_saved):
                os.remove(last_saved)
                print(f"[{count - 1}] Deleted: {os.path.basename(last_saved)}")
                count -= 1
                last_saved = None

    cap.release()
    cv2.destroyAllWindows()

    total = len(glob.glob(os.path.join(SAVE_DIR, "board_*.jpg")))
    session_count = count - find_resume_count(SAVE_DIR) if count > 0 else 0
    print(f"\nDone. This session: {session_count}  |  Total: {total}")


if __name__ == "__main__":
    main()
