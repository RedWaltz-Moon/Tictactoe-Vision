"""
全局配置 — 修改这里即可调整所有参数，不必改业务代码。
"""

# ============================================================
# 串口
# ============================================================
SERIAL_PORT = "/dev/ttyUSB0"   # 香橙派串口设备
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0          # 读超时(秒)

# 帧协议
FRAME_HEAD = 0xAA
FRAME_TAIL = 0xBB

# ============================================================
# 棋盘物理尺寸 (mm) — 待实测后填入
# ============================================================
BOARD_WIDTH_MM = 90.0
BOARD_HEIGHT_MM = 90.0

CELL_SIZE_MM = BOARD_WIDTH_MM / 3.0

# ============================================================
# YOLO
# ============================================================
YOLO_MODEL_PATH = "models/chessboard_v8n.pt"   # PC端
YOLO_CONF_THRESHOLD = 0.5
YOLO_MISSING_FRAME_LIMIT = 3                     # 连续N帧检测失败→报错

# ============================================================
# 推理后端
#   "ultralytics" → PC 开发 (PyTorch + ultralytics)
#   "ascend"      → 香橙派部署 (ACL + OM 模型)
# ============================================================
BACKEND = "ascend"                             # 当前后端
ASCEND_MODEL_PATH = "models/chessboard_v8n.om"  # ATC 编译产物
ASCEND_INPUT_SIZE = 640                         # YOLO 输入分辨率

# ============================================================
# 棋子识别 (HSV)
#   棋盘: 纸白底黑线 (白底高V低S)
#   棋子: 铁色棋子 + 红色棋子 (阈值由 tools/tune_hsv.py 实测调定)
# ============================================================
CLASSIFIER_SAMPLE_RATIO = 0.6              # 铁色采样：取格子中心60%区域
CLASSIFIER_RED_SAMPLE_RATIO = 0.35          # 红色采样：取格子中心35%（避开边缘红色底板）
CLASSIFIER_EMPTY_V_HIGH = 180              # V > 此值 + S<40 → 确定空白(白纸)
CLASSIFIER_EMPTY_V_MID = 150               # V > 此值 + 低圆度 → 倾向空白
CLASSIFIER_EMPTY_S_MAX = 40                # S < 此值 → 白纸(低饱和度)

# 铁色棋子 HSV 范围 (暗色金属，H 放宽因光照变化大)
CLASSIFIER_IRON_H_LOW  = 80
CLASSIFIER_IRON_H_HIGH = 150
CLASSIFIER_IRON_S_LOW  = 110
CLASSIFIER_IRON_S_HIGH = 255
CLASSIFIER_IRON_V_LOW  = 80
CLASSIFIER_IRON_V_HIGH = 185

# 红色棋子 HSV 范围 (H 跨边界: 160→180 + 0→77)
CLASSIFIER_RED_H_LOW   = 160
CLASSIFIER_RED_H_HIGH  = 77
CLASSIFIER_RED_S_LOW   = 37
CLASSIFIER_RED_S_HIGH  = 135
CLASSIFIER_RED_V_LOW   = 75
CLASSIFIER_RED_V_HIGH  = 215

CLASSIFIER_CIRCULARITY_MIN = 0.7           # 圆形度下限（过滤非圆干扰）
CLASSIFIER_MASK_RATIO_MIN = 0.20           # 全格 mask 占比下限（≥此值才判定为棋子）
CLASSIFIER_VALIDATE_RETRY = 3              # 逻辑校验失败重试次数

# ============================================================
# 棋盘裁剪（矩形框裁剪，非透视变换）
# ============================================================
DETECT_THRESH_METHOD = 1          # 0=Otsu, 1=Adaptive, 2=Manual
DETECT_MANUAL_THRESH = 127        # Manual 模式下的阈值
DETECT_MORPH_KERNEL = 3           # 形态学核大小
DETECT_MORPH_CLOSE_ITER = 2       # 闭操作迭代次数
DETECT_BLUR_SIZE = 11             # 高斯模糊核大小（须奇数）
DETECT_MIN_AREA_RATIO = 0.17      # 最小轮廓面积占全图比例
CROP_MARGIN_PX = 38               # 裁切框外围边距（像素）

# ============================================================
# 旋转校正：固定角度（逆时针为正）
#   摄像头固定安装，棋盘始终偏转相同角度，无需自动检测
# ============================================================
BOARD_ROTATION_DEG = -45            # 逆时针旋转角度（负数=顺时针校正）
ROTATION_ANGLE_THRESH = 20         # minAreaRect 角度 > 此值 → 判定为旋转棋盘

# ============================================================
# 鲁棒性
# ============================================================
CORNER_SHIFT_THRESHOLD_PX = 50    # 角点偏移超此值→丢弃帧
BOARD_AREA_CHANGE_RATIO = 0.2     # 棋盘面积变化>20%→重新定位
CORNER_SMOOTH_WINDOW = 3          # 角点滑动平均窗口

# ============================================================
# AI
# ============================================================
AI_DIFFICULTY = 0.75         # 0.0=纯随机 0.5=偶尔失误 1.0=永不输

# ============================================================
# 棋子库存
# ============================================================
IRON_COUNT = 4   # 铁色棋子总数
RED_COUNT = 3    # 红色棋子总数

# ============================================================
# 系统
# ============================================================
CAMERA_INDEX = 0
MECHANICAL_DELAY_S = 0.5     # 机械臂放子后等待稳定时间
GAME_TIMEOUT_S = 30.0        # 等待玩家落子超时
