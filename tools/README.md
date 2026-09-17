# Tools — TicTacToe Vision 开发辅助工具

本目录包含棋盘视觉系统的调参、采集、标注、训练和验证工具。所有脚本从 `tools/` 目录运行时会自动将工作目录切换到项目根 (`tictactoe_vision/`)，因此路径参数和输出文件始终相对于项目根。

## 工具一览

| 阶段 | 工具 | 用途 |
|---|---|---|
| 调参 | `tune_crop.py` | 棋盘裁切参数可视化调节 |
| 调参 | `tune_hsv.py` | 铁色/红色棋子 HSV 阈值调节 |
| 验证 | `check_crop.py` | 验证裁切效果（输入→输出尺寸） |
| 采集 | `capture.py` | 采集棋盘照片数据集 |
| 标注 | `labelme_to_yolo.py` | labelme JSON → YOLO txt 转换 |
| 标注 | `verify_labels.py` | 在图片上画标注框可视化校验 |
| 训练 | `train.py` | 划分数据集 + 训练 YOLOv8n |

---

### `tune_crop.py` — 棋盘裁切调参

实时调节棋盘 bounding-rect 裁切参数，不改变画面比例，仅丢弃无关背景。

```
python tools/tune_crop.py              # 摄像头实时
python tools/tune_crop.py path/to/image.jpg     # 静态图片
```

**可调参数：**
- `ThreshMethod` — 0=Otsu, 1=Adaptive, 2=Manual 二值化
- `ManualThresh` — 手动阈值（Method=2 时生效）
- `MorphKernel / CloseIter` — 形态学闭操作参数
- `BlurSize` — 高斯模糊核大小
- `Margin` — 裁切框外扩像素
- `MinArea/100` — 最小轮廓面积比例

**操作：** 滑条实时调节 → 按 `s` 打印 config 代码 → 按 `q` 退出

---

### `tune_hsv.py` — HSV 阈值调参

测定铁色（暗灰）和红色棋子在 HSV 色彩空间中的阈值区间，用于 `classifier.py` 的棋子识别。

```
python tools/tune_hsv.py               # 摄像头实时
python tools/tune_hsv.py path/to/image.jpg      # 静态图片
```

**操作：**
- `1` / `2` — 切换铁色 / 红色模式
- 鼠标点击 — 终端打印该像素 HSV 值（采样参考）
- 滑条拖拽 — 实时显示 mask 效果
- `s` — 打印当前两组阈值的 config 代码
- `q` — 退出

**说明：** 铁色棋子低 V（暗）、低 S（灰）；红色棋子 H 在 0-10 或 160-180，中高 S/V。红色跨 H=0 边界时滑条自动分段处理。

---

### `check_crop.py` — 裁切验证

对单张图片运行 `detector.crop()`，输出裁切前后的尺寸、rect 坐标、画面缩减比例。

```
python tools/check_crop.py path/to/image.jpg
```

输出示例：
```
Input:    640x480
Cropped:  445x384
Rect:     (59, 10, 504, 394)
Reduced:  44%
Saved:    test_cropped.jpg
```

---

### `capture.py` — 数据采集

拍摄棋盘照片，存入 `data/images/`。支持断点续采（自动扫描已有文件从最大序号+1 开始）。

```
python tools/capture.py
```

**操作：** `SPACE` 拍照 | `D` 删除上一张 | `ESC` 退出

**采集要求：** 覆盖不同旋转角度、棋子状态、光照条件，目标 100+ 张。

---

### `labelme_to_yolo.py` — 标注格式转换

将 labelme 标注的 JSON 文件（Rectangle/Polygon，标签 `chessboard`）转换为 YOLO 标准检测格式 txt。

```
python tools/labelme_to_yolo.py            # 转换 data/labels/ 下所有 JSON
python tools/labelme_to_yolo.py --show     # 转换后自动可视化校验
```

转换公式：YOLO 格式 = `class x_center y_center width height`（归一化坐标）。

---

### `verify_labels.py` — 标注可视化

在图片上叠加 YOLO bbox 标签，逐张浏览校验标注质量。

```
python tools/verify_labels.py
```

**操作：** `N`/`Space` 下一张 | `P` 上一张 | `Q`/`ESC` 退出

---

### `train.py` — YOLO 训练

划分 train/val 数据集并训练 YOLOv8n 棋盘检测模型。

```
python tools/train.py prepare    # 划分数据集 (85% train / 15% val)
python tools/train.py train      # 训练 YOLOv8n
python tools/train.py all        # 准备 + 训练
```

训练完成后最优权重自动拷贝到 `models/chessboard_v8n.pt`。

---

## 目录结构

```
tictactoe_vision/
├── main.py              # 主程序
├── config.py            # 全局配置
├── detector.py          # 棋盘检测
├── classifier.py        # 棋子识别
├── ai_engine.py         # AI（Minimax）
├── calibrator.py        # 相机标定
├── serial_protocol.py   # 串口协议
├── data/
│   ├── images/          # 采集的棋盘照片
│   ├── labels/          # YOLO 标注 + labelme JSON
│   └── dataset.yaml     # YOLO 训练配置
├── models/              # 训练好的模型权重
└── tools/               # 本目录 — 开发辅助工具
    ├── README.md
    ├── tune_crop.py
    ├── tune_hsv.py
    ├── check_crop.py
    ├── capture.py
    ├── labelme_to_yolo.py
    ├── verify_labels.py
    └── train.py
```
