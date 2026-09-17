# TicTacToe Vision

一个面向 3×3 井字棋机器人的视觉与决策系统：USB 摄像头采集画面，YOLO/OpenCV 定位棋盘并识别棋子，Minimax 计算落子，再通过串口把结果发送给 MCU。

## 功能

- YOLOv8 棋盘检测，传统 OpenCV 轮廓检测作为回退方案
- 3×3 棋盘切分与 HSV 棋子识别
- Minimax + Alpha-Beta 剪枝决策
- 像素坐标到棋盘物理坐标的标定与转换
- `0xAA ... 0xBB` 串口帧协议
- PC Mock 调试和昇腾 310B 部署支持

详细设计与通信时序见 [技术规格书](docs/TECHNICAL_SPEC.md)，昇腾部署步骤见 [DEPLOY.md](DEPLOY.md)。

## 项目结构

```text
tictactoe_vision/
├── main.py                 # 主程序与状态机
├── config.py               # 串口、模型、相机及识别参数
├── detector.py             # 棋盘定位与裁切
├── classifier.py           # 棋子分类与状态校验
├── ai_engine.py            # Minimax 决策
├── calibrator.py           # 坐标标定
├── serial_protocol.py      # 串口协议编解码
├── ascend_inference.py     # 昇腾 ACL 推理封装
├── models/                 # 已训练的 PT/ONNX 模型
├── data/                   # 图片、标注及数据集配置
├── tools/                  # 采集、标注、训练、导出和调参工具
└── docs/                   # 技术文档
```

`runs/` 是 Ultralytics 训练输出，不是程序运行所必需；仓库默认忽略后续生成的训练输出。当前目录中的历史 `runs/` 含 `last.pt` 检查点，确认不再需要续训后可以手动删除。

## 环境要求

- Python 3.10 或更高版本
- USB 摄像头
- PC 开发：PyTorch/Ultralytics 环境
- 实机部署：Orange Pi AI Pro（Ascend 310B）及已安装的 CANN/ACL

建议使用虚拟环境：

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

PC 开发环境：

```bash
python -m pip install -U pip
pip install -r requirements-dev.txt
```

昇腾设备运行环境：

```bash
pip install -r requirements.txt
```

> `requirements.txt` 不安装 CANN/ACL；这部分需按设备环境单独配置。

## 配置

运行前检查 [config.py](config.py)：

- PC 调试将 `BACKEND` 改为 `"ultralytics"`
- 昇腾部署使用 `BACKEND = "ascend"`，并准备 `models/chessboard_v8n.om`
- 按设备修改 `SERIAL_PORT`、`CAMERA_INDEX` 和棋子 HSV 阈值
- 默认 PC 模型为 `models/chessboard_v8n.pt`

## 运行

```bash
# 连接真实串口和摄像头
python main.py

# PC 调试，键盘模拟 MCU 指令
python main.py --mock

# 仅运行标定流程
python main.py --calibrate
```

Mock 模式按键：

| 按键 | 动作 |
|---|---|
| `N` | 开始新对局，AI 先手 |
| `H` | 开始新对局，人先手 |
| `R` | 玩家落子完成，触发检测 |
| `X` | 复位 |
| `Q` | 退出 |

## 串口协议摘要

帧格式为 `0xAA + CMD + DATA + 0xBB`。

| CMD | 方向 | 含义 |
|---|---|---|
| `0x01` PLAYER_DONE | MCU → 视觉 | 玩家落子完成 |
| `0x02` GAME_START | MCU → 视觉 | 新对局并复位状态 |
| `0x03` RESET | MCU → 视觉 | 软复位 |
| `0x05` WHO_FIRST | MCU → 视觉 | `0x00` AI 先手，`0x01` 人先手 |
| `0x10` AI_MOVE | 视觉 → MCU | 落子索引和物理坐标 |
| `0x13` THINKING | 视觉 → MCU | 正在处理 |
| `0x14` GAME_OVER | 视觉 → MCU | 对局结果 |

## 数据与训练

开发工具的完整用法见 [tools/README.md](tools/README.md)。常用流程：

```bash
python tools/capture.py
python tools/labelme_to_yolo.py
python tools/verify_labels.py
python tools/train.py all
python tools/export_onnx.py
```

训练完成后，`tools/train.py` 会把最优权重复制到 `models/chessboard_v8n.pt`。

## 快速自检

无需摄像头即可检查纯逻辑模块：

```bash
python ai_engine.py
python serial_protocol.py
python calibrator.py
python -m compileall -q .
```

视觉和整机流程还需要摄像头、模型及对应推理后端。

## 说明

- 模型权重和训练数据会显著增加仓库体积。
- 本项目采用 [MIT License](LICENSE)，版权所有 © 2026 RedWaltz。
