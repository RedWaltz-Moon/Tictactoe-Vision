# Orange Pi AI Pro (Ascend 310B) 部署指南

## 0. 硬件清单

| 项目 | 说明 |
|------|------|
| 开发板 | 华为昇腾香橙派AI Pro (8TOPS, Ascend 310B4) |
| 系统 | openEuler / Ubuntu (ARM64) |
| 摄像头 | USB 摄像头 (固定俯拍) |
| 串口 | `/dev/ttyUSB0` → MCU (115200 8N1) |
| 棋盘 | 红色PLA底板 + 白色纸面 (90mm×90mm) |
| 棋子 | 铁色(M4×4) + 红色(M3×3) 圆形金属块 |

---

## 1. 前置环境检查

在香橙派上依次执行：

```bash
# 确认 CANN 已安装
ls /usr/local/Ascend/ascend-toolkit/latest/
# 或
ls ~/Ascend/ascend-toolkit/latest/

# 确认 Python
python3 --version   # ≥ 3.9

# 确认摄像头
ls /dev/video*
# 如果没有，先插 USB 摄像头再查看

# 确认串口
ls -l /dev/ttyUSB*
# 如果没有，检查 USB 转串口是否插好
# 确认当前用户有串口权限: groups | grep dialout
```

---

## 2. 安装 Python 依赖

```bash
cd ~/tictactoe_vision
pip install -r requirements.txt
# 只需要 opencv-python, numpy, pyserial
```

---

## 3. 模型转换 (PT → ONNX → OM)

### 3a. PC端：导出 ONNX

```bash
# 在 PC 上
cd tictactoe_vision
python tools/export_onnx.py
# → 生成 models/chessboard_v8n.onnx
```

### 3b. PC端：验证 ONNX 输出

```bash
pip install onnxruntime
python tools/test_onnx.py data/images/test_board.jpg
# 应输出 "Output shape OK: (1, 84, 8400)"
```

### 3c. 传输 ONNX 到香橙派

```bash
# 如果整个项目还没传上来，直接传整个文件夹
# 在 PC 上:
scp -r tictactoe_vision/ pi@<orangepi-ip>:~/

# 如果只传 ONNX:
scp models/chessboard_v8n.onnx pi@<orangepi-ip>:~/tictactoe_vision/models/
```

### 3d. 香橙派端：ATC 编译

```bash
# 先 source CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 或
source ~/Ascend/ascend-toolkit/set_env.sh

cd ~/tictactoe_vision

# 编译
atc --model=models/chessboard_v8n.onnx \
    --framework=5 \
    --output=models/chessboard_v8n \
    --soc_version=Ascend310B4 \
    --input_shape="images:1,3,640,640" \
    --input_format=NCHW

# 检查产物
ls -lh models/chessboard_v8n.om
# 应该有 ~6-10MB
```

> **注意：** `--framework=5` 表示 ONNX。如果 ATC 报算子不支持，尝试在 `export_onnx.py` 中降低 `opset` 到 11。

---

## 4. 配置

编辑 `config.py` 确认以下配置：

```python
# 推理后端
BACKEND = "ascend"

# 串口
SERIAL_PORT = "/dev/ttyUSB0"

# 相机 (通常为 0，如果有多个摄像头可能需要改成 1 或 2)
CAMERA_INDEX = 0

# AI
AI_DIFFICULTY = 1.0
```

---

## 5. 测试

### 5a. 独立测试 NPU 推理

```bash
cd ~/tictactoe_vision
python ascend_inference.py data/images/test_board.jpg models/chessboard_v8n.om
# 应输出 "Detected 1 board(s)"
```

### 5b. 串口环回测试 (不接 MCU)

```bash
# 用 mock 模式测试逻辑
python main.py --mock
# 按键: n=AI先手 h=人先手 r=READY q=退出
```

### 5c. 实机测试 (接 MCU + 摄像头)

```bash
python main.py
```

---

## 6. 开机自启 (可选)

创建 systemd 服务文件 `/etc/systemd/system/tictactoe.service`：

```ini
[Unit]
Description=TicTacToe Vision System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/tictactoe_vision
ExecStartPre=/bin/bash -c 'source ~/Ascend/ascend-toolkit/set_env.sh'
ExecStart=/usr/bin/python3 /home/pi/tictactoe_vision/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tictactoe
sudo systemctl start tictactoe
sudo systemctl status tictactoe
```

---

## 7. 常见问题

| 问题 | 解决 |
|------|------|
| `import acl` 失败 | `source` CANN 的 `set_env.sh`；检查 `PYTHONPATH` 是否包含 `ascend-toolkit/latest/python/site-packages` |
| `acl.rt.set_device` 失败 | 确认 NPU 未被其他进程占用；`npu-smi info` 查看状态 |
| 摄像头打不开 | `ls /dev/video*`；检查用户是否在 `video` 组 |
| 串口无响应 | `dmesg \| grep tty` 确认设备名；`sudo chmod 666 /dev/ttyUSB0` |
| ATC 编译报错 | 检查 ONNX 的 `opset` 版本；尝试 `--log=debug` 查看具体不支持的算子 |
| 推理结果为 None | 检查光照条件；确认棋盘在摄像头视野内；降低 `YOLO_CONF_THRESHOLD` |
