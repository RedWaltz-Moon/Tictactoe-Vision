from __future__ import annotations

"""
串口协议编解码 — 0xAA 帧头 + CMD + DATA + 0xBB 帧尾。
纯逻辑模块，不依赖实际串口硬件。

指令集:
  MCU → 视觉 (交互):
    0x01 PLAYER_DONE  玩家落子完毕，触发检测
    0x02 GAME_START   新游戏开局（复位棋盘，等待 WHO_FIRST）
    0x05 WHO_FIRST    指定先手，DATA=0x00(AI先手) / 0x01(人先手)

  MCU → 视觉 (内部):
    0x03 RESET        软复位

  视觉 → MCU (交互):
    0x10 AI_MOVE      AI落子：idx(1B) X(2B大端) Y(2B大端)
    0x14 GAME_OVER    对局结束：result(1B)

  视觉 → MCU (内部):
    0x13 THINKING     处理中，请等待
"""

FRAME_HEAD = 0xAA
FRAME_TAIL = 0xBB

# ============================================================
# 命令字
# ============================================================

# MCU → 视觉
CMD_PLAYER_DONE = 0x01
CMD_GAME_START  = 0x02
CMD_RESET       = 0x03
CMD_WHO_FIRST   = 0x05

# 视觉 → MCU
CMD_AI_MOVE   = 0x10
CMD_THINKING  = 0x13
CMD_GAME_OVER = 0x14

# ============================================================
# 参数常量
# ============================================================

# GAME_OVER 结果码
RESULT_AI_WIN    = 0
RESULT_HUMAN_WIN = 1
RESULT_DRAW      = 2

# WHO_FIRST 先手参数
AI_FIRST    = 0x00
HUMAN_FIRST = 0x01


# ============================================================
# 打包 (编码)
# ============================================================

def pack_ai_move(index: int, x_mm: float, y_mm: float) -> bytes:
    """
    AI_MOVE 帧: 0xAA 0x10 idx(1B) X_H(1B) X_L(1B) Y_H(1B) Y_L(1B) 0xBB
    X/Y 为 uint16 大端，单位 mm，四舍五入取整。
    """
    x_int = round(x_mm)
    y_int = round(y_mm)
    x_hi, x_lo = (x_int >> 8) & 0xFF, x_int & 0xFF
    y_hi, y_lo = (y_int >> 8) & 0xFF, y_int & 0xFF
    return bytes([FRAME_HEAD, CMD_AI_MOVE, index, x_hi, x_lo, y_hi, y_lo, FRAME_TAIL])


def pack_thinking() -> bytes:
    """THINKING 帧: 0xAA 0x13 0xBB"""
    return bytes([FRAME_HEAD, CMD_THINKING, FRAME_TAIL])


def pack_game_over(result: int) -> bytes:
    """GAME_OVER 帧: 0xAA 0x14 result(1B) 0xBB"""
    return bytes([FRAME_HEAD, CMD_GAME_OVER, result, FRAME_TAIL])


# ============================================================
# 解包 (解码)
# ============================================================

def _validate_frame(data: bytes) -> bool:
    return (
        len(data) >= 3
        and data[0] == FRAME_HEAD
        and data[-1] == FRAME_TAIL
    )


def unpack(data: bytes) -> dict | None:
    """
    解包一帧，返回 dict: {"cmd": int, "data": bytes}，或 None (无效帧)。
    调用方需自行根据 cmd 解释 data 内容。
    """
    if not _validate_frame(data):
        return None
    return {
        "cmd": data[1],
        "data": data[2:-1],
    }


def unpack_who_first(data: bytes) -> int:
    """解包 WHO_FIRST 指令的 data 字节，返回 AI_FIRST 或 HUMAN_FIRST。"""
    return data[0] if len(data) >= 1 else AI_FIRST


# ============================================================
# 帧拼接/分割 (串口流式数据)
# ============================================================

class FrameBuffer:
    """
    串口流式数据缓冲，从连续字节流中提取完整帧。
    帧格式: 0xAA ... 0xBB，帧内不含转义。

    用法:
        buf = FrameBuffer()
        buf.feed(some_bytes)       # 喂入新收到的字节
        while frame := buf.pop():  # 取出所有完整帧
            handle(frame)
    """

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        self._buffer.extend(data)

    def pop(self) -> bytes | None:
        """返回一个完整帧 (含头尾) 或 None。"""
        while True:
            # 找帧头
            head_idx = self._buffer.find(FRAME_HEAD)
            if head_idx == -1:
                self._buffer.clear()
                return None

            # 丢弃帧头前的垃圾字节
            if head_idx > 0:
                del self._buffer[:head_idx]

            # 找帧尾
            tail_idx = self._buffer.find(FRAME_TAIL, 1)
            if tail_idx == -1:
                return None  # 帧不完整，等待更多数据

            # 提取完整帧
            frame = bytes(self._buffer[: tail_idx + 1])
            del self._buffer[: tail_idx + 1]

            # 帧长度至少3字节 (头+cmd+尾)
            if len(frame) >= 3:
                return frame
            # 否则丢弃，继续找下一帧


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    # 测试 AI_MOVE 打包
    move = pack_ai_move(4, 75.3, 75.8)
    print(f"pack_ai_move(4, 75.3, 75.8) = {move.hex(' ')}")
    # 解析
    parsed = unpack(move)
    idx = parsed["data"][0]
    x = (parsed["data"][1] << 8) | parsed["data"][2]
    y = (parsed["data"][3] << 8) | parsed["data"][4]
    print(f"  -> index={idx}, x={x}mm, y={y}mm")
    assert idx == 4 and x == 75 and y == 76, "AI_MOVE pack/unpack mismatch"

    # 测试 GAME_OVER
    go = pack_game_over(RESULT_AI_WIN)
    print(f"pack_game_over(AI_WIN) = {go.hex(' ')}")
    assert go == bytes([0xAA, CMD_GAME_OVER, RESULT_AI_WIN, 0xBB])

    # 测试 THINKING (无数据)
    thinking = pack_thinking()
    print(f"pack_thinking() = {thinking.hex(' ')}")
    assert len(thinking) == 3
    assert thinking == bytes([0xAA, CMD_THINKING, 0xBB])

    # 测试 FrameBuffer
    buf = FrameBuffer()
    buf.feed(bytes([0xAA, CMD_THINKING]))
    assert buf.pop() is None
    buf.feed(bytes([0xBB]))
    assert buf.pop() == bytes([0xAA, CMD_THINKING, 0xBB])

    # 喂入多帧+垃圾
    buf.feed(bytes([0xFF, 0xAA, CMD_THINKING, 0xBB, 0xAA, CMD_GAME_OVER, 0x00, 0xBB, 0x00]))
    f1 = buf.pop()
    f2 = buf.pop()
    assert f1 == bytes([0xAA, CMD_THINKING, 0xBB])
    assert f2 == bytes([0xAA, CMD_GAME_OVER, 0x00, 0xBB])
    assert buf.pop() is None

    # 验证常量
    assert CMD_PLAYER_DONE == 0x01
    assert CMD_GAME_START == 0x02
    assert CMD_RESET == 0x03
    assert CMD_WHO_FIRST == 0x05
    assert CMD_AI_MOVE == 0x10
    assert CMD_THINKING == 0x13
    assert CMD_GAME_OVER == 0x14
    assert AI_FIRST == 0x00
    assert HUMAN_FIRST == 0x01
    assert RESULT_AI_WIN == 0
    assert RESULT_HUMAN_WIN == 1
    assert RESULT_DRAW == 2

    print("serial_protocol self-test passed")
