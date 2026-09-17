from __future__ import annotations

"""
主程序入口 — 串口监听 + 状态机 + 全部模块串联。

用法:
    python main.py                      # 正常启动
    python main.py --mock              # Mock 模式 (键盘模拟 MCU，PC 调试用)

Mock 模式按键:
    n = GAME_START + WHO_FIRST(AI)
    h = GAME_START + WHO_FIRST(人)
    r = PLAYER_DONE
    x = RESET
    q = 退出
"""

import sys, os
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ_ROOT)
os.chdir(_PROJ_ROOT)

import time
import os
from enum import Enum, auto

import cv2
import numpy as np

from config import (
    SERIAL_PORT,
    SERIAL_BAUDRATE,
    SERIAL_TIMEOUT,
    CAMERA_INDEX,
    BOARD_WIDTH_MM,
    BOARD_HEIGHT_MM,
    AI_DIFFICULTY,
    MECHANICAL_DELAY_S,
    GAME_TIMEOUT_S,
    IRON_COUNT,
    RED_COUNT,
)
from ai_engine import ai_move, check_winner, is_draw, game_over, EMPTY, IRON, RED
from serial_protocol import (
    CMD_PLAYER_DONE,
    CMD_GAME_START,
    CMD_RESET,
    CMD_WHO_FIRST,
    AI_FIRST,
    HUMAN_FIRST,
    RESULT_AI_WIN,
    RESULT_HUMAN_WIN,
    RESULT_DRAW,
    FrameBuffer,
    pack_ai_move,
    pack_thinking,
    pack_game_over,
    unpack,
    unpack_who_first,
)
from detector import BoardDetector
from classifier import CellClassifier
from calibrator import (
    compute_homography,
    pixel_to_physical,
    cell_center_physical,
    save_calibration,
    load_calibration,
)


# ================================================================
# 状态机
# ================================================================

class State(Enum):
    IDLE = auto()           # 等待 GAME_START + WHO_FIRST
    AI_THINK = auto()       # AI 计算 + 发送 AI_MOVE
    WAIT_PLAYER = auto()    # 等待 PLAYER_DONE
    CALIBRATING = auto()    # 标定模式


class TicTacToeSystem:
    """系统顶层控制器。"""

    def __init__(self, mock_serial: bool = False):
        self.mock = mock_serial
        self.state = State.IDLE
        self.ser = None
        self.fb = FrameBuffer()

        # 游戏状态
        self.board = [EMPTY] * 9
        self.ai_piece = IRON
        self.human_piece = RED
        self.ai_first = True
        self.turn = IRON  # 铁色先手
        self.iron_left = IRON_COUNT
        self.red_left = RED_COUNT
        self._first_move = True  # AI先手开局第一步跳过拍照

        # 视觉模块
        self.detector = BoardDetector()
        self.classifier = CellClassifier()

        # 标定
        self.calib = None

        # 相机
        self.cap = None

    # ================================================================
    # 入口
    # ================================================================

    def run(self) -> None:
        print("TicTacToe Vision System starting...")
        self._init_camera()
        self._open_serial()
        self._load_calibration()

        try:
            while True:
                self._read_serial()

                if self.state == State.IDLE:
                    pass  # 等待 GAME_START + WHO_FIRST
                elif self.state == State.AI_THINK:
                    self._do_ai_think()
                elif self.state == State.WAIT_PLAYER:
                    pass  # 等待 PLAYER_DONE
                elif self.state == State.CALIBRATING:
                    self._do_calibrate()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self._cleanup()

    # ================================================================
    # 状态处理
    # ================================================================

    def _do_ai_think(self) -> None:
        """检测棋盘 → AI 计算落子 → 发送 AI_MOVE。"""
        time.sleep(0.05)  # 避免紧循环

        # 棋子耗尽检查
        if self.ai_piece == IRON and self.iron_left <= 0:
            print("Error: iron pieces exhausted")
            self._send(pack_game_over(RESULT_DRAW))
            self.state = State.IDLE
            return
        if self.ai_piece == RED and self.red_left <= 0:
            print("Error: red pieces exhausted")
            self._send(pack_game_over(RESULT_DRAW))
            self.state = State.IDLE
            return

        # 拍照检测当前棋盘（AI先手空盘第一步跳过）
        if self._first_move:
            self._first_move = False
            print("AI first turn: skip detection, board is empty")
        else:
            self._send(pack_thinking())
            time.sleep(MECHANICAL_DELAY_S)

            prev = self.board.copy()
            new_grid = prev

            for attempt in range(3):
                frame = self._capture()
                if frame is None:
                    print("Warning: camera capture failed")
                    continue

                board_img, _ = self.detector.crop(frame)
                if board_img is None:
                    print(f"Detect[{attempt}]: board not found, retry")
                    continue

                new_grid, _ = self.classifier.classify(board_img, prev)
                print(f"Detect[{attempt}]: prev={prev} new={new_grid}")

                if new_grid == prev:
                    continue

                # 找到新落子并扣减库存
                for i in range(9):
                    if prev[i] == EMPTY and new_grid[i] != EMPTY:
                        self._consume_piece(new_grid[i])
                        break

                break
            else:
                print("Detect: no change after 3 attempts, using prev")

            self.board = new_grid

            # 检测后立即终局判断
            if game_over(self.board):
                winner = check_winner(self.board)
                if winner == self.human_piece:
                    self._send(pack_game_over(RESULT_HUMAN_WIN))
                    print("Game Over: Human wins")
                elif is_draw(self.board):
                    self._send(pack_game_over(RESULT_DRAW))
                    print("Game Over: Draw")
                self.state = State.IDLE
                return

        # AI 决策
        move_idx = ai_move(self.board, self.ai_piece, AI_DIFFICULTY)
        if move_idx < 0:
            print("Error: AI returned invalid move")
            self.state = State.IDLE
            return

        self.board[move_idx] = self.ai_piece
        self._consume_piece(self.ai_piece)

        # 计算物理坐标
        if self.calib is not None:
            x_mm, y_mm = cell_center_physical(
                move_idx, self.calib["board_width_mm"], self.calib["board_height_mm"]
            )
        else:
            x_mm, y_mm = 0.0, 0.0

        self._send(pack_ai_move(move_idx, x_mm, y_mm))
        print(f"AI  → index={move_idx}, ({x_mm:.1f}, {y_mm:.1f})mm")

        # AI 是否赢了
        if game_over(self.board):
            winner = check_winner(self.board)
            if winner == self.ai_piece:
                self._send(pack_game_over(RESULT_AI_WIN))
                print("Game Over: AI wins")
            elif is_draw(self.board):
                self._send(pack_game_over(RESULT_DRAW))
                print("Game Over: Draw")
            self.state = State.IDLE
        else:
            self.turn = self.human_piece
            self.state = State.WAIT_PLAYER

    def _do_calibrate(self) -> None:
        """标定流程：拍照 → 检测棋盘 → 计算 Homography → 保存。"""
        print("Calibrate: capturing frame...")
        frame = self._capture()
        if frame is None:
            print("Calibrate: camera capture failed")
            self.state = State.IDLE
            return

        board_img, corners = self.detector.detect(frame)
        if board_img is None:
            print("Calibrate: board not found")
            self.state = State.IDLE
            return

        H = compute_homography(corners, BOARD_WIDTH_MM, BOARD_HEIGHT_MM)
        save_calibration(H, BOARD_WIDTH_MM, BOARD_HEIGHT_MM)
        self._load_calibration()
        print("Calibrate: done")
        self.state = State.IDLE

    # ================================================================
    # 串口
    # ================================================================

    def _open_serial(self) -> None:
        if self.mock:
            print("Serial: MOCK mode (keyboard input)")
            return
        try:
            import serial
            self.ser = serial.Serial(
                SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT
            )
            print(f"Serial: {SERIAL_PORT} @ {SERIAL_BAUDRATE}")
        except Exception as e:
            print(f"Serial open failed: {e}")
            print("Fallback to MOCK mode")
            self.mock = True

    def _send(self, data: bytes) -> None:
        """发送一帧到串口（或 mock 模式打印）。"""
        if self.mock:
            print(f"  [TX] {data.hex(' ')}")
        elif self.ser and self.ser.is_open:
            self.ser.write(data)

    def _read_serial(self) -> None:
        """非阻塞读取串口，处理完整帧。"""
        if self.mock:
            self._read_mock_input()
            return

        if self.ser and self.ser.in_waiting > 0:
            data = self.ser.read(self.ser.in_waiting)
            self.fb.feed(data)
            while frame := self.fb.pop():
                self._handle_frame(frame)

    def _handle_frame(self, frame: bytes) -> None:
        """处理一帧完整指令。"""
        parsed = unpack(frame)
        if parsed is None:
            return
        cmd, data = parsed["cmd"], parsed["data"]
        print(f"  [RX] cmd=0x{cmd:02X} data={data.hex(' ')}")

        if cmd == CMD_GAME_START:
            self._on_game_start()
        elif cmd == CMD_WHO_FIRST:
            self._on_who_first(data)
        elif cmd == CMD_PLAYER_DONE:
            self._on_player_done()
        elif cmd == CMD_RESET:
            self._on_reset()

    def _on_game_start(self) -> None:
        """① GAME_START: 复位棋盘，等待 WHO_FIRST。"""
        self.board = [EMPTY] * 9
        self.iron_left = IRON_COUNT
        self.red_left = RED_COUNT
        self.detector.reset()
        self._first_move = True
        self.state = State.IDLE
        print("GAME_START: board reset, waiting WHO_FIRST...")

    def _on_who_first(self, data: bytes) -> None:
        """② WHO_FIRST: 指定先手，正式开始。"""
        first = unpack_who_first(data)
        self.ai_first = (first == AI_FIRST)
        # 只有AI先手且空盘才跳过拍照；人先手_do_ai_think之前一定是人走过，不是空盘
        if not self.ai_first:
            self._first_move = False
        if self.ai_first:
            self.ai_piece = IRON
            self.human_piece = RED
            self.turn = IRON
            print("WHO_FIRST: AI first (Iron)")
            self.state = State.AI_THINK
        else:
            self.ai_piece = RED
            self.human_piece = IRON
            self.turn = IRON  # 人执铁色先走
            print("WHO_FIRST: Human first (Iron)")
            self.state = State.WAIT_PLAYER

    def _on_player_done(self) -> None:
        """④ PLAYER_DONE: 玩家落子完毕，触发自动检测。"""
        if self.state == State.IDLE:
            print("PLAYER_DONE: ignoring (still IDLE, wait GAME_START+WHO_FIRST)")
            return
        print("PLAYER_DONE: triggering detection...")
        self.state = State.AI_THINK

    def _on_reset(self) -> None:
        """RESET: 软复位。"""
        self.board = [EMPTY] * 9
        self.iron_left = IRON_COUNT
        self.red_left = RED_COUNT
        self.detector.reset()
        self.state = State.IDLE
        print("RESET → IDLE")

    # ================================================================
    # 棋子库存
    # ================================================================

    def _consume_piece(self, piece: int) -> None:
        if piece == IRON:
            self.iron_left -= 1
        elif piece == RED:
            self.red_left -= 1

    def _has_piece(self, piece: int) -> bool:
        if piece == IRON:
            return self.iron_left > 0
        return self.red_left > 0

    # ================================================================
    # Mock 键盘输入
    # ================================================================

    def _read_mock_input(self) -> None:
        """非阻塞读键盘 (仅 mock 模式)。"""
        try:
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                self._handle_mock_key(ch)
        except ImportError:
            # Linux: select-based non-blocking stdin
            import select
            import sys as _sys
            if select.select([_sys.stdin], [], [], 0)[0]:
                ch = _sys.stdin.read(1).lower()
                self._handle_mock_key(ch)

    def _handle_mock_key(self, ch: str) -> None:
        """处理单个 mock 按键 (Windows/Linux 共用)。"""
        if ch == "n":
            # AI先手：GAME_START + WHO_FIRST(AI)
            self._handle_frame(bytes([0xAA, CMD_GAME_START, 0xBB]))
            self._handle_frame(bytes([0xAA, CMD_WHO_FIRST, AI_FIRST, 0xBB]))
        elif ch == "h":
            # 人先手：GAME_START + WHO_FIRST(人)
            self._handle_frame(bytes([0xAA, CMD_GAME_START, 0xBB]))
            self._handle_frame(bytes([0xAA, CMD_WHO_FIRST, HUMAN_FIRST, 0xBB]))
        elif ch == "r":
            self._handle_frame(bytes([0xAA, CMD_PLAYER_DONE, 0xBB]))
        elif ch == "c":
            self.state = State.CALIBRATING
        elif ch == "x":
            self._handle_frame(bytes([0xAA, CMD_RESET, 0xBB]))
        elif ch == "q":
            raise KeyboardInterrupt

    # ================================================================
    # 相机
    # ================================================================

    def _init_camera(self) -> None:
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print(f"WARNING: Camera index {CAMERA_INDEX} not available")
            self.cap = None
        else:
            print(f"Camera {CAMERA_INDEX} ready")

    def _capture(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        for _ in range(2):
            self.cap.read()
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    # ================================================================
    # 标定
    # ================================================================

    def _load_calibration(self) -> None:
        self.calib = load_calibration()
        if self.calib is not None:
            print("Calibration loaded")
        else:
            print("No calibration file found, using zero coordinates")

    # ================================================================
    # 清理
    # ================================================================

    def _cleanup(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup done.")


# ================================================================
# CLI 入口
# ================================================================

def main() -> None:
    mock = "--mock" in sys.argv
    calibrate_only = "--calibrate" in sys.argv

    system = TicTacToeSystem(mock_serial=mock)

    if calibrate_only:
        print("Calibrate mode")
        system._init_camera()
        system._do_calibrate()
        system._cleanup()
        return

    system.run()


if __name__ == "__main__":
    main()
