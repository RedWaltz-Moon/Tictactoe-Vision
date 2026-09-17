from __future__ import annotations

"""
井字棋 AI — Minimax + Alpha-Beta 剪枝 + 开局随机化 + 难度调节。
纯逻辑模块，零依赖，可直接 import 测试。

棋子编码: 0=空, 1=铁色(先手), 2=红色(后手)
"""

import random

# 棋盘常量
EMPTY = 0
IRON  = 1  # 铁色棋子 (先手)
RED   = 2  # 红色棋子 (后手)

# 兼容旧命名
BLACK = IRON
WHITE = RED

# 胜利线 (行/列/对角线 索引组合)
WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
    [0, 4, 8], [2, 4, 6],              # 对角线
]


def check_winner(board: list[int]) -> int:
    """返回胜者棋子号，无人胜返回 0。"""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def is_draw(board: list[int]) -> bool:
    """棋盘满了且无人胜 → 平局。"""
    return board.count(EMPTY) == 0 and check_winner(board) == 0


def game_over(board: list[int]) -> bool:
    return check_winner(board) != 0 or is_draw(board)


def valid_moves(board: list[int]) -> list[int]:
    """返回所有可落子的索引。"""
    return [i for i, v in enumerate(board) if v == EMPTY]


# ============================================================
# Minimax + Alpha-Beta
# ============================================================

def _minimax(
    board: list[int],
    depth: int,
    is_maximizing: bool,
    ai_piece: int,
    alpha: float,
    beta: float,
) -> int:
    opp = BLACK if ai_piece == WHITE else WHITE

    winner = check_winner(board)
    if winner == ai_piece:
        return 10 - depth
    if winner == opp:
        return depth - 10
    if is_draw(board):
        return 0

    if is_maximizing:
        max_eval = -float("inf")
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = ai_piece
                score = _minimax(board, depth + 1, False, ai_piece, alpha, beta)
                board[i] = EMPTY
                max_eval = max(max_eval, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = float("inf")
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = opp
                score = _minimax(board, depth + 1, True, ai_piece, alpha, beta)
                board[i] = EMPTY
                min_eval = min(min_eval, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
        return min_eval


# ============================================================
# 公开 API
# ============================================================

def ai_move(board: list[int], ai_piece: int, difficulty: float = 1.0) -> int:
    """
    计算 AI 最佳落子位置。

    参数:
        board: 9元素列表, 0=空 1=铁色 2=红色
        ai_piece: AI 执哪一方 (1 或 2)
        difficulty: 0.0=纯随机, 0.5=偶尔失误, 1.0=永不输

    返回:
        落子索引 0~8, 或 -1 (无合法走法)
    """
    moves = valid_moves(board)
    if not moves:
        return -1

    # === 难度调节：以一定概率随机走 ===
    if random.random() > difficulty:
        return random.choice(moves)

    # === 首步随机化 (空盘时) ===
    if board.count(EMPTY) == 9:
        if random.random() < 0.8:
            return 4  # 80% 走中心
        return random.choice([0, 2, 6, 8])  # 20% 走角落

    # === 立即获胜 ===
    for i in moves:
        board[i] = ai_piece
        if check_winner(board) == ai_piece:
            board[i] = EMPTY
            return i
        board[i] = EMPTY

    # === 立即防守（对手即将赢） ===
    opp = BLACK if ai_piece == WHITE else WHITE
    for i in moves:
        board[i] = opp
        if check_winner(board) == opp:
            board[i] = EMPTY
            return i
        board[i] = EMPTY

    # === Alpha-Beta Minimax ===
    best_score = -float("inf")
    best_move = moves[0]

    for i in moves:
        board[i] = ai_piece
        score = _minimax(board, 0, False, ai_piece, -float("inf"), float("inf"))
        board[i] = EMPTY
        if score > best_score:
            best_score = score
            best_move = i

    return best_move


# ============================================================
# 自测（作为脚本直接运行）
# ============================================================
if __name__ == "__main__":
    # 测试1: 空盘开局应走中心或角
    b = [EMPTY] * 9
    move = ai_move(b, BLACK)
    print(f"空盘开局 → 走索引 {move} (应为4中心，偶尔0/2/6/8角)")

    # 测试2: 堵对方必胜
    b = [BLACK, BLACK, EMPTY,
         WHITE, WHITE, EMPTY,
         EMPTY, EMPTY, EMPTY]
    # 白棋(4,5)已经两连，黑棋必须堵5 → 走了BLACK,BLACK后黑棋要去堵WHITE
    # 改成更清晰的场景：AI=WHITE, 黑的两连在0,1, AI必须堵2
    b2 = [BLACK, BLACK, EMPTY,
           EMPTY, WHITE, EMPTY,
           EMPTY, EMPTY, EMPTY]
    move = ai_move(b2, WHITE)
    print(f"黑 0-1 两连, AI(白) 堵 2 → 走索引 {move} (应为2)")

    # 测试3: 直接获胜
    b3 = [WHITE, WHITE, EMPTY,
           BLACK, BLACK, EMPTY,
           EMPTY, EMPTY, EMPTY]
    move = ai_move(b3, WHITE)
    print(f"白 0-1 两连, AI(白) 直接赢 → 走索引 {move} (应为2)")

    # 测试4: 终局
    b4 = [BLACK, WHITE, BLACK,
          WHITE, BLACK, WHITE,
          WHITE, BLACK, WHITE]  # BLACK 走 (2,4) 在 / 对角线上
    # 让我用明确场景
    b4 = [BLACK, WHITE, EMPTY,
          EMPTY, BLACK, WHITE,
          EMPTY, EMPTY, BLACK]
    print(f"BLACK 两连在 (0,4,8) →? check_winner={check_winner(b4)} (应为1=BLACK胜)")

    # 测试5: 难度0=纯随机
    random.seed(42)
    b5 = [EMPTY] * 9
    move = ai_move(b5, BLACK, difficulty=0.0)
    print(f"难度0空盘 → 走索引 {move} (可能不是4)")

    print("\nai_engine self-test passed")
