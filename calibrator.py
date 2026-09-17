from __future__ import annotations

"""
坐标标定：像素坐标 ↔ 棋盘物理坐标 (mm)。

两种标定模式:
  1. 自动标定: 输入棋盘4个角点的像素坐标 + 物理尺寸 → 计算仿射变换矩阵
  2. 相机标定: 棋盘格内参标定 → 去畸变矩阵 (可选，固定俯拍通常不需要)
"""

import json
import os

import cv2
import numpy as np

# 棋盘角点物理坐标 (棋盘左上角为原点，X右，Y下)
# 顺序: 左上、右上、左下、右下
CORNERS_PHYSICAL_ORDER = ["top_left", "top_right", "bottom_left", "bottom_right"]


def compute_homography(
    pixel_corners: np.ndarray,
    board_width_mm: float,
    board_height_mm: float,
) -> np.ndarray:
    """
    计算像素 → 物理坐标的仿射变换矩阵 (3x3)。

    参数:
        pixel_corners: 4个角点的像素坐标, shape (4,2), dtype float32
                       顺序: [左上, 右上, 左下, 右下]
        board_width_mm:  棋盘物理宽度
        board_height_mm: 棋盘物理高度

    返回:
        H: 3x3 变换矩阵，使得 phys_xy = H @ pixel_xy_homogeneous
    """
    physical_corners = np.array(
        [
            [0, 0],
            [board_width_mm, 0],
            [0, board_height_mm],
            [board_width_mm, board_height_mm],
        ],
        dtype=np.float32,
    )
    # 用仿射变换 (3对点即可，但4对更鲁棒)
    H, _ = cv2.findHomography(pixel_corners, physical_corners, method=cv2.RANSAC)
    return H


def pixel_to_physical(
    H: np.ndarray, px: float, py: float
) -> tuple[float, float]:
    """单点像素坐标 → 物理坐标。"""
    pt = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
    phys = cv2.perspectiveTransform(pt, H).reshape(-1, 2)
    return float(phys[0, 0]), float(phys[0, 1])


def physical_to_pixel(
    H: np.ndarray, mx: float, my: float
) -> tuple[float, float]:
    """
    单点物理坐标 → 像素坐标 (逆变换)。
    注意: 物理坐标原点为棋盘左上角，X右Y下。
    """
    H_inv = np.linalg.inv(H)
    pt = np.array([[mx, my]], dtype=np.float32).reshape(-1, 1, 2)
    pix = cv2.perspectiveTransform(pt, H_inv).reshape(-1, 2)
    return float(pix[0, 0]), float(pix[0, 1])


def cell_center_physical(
    index: int, board_width_mm: float, board_height_mm: float
) -> tuple[float, float]:
    """
    根据索引 (0~8) 计算该格中心的物理坐标。

    索引布局:
      0 1 2
      3 4 5
      6 7 8
    """
    cell_w = board_width_mm / 3.0
    cell_h = board_height_mm / 3.0
    row, col = divmod(index, 3)
    x = (col + 0.5) * cell_w
    y = (row + 0.5) * cell_h
    return x, y


def all_cell_centers(
    board_width_mm: float, board_height_mm: float,
) -> list[tuple[float, float]]:
    """返回 9 个格子中心的物理坐标列表, index 0~8。"""
    return [
        cell_center_physical(i, board_width_mm, board_height_mm)
        for i in range(9)
    ]


# ============================================================
# 标定数据存取
# ============================================================

CALIB_FILE = "calibration.json"


def save_calibration(
    H: np.ndarray,
    board_w: float,
    board_h: float,
    filepath: str = CALIB_FILE,
) -> None:
    """保存标定矩阵到 JSON 文件。"""
    data = {
        "H": H.tolist(),
        "board_width_mm": board_w,
        "board_height_mm": board_h,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {filepath}")


def load_calibration(filepath: str = CALIB_FILE) -> dict | None:
    """加载标定矩阵，文件不存在返回 None。"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        data = json.load(f)
    data["H"] = np.array(data["H"], dtype=np.float32)
    return data


# ============================================================
# 相机内参标定 (棋盘格法)
# ============================================================

def calibrate_camera(
    images: list[np.ndarray],
    pattern_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用多张棋盘格图片标定相机内参。

    返回:
        K: 内参矩阵 (3,3)
        dist: 畸变系数 (5,)
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        * square_size_mm
    )

    obj_points = []
    img_points = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    if len(obj_points) < 3:
        raise ValueError(
            f"Need at least 3 valid chessboard images, got {len(obj_points)}"
        )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        images[0].shape[:2][::-1],  # (width, height)
        None,
        None,
    )
    return K, dist


def undistort(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """对图像去畸变。"""
    return cv2.undistort(image, K, dist)


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    # 模拟: 已知4个角点的像素坐标 + 棋盘150x150mm
    pixels = np.array(
        [[100, 50], [400, 80], [90, 320], [420, 350]], dtype=np.float32
    )
    H = compute_homography(pixels, 150.0, 150.0)
    print(f"Homography (3x3):\n{H}")

    # 验证: 左上角像素应映射到 (0,0)
    phys = pixel_to_physical(H, 100, 50)
    print(f"左上角 像素(100,50) → 物理{phys} (应为 ~(0,0))")

    # 格子中心
    centers = all_cell_centers(150.0, 150.0)
    print(f"9格物理坐标 (mm):")
    for i, (x, y) in enumerate(centers):
        print(f"  索引{i}: ({x:.1f}, {y:.1f})")

    # 逆变换: 格子4的中心物理坐标 (75,75) 应该在图像中央
    px, py = physical_to_pixel(H, 75.0, 75.0)
    print(f"索引4物理(75,75) → 像素({px:.1f}, {py:.1f})")

    # 测试存取 (写到项目目录，避免 Windows tempfile 权限问题)
    test_file = "_calib_test.json"
    try:
        save_calibration(H, 150.0, 150.0, test_file)
        loaded = load_calibration(test_file)
        assert loaded is not None
        print(f"Load OK, H shape={loaded['H'].shape}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

    print("calibrator self-test passed")
