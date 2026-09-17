"""
Ascend NPU 推理模块 — 替代 ultralytics.YOLO，直接在昇腾 310B NPU 上运行 YOLOv8n。

用法:
    from ascend_inference import AscendInference
    model = AscendInference("models/chessboard_v8n.om")
    results = model(frame, conf=0.5)          # 等价于 ultralytics.YOLO(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 兼容属性链

依赖: CANN toolkit 预装的 acl Python 模块 (已验证 CANN 7.6 / 8.0)
"""

from __future__ import annotations

import cv2
import numpy as np

# CANN 7.6 确认过的常量
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_NORMAL_ONLY = 2


# ================================================================
# Fake result wrappers — 模拟 ultralytics 的返回类型
# ================================================================

class _FakeXYXY:
    __slots__ = ("_array",)
    def __init__(self, array): self._array = array
    def cpu(self): return self
    def numpy(self): return self._array


class _FakeBoxes:
    __slots__ = ("xyxy",)
    def __init__(self, array): self.xyxy = _FakeXYXY(array)


class _FakeResult:
    __slots__ = ("boxes",)
    def __init__(self, array):
        if array is not None and len(array) > 0:
            self.boxes = _FakeBoxes(array)
        else:
            self.boxes = _FakeBoxes(None)


# ================================================================
# YOLOv8n 输出后处理
# ================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def _postprocess_yolov8n(
    output: np.ndarray,
    orig_h: int,
    orig_w: int,
    input_size: int = 640,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> np.ndarray | None:
    """
    YOLOv8n ONNX 输出 → xyxy bbox 数组。

    支持:
        (1, 5, 8400)  — 单类 (4 bbox + 1 score, 已做 sigmoid)
        (1, 84, 8400) — 多类 (4 bbox + 80 logits)

    返回: (N, 5) [x1,y1,x2,y2,conf] 或 None
    """
    data = output[0]                     # (C, 8400)
    n_channels = data.shape[0]

    bbox_raw = data[:4, :]               # (4, 8400)
    cls_raw = data[4:, :]                # (1, 8400) 或 (80, 8400)

    if n_channels == 5:
        cls_max = cls_raw[0]             # 单类, score 已生效
    elif cls_raw.shape[0] > 1:
        cls_max = _sigmoid(cls_raw).max(axis=0)
    else:
        cls_max = _sigmoid(cls_raw[0])

    keep = cls_max >= conf_threshold
    if not np.any(keep):
        return None

    # bbox 坐标是输入空间 (0~input_size) 的绝对像素值，非归一化
    cx, cy, w, h = bbox_raw[0, keep], bbox_raw[1, keep], bbox_raw[2, keep], bbox_raw[3, keep]
    scores = cls_max[keep]
    half_w, half_h = w / 2, h / 2

    x1_px = cx - half_w
    y1_px = cy - half_h
    x2_px = cx + half_w
    y2_px = cy + half_h

    boxes_list = [
        [float(x1_px[i]), float(y1_px[i]), float(x2_px[i]), float(y2_px[i])]
        for i in range(len(scores))
    ]

    indices = cv2.dnn.NMSBoxes(boxes_list, scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0:
        return None

    if isinstance(indices, np.ndarray):
        keep_idx = indices.flatten()
    else:
        keep_idx = np.array([i[0] for i in indices])
    if len(keep_idx) == 0:
        return None

    # 从输入空间缩放到原图空间
    xs = orig_w / input_size
    ys = orig_h / input_size
    result = np.zeros((len(keep_idx), 5), dtype=np.float32)
    for j, idx in enumerate(keep_idx):
        result[j, 0] = max(0.0, x1_px[idx] * xs)
        result[j, 1] = max(0.0, y1_px[idx] * ys)
        result[j, 2] = min(float(orig_w), x2_px[idx] * xs)
        result[j, 3] = min(float(orig_h), y2_px[idx] * ys)
        result[j, 4] = scores[idx]

    return result[result[:, 4].argsort()[::-1]]


# ================================================================
# Ascend ACL 推理引擎 (CANN 7.6 验证)
# ================================================================

class AscendInference:
    """Ascend NPU YOLOv8n 推理引擎 (与 ultralytics.YOLO 兼容接口)。"""

    def __init__(self, model_path: str, input_size: int = 640, device_id: int = 0):
        import acl

        self._input_size = input_size
        self._device_id = device_id
        self._loaded = False

        # --- 初始化 ACL ---
        ret = acl.init()
        self._check(ret, "acl.init")

        ret = acl.rt.set_device(device_id)
        self._check(ret, "acl.rt.set_device")

        self._context, ret = acl.rt.create_context(device_id)
        self._check(ret, "acl.rt.create_context")

        # --- 加载模型 ---
        self._model_id, ret = acl.mdl.load_from_file(model_path)
        self._check(ret, f"acl.mdl.load_from_file({model_path})")

        self._model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._model_desc, self._model_id)
        self._check(ret, "acl.mdl.get_desc")

        # --- 输入/输出信息 ---
        n_in = acl.mdl.get_num_inputs(self._model_desc)
        n_out = acl.mdl.get_num_outputs(self._model_desc)

        self._input_sizes = [acl.mdl.get_input_size_by_index(self._model_desc, i) for i in range(n_in)]
        self._output_sizes = [acl.mdl.get_output_size_by_index(self._model_desc, i) for i in range(n_out)]

        self._output_dims = []
        for i in range(n_out):
            dims_info, ret = acl.mdl.get_output_dims(self._model_desc, i)
            self._check(ret, f"get_output_dims[{i}]")
            self._output_dims.append(list(dims_info.get("dims", [])))

        # --- 预分配设备内存 ---
        self._d_inputs = [acl.rt.malloc(s, ACL_MEM_MALLOC_NORMAL_ONLY)[0] for s in self._input_sizes]
        self._d_outputs = [acl.rt.malloc(s, ACL_MEM_MALLOC_NORMAL_ONLY)[0] for s in self._output_sizes]

        self._loaded = True
        print(f"[AscendInference] Loaded: {model_path}  "
              f"input={self._input_sizes}  output_dims={self._output_dims}")

    # ---------------------------------------------------------------
    # 公开 API — 与 ultralytics.YOLO.__call__ 兼容
    # ---------------------------------------------------------------

    def __call__(self, frame: np.ndarray, conf: float = 0.5,
                 verbose: bool = False) -> list[_FakeResult]:
        if not self._loaded:
            return [_FakeResult(None)]
        boxes = self._infer(frame, conf_threshold=conf)
        return [_FakeResult(boxes)]

    # ---------------------------------------------------------------
    # 推理核心
    # ---------------------------------------------------------------

    def _infer(self, frame: np.ndarray, conf_threshold: float = 0.5) -> np.ndarray | None:
        import acl

        orig_h, orig_w = frame.shape[:2]
        input_data = self._preprocess(frame)

        # H2D
        ret = acl.rt.memcpy(
            self._d_inputs[0], self._input_sizes[0],
            input_data.ctypes.data, input_data.nbytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        self._check(ret, "memcpy H2D")

        # 构建 dataset（每次构建，复用设备 buffer）
        in_ds = acl.mdl.create_dataset()
        out_ds = acl.mdl.create_dataset()
        in_buf = acl.create_data_buffer(self._d_inputs[0], self._input_sizes[0])
        out_buf = acl.create_data_buffer(self._d_outputs[0], self._output_sizes[0])
        acl.mdl.add_dataset_buffer(in_ds, in_buf)
        acl.mdl.add_dataset_buffer(out_ds, out_buf)

        # 推理
        ret = acl.mdl.execute(self._model_id, in_ds, out_ds)
        self._check(ret, "acl.mdl.execute")

        # D2H
        out_arr = np.empty(self._output_sizes[0] // 4, dtype=np.float32)
        ret = acl.rt.memcpy(
            out_arr.ctypes.data, self._output_sizes[0],
            self._d_outputs[0], self._output_sizes[0],
            ACL_MEMCPY_DEVICE_TO_HOST,
        )
        self._check(ret, "memcpy D2H")

        # 释放临时 data buffers + datasets
        for ds in (in_ds, out_ds):
            n = acl.mdl.get_dataset_num_buffers(ds)
            for i in range(n):
                acl.destroy_data_buffer(acl.mdl.get_dataset_buffer(ds, i))
            acl.mdl.destroy_dataset(ds)

        # 后处理
        output_tensor = out_arr.reshape(self._output_dims[0])
        return _postprocess_yolov8n(
            output_tensor, orig_h, orig_w,
            input_size=self._input_size, conf_threshold=conf_threshold,
        )

    # ---------------------------------------------------------------
    # 预处理
    # ---------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR (H,W,3) → (1,3,640,640) float32 contiguous"""
        img = cv2.resize(frame, (self._input_size, self._input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)          # CHW
        img = np.expand_dims(img, axis=0)      # NCHW
        return np.ascontiguousarray(img, dtype=np.float32)

    # ---------------------------------------------------------------
    # 辅助
    # ---------------------------------------------------------------

    def _check(self, ret, msg):
        if ret != 0:
            raise RuntimeError(f"ACL error (ret={ret}): {msg}")

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        if not self._loaded:
            return
        try:
            import acl
            for buf in getattr(self, '_d_inputs', []) + getattr(self, '_d_outputs', []):
                acl.rt.free(buf)
            acl.mdl.destroy_desc(self._model_desc)
            acl.mdl.unload(self._model_id)
            acl.rt.destroy_context(self._context)
            acl.rt.reset_device(self._device_id)
            acl.finalize()
            self._loaded = False
        except Exception:
            pass


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import sys, os

    if len(sys.argv) < 2:
        print("Usage: python ascend_inference.py <image.jpg> [model.om]")
        sys.exit(1)

    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/chessboard_v8n.om"
    img_path = sys.argv[1]

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to load: {img_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = AscendInference(model_path)

    print(f"Running inference on {img_path} ({frame.shape[1]}x{frame.shape[0]})")
    results = model(frame, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if boxes is not None and len(boxes) > 0:
        print(f"Detected {len(boxes)} board(s):")
        for i, box in enumerate(boxes):
            print(f"  [{i}] {box[:4].astype(int)}  conf={box[4]:.3f}")
    else:
        print("No board detected.")
    print("Self-test passed.")
