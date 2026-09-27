"""第 2 頁：骨架偵測與相機參數設定。

前提：第 1 頁（camera_setting.CameraSetting）已選擇並開啟相機，把同一個物件傳進來。
本頁只調整參數與讀取影像，不會關閉相機，也不會動到第 1 頁的狀態。

所有可調參數都用 Param 描述，介面依 kind 自動產生元件：
- CHOICE / BOOL -> 下拉式選單（options 為 (值, 顯示文字)）
- INT / FLOAT   -> 數字輸入（min / max / step）
呼叫 set_param() 立即生效；不合法的值會被拒絕、保留原值，並寫入 NowErrorLog。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Self

import cv2
import mediapipe as mp
import numpy as np
from camera_setting import CameraSetting, ErrorLog, ErrorReporter
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerResult,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

GROUP_MODEL = "骨架模型"
GROUP_CAMERA = "相機"
GROUP_DISPLAY = "顯示"

MODEL_VARIANTS = [("lite", "Lite（最快）"), ("full", "Full（平衡）"), ("heavy", "Heavy（最準）")]
RESOLUTION_CANDIDATES = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
FPS_OPTIONS = [15, 24, 30, 60]
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# 改變這些參數需要重建 PoseLandmarker
LANDMARKER_KEYS = frozenset({
    "model",
    "num_poses",
    "min_pose_detection_confidence",
    "min_pose_presence_confidence",
    "min_tracking_confidence",
})

NUM_LANDMARKS = 33
UPPER_BODY = frozenset(range(25))  # 0~24：臉、肩、手臂、髖，坐姿分析常用
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


class DetectionErrorCode(Enum):
    CAMERA_NOT_READY = "相機尚未開啟，請先完成第 1 頁的相機設定"
    MODEL_NOT_FOUND = "找不到骨架模型檔"
    MODEL_LOAD_FAILED = "骨架模型載入失敗"
    UNKNOWN_PARAM = "沒有這個參數"
    INVALID_VALUE = "參數值不合法"
    CAMERA_PROP_UNSUPPORTED = "相機不支援此設定"
    DETECT_FAILED = "骨架偵測失敗"


# ---------- 參數描述 ----------


class ParamKind(Enum):
    CHOICE = "choice"  # 下拉式選單
    BOOL = "bool"  # 下拉式選單（開 / 關）
    INT = "int"  # 數字輸入
    FLOAT = "float"  # 數字輸入


@dataclass
class Param:
    key: str
    label: str
    group: str
    kind: ParamKind
    default: Any
    options: list[tuple[Any, str]] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    help: str = ""
    value: Any = None

    def __post_init__(self) -> None:
        if self.kind is ParamKind.BOOL and not self.options:
            self.options = [(True, "開"), (False, "關")]
        if self.value is None:
            self.value = self.default

    @property
    def is_dropdown(self) -> bool:
        return self.kind in (ParamKind.CHOICE, ParamKind.BOOL)

    @property
    def display_value(self) -> str:
        for value, label in self.options:
            if value == self.value:
                return label
        return str(self.value)

    def coerce(self, raw: Any) -> Any:
        """轉成正確型別並檢查範圍，不合法時丟 ValueError。下拉選單可傳值或顯示文字。"""
        if self.is_dropdown:
            for value, label in self.options:
                if raw == value or raw == label:
                    return value
            raise ValueError(f"可選值：{[label for _, label in self.options]}")
        if isinstance(raw, bool):
            raise ValueError("請輸入數字")
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise ValueError("請輸入數字") from None
        if self.kind is ParamKind.INT:
            if not num.is_integer():
                raise ValueError("請輸入整數")
            num = int(num)
        if not self.min <= num <= self.max:
            raise ValueError(f"範圍 {self.min} ~ {self.max}")
        return num


class ParamSet:
    """依加入順序保存參數，介面用 groups() / specs(group) 逐區產生元件。"""

    def __init__(self, params: list[Param]) -> None:
        self._params = {p.key: p for p in params}

    def __contains__(self, key: str) -> bool:
        return key in self._params

    def __getitem__(self, key: str) -> Any:
        return self._params[key].value

    def spec(self, key: str) -> Param:
        return self._params[key]

    def specs(self, group: str | None = None) -> list[Param]:
        return [p for p in self._params.values() if group is None or p.group == group]

    def groups(self) -> list[str]:
        return list(dict.fromkeys(p.group for p in self._params.values()))

    def values(self) -> dict[str, Any]:
        return {key: p.value for key, p in self._params.items()}


# ---------- 骨架偵測 ----------


@dataclass
class DetectionFrame:
    ok: bool
    frame: np.ndarray | None = None  # 已畫上骨架、可直接顯示的 BGR 影像
    result: PoseLandmarkerResult | None = None  # 關鍵點座標以「鏡像前」的影像為準
    inference_ms: float = 0.0


class SkeletonDetection(ErrorReporter):
    """第 2 頁。

    用法：
        page = SkeletonDetection(camera)          # camera 為第 1 頁已開啟的 CameraSetting
        for group in page.params.groups():        # 依參數描述產生下拉選單 / 數字輸入
            for spec in page.params.specs(group): ...
        page.set_param("model", "heavy")          # 介面元件變動時呼叫，立即生效
        f = page.process_frame()                  # 每幀呼叫，f.frame 直接顯示
        page.close()                              # 離開本頁；不會關閉相機
    """

    def __init__(
        self,
        camera: CameraSetting,
        models_dir: Path = MODELS_DIR,
        error_log: ErrorLog | None = None,
    ) -> None:
        super().__init__(error_log if error_log is not None else camera.error_log)
        self.camera = camera
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()  # 保護 landmarker：偵測中不能被替換或關閉
        self._landmarker: PoseLandmarker | None = None
        self._last_ts = 0
        self._not_ready_reported = False

        # 本頁期間相機發生的錯誤（例如被拔除）也要在本頁彈窗；總紀錄已由相機寫入，不重複
        self._forward_camera_error = self.now_error_log.add
        camera.add_error_listener(self._forward_camera_error)

        if not self.ready:
            self._report(DetectionErrorCode.CAMERA_NOT_READY)
            self._not_ready_reported = True
        self.params = ParamSet(self._build_params())
        self._landmarker = self._create_landmarker()

    def _context_index(self) -> int | None:
        return self.camera.camera_index

    @property
    def ready(self) -> bool:
        return self.camera.is_opened

    # ---------- 參數 ----------

    def _build_params(self) -> list[Param]:
        resolution = self._current_resolution()
        resolutions = self._probe_resolutions(resolution)
        fps = round(self.camera.get_property(cv2.CAP_PROP_FPS) or 0)
        conf = {"kind": ParamKind.FLOAT, "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}
        return [
            Param("model", "骨架模型", GROUP_MODEL, ParamKind.CHOICE, "full", MODEL_VARIANTS,
                  help="越準確越慢，可參考畫面上的推論時間"),
            Param("num_poses", "偵測人數上限", GROUP_MODEL, ParamKind.INT, 1,
                  min=1, max=5, step=1),
            Param("min_pose_detection_confidence", "偵測信心門檻", GROUP_MODEL, **conf,
                  help="越高越不容易誤判，但人可能偵測不到"),
            Param("min_pose_presence_confidence", "存在信心門檻", GROUP_MODEL, **conf),
            Param("min_tracking_confidence", "追蹤信心門檻", GROUP_MODEL, **conf,
                  help="低於門檻時重新偵測，越高越穩定但越慢"),
            Param("resolution", "解析度", GROUP_CAMERA, ParamKind.CHOICE, resolution,
                  [(r, f"{r[0]} x {r[1]}") for r in resolutions],
                  help="選項為這台相機實測可用的解析度"),
            Param("fps", "FPS", GROUP_CAMERA, ParamKind.CHOICE, fps if fps in FPS_OPTIONS else 30,
                  [(f, str(f)) for f in FPS_OPTIONS]),
            Param("rotation", "旋轉", GROUP_CAMERA, ParamKind.CHOICE, 0,
                  [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")],
                  help="相機直放或倒放時使用，會影響偵測"),
            Param("brightness", "亮度", GROUP_CAMERA, ParamKind.INT, 0,
                  min=-100, max=100, step=5, help="軟體調整，環境太暗時可調高"),
            Param("contrast", "對比", GROUP_CAMERA, ParamKind.FLOAT, 1.0,
                  min=0.5, max=3.0, step=0.1, help="軟體調整"),
            Param("show_skeleton", "顯示骨架", GROUP_DISPLAY, ParamKind.BOOL, True),
            Param("landmark_set", "顯示範圍", GROUP_DISPLAY, ParamKind.CHOICE, "full",
                  [("full", "全身（33 點）"), ("upper", "上半身（0~24 點）")]),
            Param("visibility_threshold", "關鍵點可見度門檻", GROUP_DISPLAY, **conf,
                  help="低於門檻的關鍵點不畫出"),
            Param("mirror", "鏡像顯示", GROUP_DISPLAY, ParamKind.BOOL, True,
                  help="只影響畫面，不影響偵測結果的左右"),
        ]

    def set_param(self, key: str, raw: Any) -> bool:
        """介面元件變動時呼叫。成功回傳 True；失敗保留原值並寫入 NowErrorLog。"""
        if key not in self.params:
            self._report(DetectionErrorCode.UNKNOWN_PARAM, key)
            return False
        spec = self.params.spec(key)
        try:
            value = spec.coerce(raw)
        except ValueError as e:
            self._report(DetectionErrorCode.INVALID_VALUE, f"{spec.label}：{e}（收到 {raw!r}）")
            return False
        if value == spec.value:
            return True
        old, spec.value = spec.value, value
        if not self._apply(key, old):
            spec.value = old
            return False
        return True

    def reset(self) -> None:
        """全部恢復預設值。"""
        for spec in self.params.specs():
            self.set_param(spec.key, spec.default)

    def _apply(self, key: str, old: Any) -> bool:
        if key in LANDMARKER_KEYS:
            new = self._create_landmarker()
            if new is None:
                return False
            with self._lock:
                old_landmarker, self._landmarker = self._landmarker, new
            if old_landmarker is not None:
                old_landmarker.close()
            return True
        if key in ("resolution", "fps"):
            if not self.ready:
                self._report(DetectionErrorCode.CAMERA_NOT_READY)
                return False
            return self._apply_resolution(old) if key == "resolution" else self._apply_fps(old)
        return True  # 其他參數在 process_frame 中即時讀取

    def _apply_resolution(self, old: tuple[int, int]) -> bool:
        w, h = self.params["resolution"]
        self.camera.set_property(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.camera.set_property(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual = self._current_resolution()
        if actual != (w, h):
            self._report(DetectionErrorCode.CAMERA_PROP_UNSUPPORTED,
                         f"解析度 {w} x {h}，實際為 {actual[0]} x {actual[1]}")
            self.camera.set_property(cv2.CAP_PROP_FRAME_WIDTH, old[0])
            self.camera.set_property(cv2.CAP_PROP_FRAME_HEIGHT, old[1])
            return False
        return True

    def _apply_fps(self, old: int) -> bool:
        fps = self.params["fps"]
        self.camera.set_property(cv2.CAP_PROP_FPS, fps)
        actual = round(self.camera.get_property(cv2.CAP_PROP_FPS) or 0)
        if actual != fps:
            self._report(DetectionErrorCode.CAMERA_PROP_UNSUPPORTED, f"FPS {fps}，實際為 {actual}")
            self.camera.set_property(cv2.CAP_PROP_FPS, old)
            return False
        return True

    def _current_resolution(self) -> tuple[int, int] | None:
        w = self.camera.get_property(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.camera.get_property(cv2.CAP_PROP_FRAME_HEIGHT)
        return (int(w), int(h)) if w and h else None

    def _probe_resolutions(self, current: tuple[int, int] | None) -> list[tuple[int, int]]:
        """逐一試設候選解析度並讀回，只保留相機真的支援的，最後還原。"""
        if current is None:
            return []
        found = {current}
        for w, h in RESOLUTION_CANDIDATES:
            self.camera.set_property(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.camera.set_property(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if self._current_resolution() == (w, h):
                found.add((w, h))
        self.camera.set_property(cv2.CAP_PROP_FRAME_WIDTH, current[0])
        self.camera.set_property(cv2.CAP_PROP_FRAME_HEIGHT, current[1])
        return sorted(found, key=lambda r: r[0] * r[1])

    # ---------- 模型 ----------

    def _create_landmarker(self) -> PoseLandmarker | None:
        variant = self.params["model"]
        path = self.models_dir / f"pose_landmarker_{variant}.task"
        if not path.exists():
            self._report(DetectionErrorCode.MODEL_NOT_FOUND,
                         f"{path}，請執行 python scripts/download_models.py {variant}")
            return None
        try:
            options = vision.PoseLandmarkerOptions(
                # 用 buffer 讀入，避免路徑含中文時 MediaPipe 開檔失敗
                base_options=mp_tasks.BaseOptions(model_asset_buffer=path.read_bytes()),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=self.params["num_poses"],
                min_pose_detection_confidence=self.params["min_pose_detection_confidence"],
                min_pose_presence_confidence=self.params["min_pose_presence_confidence"],
                min_tracking_confidence=self.params["min_tracking_confidence"],
            )
            return PoseLandmarker.create_from_options(options)
        except Exception as e:
            self._report(DetectionErrorCode.MODEL_LOAD_FAILED, str(e))
            return None

    # ---------- 每幀處理 ----------

    def process_frame(self) -> DetectionFrame:
        """讀一幀、偵測骨架並畫出。失敗時 ok=False，同一種失敗連續發生只記錄一次。"""
        if not self.ready:
            if not self._not_ready_reported:
                self._report(DetectionErrorCode.CAMERA_NOT_READY)
                self._not_ready_reported = True
            return DetectionFrame(False)
        self._not_ready_reported = False

        ok, frame = self.camera.read_frame()
        if not ok:
            return DetectionFrame(False)  # 相機已記錄錯誤，並轉送到本頁 NowErrorLog
        frame = self._preprocess(frame)

        result, inference_ms = None, 0.0
        with self._lock:
            if self._landmarker is not None:
                # VIDEO 模式要求時間戳嚴格遞增
                self._last_ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                start = time.perf_counter()
                try:
                    result = self._landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._last_ts
                    )
                except Exception as e:
                    self._report(DetectionErrorCode.DETECT_FAILED, str(e))
                inference_ms = (time.perf_counter() - start) * 1000

        if result is not None and self.params["show_skeleton"]:
            self._draw(frame, result)
        if self.params["mirror"]:
            frame = cv2.flip(frame, 1)
        return DetectionFrame(True, frame, result, inference_ms)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        rotation = self.params["rotation"]
        if rotation:
            frame = cv2.rotate(frame, ROTATIONS[rotation])
        brightness, contrast = self.params["brightness"], self.params["contrast"]
        if brightness != 0 or contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        return frame

    def _draw(self, frame: np.ndarray, result: PoseLandmarkerResult) -> None:
        h, w = frame.shape[:2]
        indices = UPPER_BODY if self.params["landmark_set"] == "upper" else range(NUM_LANDMARKS)
        threshold = self.params["visibility_threshold"]
        for pose in result.pose_landmarks:
            points = {
                i: (int(lm.x * w), int(lm.y * h))
                for i, lm in enumerate(pose)
                if i in indices and (lm.visibility if lm.visibility is not None else 1.0) >= threshold
            }
            for a, b in POSE_CONNECTIONS:
                if a in points and b in points:
                    cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
            for point in points.values():
                cv2.circle(frame, point, 4, (0, 255, 0), -1)

    # ---------- 離開本頁 ----------

    def close(self) -> None:
        """釋放模型並停止接收相機錯誤；相機由第 1 頁管理，這裡不關閉。"""
        self.camera.remove_error_listener(self._forward_camera_error)
        with self._lock:
            landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
