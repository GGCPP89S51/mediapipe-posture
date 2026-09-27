"""相機參數設定。

所有相機操作都不會拋出例外：失敗時回傳 False / None，並把錯誤同時寫入
- NowErrorLog：目前尚未處理的錯誤，介面據此跳出「請檢查設備」彈窗，處理完呼叫 clear()
- ErrorLog：程式執行以來的所有錯誤（總紀錄），同時寫入 logs/camera_error.log
"""
from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Self

import cv2
import numpy as np

try:  # 取得 DirectShow 裝置名稱（僅 Windows），裝置順序與 cv2.CAP_DSHOW 的編號一致
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class CameraErrorCode(Enum):
    NO_CAMERA_FOUND = "找不到任何相機"
    INVALID_INDEX = "相機編號不在可用清單中"
    NOT_SELECTED = "尚未選擇相機"
    OPEN_FAILED = "相機開啟失敗"
    NOT_OPENED = "相機尚未開啟"
    READ_FAILED = "讀取影像失敗"
    UNEXPECTED = "未預期的錯誤"


@dataclass(frozen=True)
class ErrorRecord:
    code: Enum  # 各頁自己的錯誤代碼，例如 CameraErrorCode、DetectionErrorCode
    detail: str = ""
    camera_index: int | None = None
    time: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        cam = f"[相機 {self.camera_index}] " if self.camera_index is not None else ""
        detail = f"：{self.detail}" if self.detail else ""
        return f"{self.time:%Y-%m-%d %H:%M:%S} {cam}{self.code.value}{detail}"


class NowErrorLog:
    """目前尚未處理的錯誤。介面顯示彈窗後呼叫 clear()。"""

    def __init__(self) -> None:
        self._records: list[ErrorRecord] = []
        self._lock = threading.Lock()

    def add(self, record: ErrorRecord) -> None:
        with self._lock:
            self._records.append(record)

    def has_errors(self) -> bool:
        with self._lock:
            return bool(self._records)

    def records(self) -> list[ErrorRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class ErrorLog:
    """所有錯誤的總紀錄，只增不減，並同步寫入檔案。"""

    def __init__(self, log_file: Path | None = LOG_DIR / "camera_error.log") -> None:
        self._records: list[ErrorRecord] = []
        self._lock = threading.Lock()
        self._logger = logging.getLogger("camera_error")
        if log_file is not None and not self._logger.handlers:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.ERROR)

    def add(self, record: ErrorRecord) -> None:
        with self._lock:
            self._records.append(record)
        self._logger.error(str(record))

    def records(self) -> list[ErrorRecord]:
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


class ErrorReporter:
    """各頁共用的錯誤回報：每頁有自己的 NowErrorLog，ErrorLog（總紀錄）可以共用。"""

    def __init__(self, error_log: ErrorLog | None = None) -> None:
        self.now_error_log = NowErrorLog()
        self.error_log = error_log if error_log is not None else ErrorLog()
        self._listeners: list[Callable[[ErrorRecord], None]] = []

    def add_error_listener(self, callback: Callable[[ErrorRecord], None]) -> None:
        """註冊錯誤通知（例如介面跳出彈窗）。callback 內的例外不會中斷流程。"""
        self._listeners.append(callback)

    def remove_error_listener(self, callback: Callable[[ErrorRecord], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _context_index(self) -> int | None:
        """錯誤紀錄上標示的相機編號，子類別可覆寫。"""
        return None

    def _report(self, code: Enum, detail: str = "", index: int | None = None) -> None:
        record = ErrorRecord(code, detail, self._context_index() if index is None else index)
        self.now_error_log.add(record)
        self.error_log.add(record)
        for callback in list(self._listeners):
            try:
                callback(record)
            except Exception as e:  # 介面端出錯也不能讓流程中斷
                self.error_log.add(ErrorRecord(CameraErrorCode.UNEXPECTED, f"錯誤通知失敗：{e}"))


class CameraSetting(ErrorReporter):
    """選擇並開啟相機。

    用法：
        cam = CameraSetting()
        cam.add_error_listener(lambda rec: 顯示彈窗(rec))  # 可選：錯誤發生時立即通知介面
        cameras = cam.scan_cameras()  # {0: "USB2.0 HD UVC WebCam", ...}，可直接做成下拉選單
        if cameras and cam.select_camera(0) and cam.open():
            ok, frame = cam.read_frame()
    """

    # Windows 用 DirectShow 開啟較快，且可避免 MSMF 的長時間等待
    DEFAULT_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

    def __init__(
        self,
        max_scan: int = 5,
        backend: int = DEFAULT_BACKEND,
        error_log: ErrorLog | None = None,
    ) -> None:
        super().__init__(error_log)
        self.max_scan = max_scan
        self.backend = backend
        self.available_cameras: dict[int, str] = {}  # 編號 -> 裝置名稱
        self.camera_index: int | None = None
        self._cap: cv2.VideoCapture | None = None
        self._read_failing = False

    def _context_index(self) -> int | None:
        return self.camera_index

    # ---------- 選擇相機 ----------

    def _device_names(self) -> list[str]:
        """依編號順序回傳裝置名稱；非 DirectShow 或取不到時回傳空清單。"""
        if FilterGraph is None or self.backend != cv2.CAP_DSHOW:
            return []
        try:
            return FilterGraph().get_input_devices()
        except Exception:
            return []  # 取不到名稱不影響掃描，改用「相機 N」顯示

    def scan_cameras(self) -> dict[int, str]:
        """掃描 0 ~ max_scan-1 號相機，回傳 {編號: 名稱}（找不到時為空 dict）。"""
        names = self._device_names()
        found: dict[int, str] = {}
        # 探測不存在的編號時 OpenCV 會印出 WARN，掃描期間暫時關閉
        prev_level = cv2.getLogLevel()
        cv2.setLogLevel(2)  # 2 = ERROR
        for index in range(self.max_scan):
            if index == self.camera_index and self.is_opened:
                # 使用中的相機無法重複開啟，直接視為可用
                found[index] = self.available_cameras.get(index, f"相機 {index}")
                continue
            try:
                cap = cv2.VideoCapture(index, self.backend)
                if cap.isOpened():
                    found[index] = names[index] if index < len(names) else f"相機 {index}"
                cap.release()
            except Exception:
                pass  # 單一編號探測失敗不算錯誤，繼續掃描下一個
        cv2.setLogLevel(prev_level)
        self.available_cameras = found
        if not found:
            self._report(CameraErrorCode.NO_CAMERA_FOUND, f"已掃描編號 0~{self.max_scan - 1}")
        return found

    def select_camera(self, index: int) -> bool:
        """選擇相機。若原本已開啟其他相機，會先關閉。"""
        if index not in self.available_cameras:
            self._report(
                CameraErrorCode.INVALID_INDEX, f"可用清單：{list(self.available_cameras)}", index
            )
            return False
        if index != self.camera_index:
            self.close()
        self.camera_index = index
        return True

    # ---------- 開啟 / 關閉 ----------

    def open(self) -> bool:
        if self.camera_index is None:
            self._report(CameraErrorCode.NOT_SELECTED)
            return False
        if self.is_opened:
            return True
        try:
            cap = cv2.VideoCapture(self.camera_index, self.backend)
            if not cap.isOpened():
                cap.release()
                self._report(CameraErrorCode.OPEN_FAILED, "請確認相機已連接且未被其他程式占用")
                return False
        except Exception as e:
            self._report(CameraErrorCode.UNEXPECTED, str(e))
            return False
        self._cap = cap
        self._read_failing = False
        return True

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                self._report(CameraErrorCode.UNEXPECTED, f"關閉相機失敗：{e}")
            self._cap = None

    @property
    def camera_name(self) -> str | None:
        return self.available_cameras.get(self.camera_index)

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_property(self, prop: int) -> float | None:
        """讀取 OpenCV 相機屬性（cv2.CAP_PROP_*）；相機未開啟時回傳 None。"""
        if not self.is_opened:
            return None
        try:
            return self._cap.get(prop)
        except Exception:
            return None

    def set_property(self, prop: int, value: float) -> bool:
        """設定 OpenCV 相機屬性。回傳相機是否接受；實際值請用 get_property 讀回確認。"""
        if not self.is_opened:
            self._report(CameraErrorCode.NOT_OPENED)
            return False
        try:
            return bool(self._cap.set(prop, value))
        except Exception as e:
            self._report(CameraErrorCode.UNEXPECTED, f"設定相機屬性失敗：{e}")
            return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """讀取一張 BGR 影像。

        連續讀取失敗（例如相機被拔除）只會記錄第一次，避免每幀都寫入錯誤。
        """
        if not self.is_opened:
            self._report(CameraErrorCode.NOT_OPENED)
            return False, None
        try:
            ok, frame = self._cap.read()
        except Exception as e:
            ok, frame = False, None
            detail = str(e)
        else:
            detail = "相機可能已被拔除或斷線"
        if not ok:
            if not self._read_failing:
                self._report(CameraErrorCode.READ_FAILED, detail)
            self._read_failing = True
            return False, None
        self._read_failing = False
        return True, frame

    # ---------- context manager ----------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
