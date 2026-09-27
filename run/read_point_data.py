"""骨架關鍵點資料儲存，供後續坐姿分析使用。

每次記錄是一個 session（開始記錄 ~ 結束記錄），session 資訊包含相機名稱與第 2 頁的參數。
每一幀每個人存成一列（寬表格），33 個關鍵點各有以下欄位，例如 left_shoulder_x：
- x, y       正規化影像座標（0~1，旋轉後、鏡像前）
- z          相對深度（以髖部為基準，越小越靠近鏡頭）
- vis, pres  可見度、存在機率（0~1）
- wx, wy, wz 世界座標（公尺，原點為髖部中心），計算角度建議用這組
畫面中沒有人的幀也會記錄一列（pose_index 與關鍵點為空），可用來分析離座時間。

兩種儲存方式介面相同：
    store = SqliteStore()   # data/records/posture.db（建議）
    store = CsvStore()      # data/records/<session_id>.csv + .json
    store.start_session(camera.camera_name, page.params.values())
    store.record(page.process_frame())   # 每幀呼叫，會先暫存、累積一批再寫入
    store.end_session()
    df = store.load_session(session_id)  # pandas DataFrame
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Self

import pandas as pd
from camera_setting import ErrorLog, ErrorReporter
from Skelenton_detection import DetectionFrame

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "records"

LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)
LANDMARK_FIELDS = ("x", "y", "z", "vis", "pres", "wx", "wy", "wz")
META_COLUMNS = ("session_id", "frame_index", "unix_time", "elapsed_s", "inference_ms", "pose_index")
LANDMARK_COLUMNS = tuple(f"{name}_{f}" for name in LANDMARK_NAMES for f in LANDMARK_FIELDS)
COLUMNS = META_COLUMNS + LANDMARK_COLUMNS


class StorageErrorCode(Enum):
    OPEN_FAILED = "無法開啟資料儲存位置"
    WRITE_FAILED = "寫入資料失敗"
    BUFFER_OVERFLOW = "暫存資料過多，已捨棄最舊的資料"
    NO_SESSION = "尚未開始記錄"
    SESSION_NOT_FOUND = "找不到該筆紀錄"
    READ_FAILED = "讀取資料失敗"


class PointDataStore(ErrorReporter, ABC):
    """儲存方式的共同介面：暫存、批次寫入、錯誤處理在這裡，子類別只負責實際讀寫。"""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        batch_size: int = 30,
        min_interval_s: float = 0.0,
        max_buffer: int = 10_000,
        error_log: ErrorLog | None = None,
    ) -> None:
        """
        batch_size：累積幾列寫入一次（30 列約為 30 FPS 下每秒寫一次）
        min_interval_s：兩次記錄的最短間隔，0 表示每幀都記錄；長時間記錄可設 0.2 等降低資料量
        max_buffer：寫入持續失敗時最多暫存幾列，超過捨棄最舊的，避免記憶體無限增加
        """
        super().__init__(error_log)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.min_interval_s = min_interval_s
        self.max_buffer = max_buffer
        self.session_id: str | None = None
        self.frame_count = 0
        self._lock = threading.Lock()
        self._buffer: list[tuple] = []
        self._started = 0.0
        self._last_record = -math.inf
        self._write_failing = False
        self._no_session_reported = False

    # ---------- session ----------

    @property
    def recording(self) -> bool:
        return self.session_id is not None

    @property
    def elapsed_s(self) -> float:
        """目前這段記錄已經過的秒數，未記錄時為 0。"""
        return time.time() - self._started if self.recording else 0.0

    def start_session(self, camera: str = "", params: dict[str, Any] | None = None) -> str | None:
        """開始記錄，回傳 session_id；失敗回傳 None。若已在記錄中會先結束上一段。"""
        if self.recording:
            self.end_session()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        meta = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ended_at": None,
            "camera": camera,
            "frame_count": 0,
            "params": params or {},
        }
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._open_session(session_id, meta)
        except Exception as e:
            self._report(StorageErrorCode.OPEN_FAILED, f"{self.data_dir}：{e}")
            return None
        with self._lock:
            self.session_id = session_id
            self.frame_count = 0
            self._buffer.clear()
            self._started = time.time()
            self._last_record = -math.inf
            self._write_failing = False
            self._no_session_reported = False
        return session_id

    def end_session(self) -> None:
        """寫入剩餘資料並結束記錄。"""
        if not self.recording:
            return
        self.flush()
        try:
            self._close_session(self.session_id, datetime.now().isoformat(timespec="seconds"),
                                self.frame_count)
        except Exception as e:
            self._report(StorageErrorCode.WRITE_FAILED, f"結束記錄失敗：{e}")
        self.session_id = None

    # ---------- 記錄 ----------

    def record(self, frame: DetectionFrame) -> bool:
        """記錄一幀。失敗的幀（ok=False）與未達 min_interval_s 的幀不記錄，回傳 False。"""
        if not self.recording:
            if not self._no_session_reported:
                self._report(StorageErrorCode.NO_SESSION)
                self._no_session_reported = True
            return False
        if not frame.ok:
            return False
        now = time.time()
        if now - self._last_record < self.min_interval_s:
            return False
        with self._lock:
            self._last_record = now
            self._buffer.extend(self._to_rows(frame, now))
            self.frame_count += 1
            full = len(self._buffer) >= self.batch_size
        if full:
            self.flush()
        return True

    def _to_rows(self, frame: DetectionFrame, now: float) -> list[tuple]:
        meta = (self.session_id, self.frame_count, now, round(now - self._started, 3),
                round(frame.inference_ms, 2))
        poses = frame.result.pose_landmarks if frame.result is not None else []
        if not poses:
            return [(*meta, None) + (None,) * len(LANDMARK_COLUMNS)]
        worlds = frame.result.pose_world_landmarks
        rows = []
        for pose_index, pose in enumerate(poses):
            world = worlds[pose_index] if pose_index < len(worlds) else [None] * len(pose)
            values = []
            for lm, wlm in zip(pose, world, strict=False):
                values += [lm.x, lm.y, lm.z, lm.visibility, lm.presence]
                values += [wlm.x, wlm.y, wlm.z] if wlm is not None else [None] * 3
            rows.append((*meta, pose_index, *values))
        return rows

    def flush(self) -> bool:
        """把暫存資料寫入。寫入失敗會保留資料等下次重試，連續失敗只記錄一次。"""
        with self._lock:
            rows, self._buffer = self._buffer, []
        if not rows or not self.recording:
            return True
        try:
            self._write_rows(rows)
        except Exception as e:
            with self._lock:
                self._buffer = rows + self._buffer
                overflow = len(self._buffer) - self.max_buffer
                if overflow > 0:
                    del self._buffer[:overflow]
            if not self._write_failing:
                self._report(StorageErrorCode.WRITE_FAILED, str(e))
                self._write_failing = True
            if overflow > 0:
                self._report(StorageErrorCode.BUFFER_OVERFLOW, f"捨棄 {overflow} 列")
            return False
        self._write_failing = False
        return True

    # ---------- 讀取 ----------

    def list_sessions(self) -> pd.DataFrame:
        """所有記錄的摘要（session_id、開始 / 結束時間、相機、幀數、參數）。"""
        try:
            return self._list_sessions()
        except Exception as e:
            self._report(StorageErrorCode.READ_FAILED, str(e))
            return pd.DataFrame()

    def load_session(self, session_id: str) -> pd.DataFrame:
        """讀取一段記錄的所有關鍵點資料；找不到時回傳空的 DataFrame。"""
        if session_id == self.session_id:
            self.flush()  # 記錄中也能讀到最新資料
        try:
            df = self._load_session(session_id)
        except Exception as e:
            self._report(StorageErrorCode.READ_FAILED, str(e))
            return pd.DataFrame(columns=COLUMNS)
        if df is None:
            self._report(StorageErrorCode.SESSION_NOT_FOUND, session_id)
            return pd.DataFrame(columns=COLUMNS)
        return df

    # ---------- 子類別實作 ----------

    @abstractmethod
    def _open_session(self, session_id: str, meta: dict[str, Any]) -> None: ...

    @abstractmethod
    def _write_rows(self, rows: list[tuple]) -> None: ...

    @abstractmethod
    def _close_session(self, session_id: str, ended_at: str, frame_count: int) -> None: ...

    @abstractmethod
    def _list_sessions(self) -> pd.DataFrame: ...

    @abstractmethod
    def _load_session(self, session_id: str) -> pd.DataFrame | None: ...

    # ---------- 關閉 ----------

    def close(self) -> None:
        self.end_session()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SqliteStore(PointDataStore):
    """所有記錄存在同一個 SQLite 檔（sessions、frames 兩張表），可直接用 SQL 查詢。"""

    def __init__(self, data_dir: Path = DATA_DIR, filename: str = "posture.db", **kwargs) -> None:
        super().__init__(data_dir, **kwargs)
        self.db_path = self.data_dir / filename
        self._conn: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()  # 介面執行緒與影像執行緒可能共用同一個連線

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            landmark_defs = ", ".join(f"{c} REAL" for c in LANDMARK_COLUMNS)
            conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT,
                    camera TEXT, frame_count INTEGER, params TEXT);
                CREATE TABLE IF NOT EXISTS frames (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    frame_index INTEGER NOT NULL, unix_time REAL, elapsed_s REAL,
                    inference_ms REAL, pose_index INTEGER, {landmark_defs});
                CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id, frame_index);
            """)
            self._conn = conn
        return self._conn

    def _open_session(self, session_id: str, meta: dict[str, Any]) -> None:
        conn = self._connect()
        with self._db_lock, conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, NULL, ?, 0, ?)",
                (session_id, meta["started_at"], meta["camera"],
                 json.dumps(meta["params"], ensure_ascii=False)),
            )

    def _write_rows(self, rows: list[tuple]) -> None:
        conn = self._connect()
        placeholders = ", ".join("?" * len(COLUMNS))
        with self._db_lock, conn:
            conn.executemany(f"INSERT INTO frames ({', '.join(COLUMNS)}) VALUES ({placeholders})", rows)

    def _close_session(self, session_id: str, ended_at: str, frame_count: int) -> None:
        conn = self._connect()
        with self._db_lock, conn:
            conn.execute("UPDATE sessions SET ended_at = ?, frame_count = ? WHERE session_id = ?",
                         (ended_at, frame_count, session_id))

    def _list_sessions(self) -> pd.DataFrame:
        conn = self._connect()
        with self._db_lock:
            df = pd.read_sql_query("SELECT * FROM sessions ORDER BY started_at", conn)
        df["params"] = df["params"].map(json.loads)
        return df

    def _load_session(self, session_id: str) -> pd.DataFrame | None:
        conn = self._connect()
        with self._db_lock:
            exists = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if not exists:
                return None
            return pd.read_sql_query(
                "SELECT * FROM frames WHERE session_id = ? ORDER BY frame_index, pose_index",
                conn, params=(session_id,),
            )

    def close(self) -> None:
        super().close()
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class CsvStore(PointDataStore):
    """每段記錄一個 CSV（關鍵點）加一個 JSON（session 資訊），可直接用 Excel 開啟。"""

    def __init__(self, data_dir: Path = DATA_DIR, **kwargs) -> None:
        super().__init__(data_dir, **kwargs)
        self._file = None
        self._writer = None

    def _paths(self, session_id: str) -> tuple[Path, Path]:
        return self.data_dir / f"{session_id}.csv", self.data_dir / f"{session_id}.json"

    def _open_session(self, session_id: str, meta: dict[str, Any]) -> None:
        csv_path, json_path = self._paths(session_id)
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        # utf-8-sig：讓 Excel 正確辨識編碼
        self._file = csv_path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        self._file.flush()

    def _write_rows(self, rows: list[tuple]) -> None:
        self._writer.writerows(rows)
        self._file.flush()

    def _close_session(self, session_id: str, ended_at: str, frame_count: int) -> None:
        if self._file is not None:
            self._file.close()
            self._file = self._writer = None
        _, json_path = self._paths(session_id)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta.update(ended_at=ended_at, frame_count=frame_count)
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _list_sessions(self) -> pd.DataFrame:
        metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.data_dir.glob("*.json"))]
        return pd.DataFrame(metas, columns=["session_id", "started_at", "ended_at", "camera",
                                            "frame_count", "params"])

    def _load_session(self, session_id: str) -> pd.DataFrame | None:
        csv_path, _ = self._paths(session_id)
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"session_id": str})
