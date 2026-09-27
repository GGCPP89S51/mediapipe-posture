"""取得目前（最新一幀）的關鍵點數值。

每幀把第 2 頁的 DetectionFrame 交給 NowPointGetter.update()，任何地方都可以用 latest 取得
最新的 KeypointSnapshot；介面的「目前關鍵點數值」視窗與之後的辨識模組都從這裡讀資料。

    getter = NowPointGetter()
    getter.update(page.process_frame())
    snap = getter.latest
    if snap.detected:
        shoulder = snap.get("left_shoulder")   # 也可用編號 snap.get(11)
        shoulder.cx, shoulder.cy               # 中心座標（像素，畫面中心為原點、y 向上）
        shoulder.wx, shoulder.wy               # 世界座標（公尺）

三種座標：
- cx, cy     中心座標（像素）：畫面中心為 (0, 0)，x 向右、y 向上為正，四捨五入到小數點後 2 位
- x, y, z    正規化影像座標：左上角為 (0, 0)，右下角為 (1, 1)
- wx, wy, wz 世界座標（公尺）：原點為髖部中心，為 MediaPipe 的估計值

初始點：使用者坐正時呼叫 set_baseline()，把當下的關鍵點存成 baseline，供辨識時比較偏移。
    getter.displacement("nose")          # (+dx, +dy)：目前相對初始點的位移（像素）

左右以「被拍攝者本人」的左右為準（MediaPipe 的定義），與畫面是否鏡像無關；
座標以鏡像前的影像計算，因此開啟鏡像顯示時，畫面上的左右與 cx 的正負相反。
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from read_point_data import LANDMARK_NAMES
from Skelenton_detection import DetectionFrame

LANDMARK_NAMES_ZH = (
    "鼻子", "左眼內側", "左眼", "左眼外側", "右眼內側", "右眼", "右眼外側", "左耳", "右耳",
    "嘴巴左側", "嘴巴右側", "左肩", "右肩", "左手肘", "右手肘", "左手腕", "右手腕",
    "左小指", "右小指", "左食指", "右食指", "左拇指", "右拇指", "左髖", "右髖",
    "左膝", "右膝", "左腳踝", "右腳踝", "左腳跟", "右腳跟", "左腳尖", "右腳尖",
)
# 坐姿判斷常用的上半身關鍵點：鼻子、左右耳、左右肩
POSTURE_KEY_POINTS = (0, 7, 8, 11, 12)


@dataclass(frozen=True)
class KeypointValue:
    index: int
    name: str  # 英文名稱，與資料庫欄位前綴相同，例如 left_shoulder
    name_zh: str
    x: float  # 正規化影像座標 0~1（旋轉後、鏡像前）
    y: float
    z: float  # 相對深度，越小越靠近鏡頭
    visibility: float  # 可見度 0~1
    presence: float  # 存在機率 0~1
    wx: float | None  # 世界座標（公尺，原點為髖部中心）
    wy: float | None
    wz: float | None
    cx: float | None = None  # 中心座標（像素），畫面中心為原點、y 向上；影像大小未知時為 None
    cy: float | None = None


@dataclass(frozen=True)
class KeypointSnapshot:
    """某一幀的關鍵點數值；沒有偵測到人時 points 為空。"""

    timestamp: float = 0.0  # time.time()
    image_size: tuple[int, int] | None = None  # (寬, 高)，中心座標以此換算
    num_poses: int = 0  # 這一幀偵測到幾個人
    pose_index: int | None = None  # points 屬於第幾個人
    points: tuple[KeypointValue, ...] = ()
    _by_name: dict[str, KeypointValue] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._by_name.update({p.name: p for p in self.points})

    @property
    def detected(self) -> bool:
        return bool(self.points)

    def get(self, key: str | int) -> KeypointValue | None:
        """用英文名稱或編號（0~32）取得關鍵點；不存在時回傳 None。"""
        if isinstance(key, int):
            return next((p for p in self.points if p.index == key), None)
        return self._by_name.get(key)

    def to_dict(self) -> dict[str, dict[str, float | None]]:
        """{名稱: {x, y, z, visibility, presence, wx, wy, wz}}，方便轉成 JSON 或 DataFrame。"""
        fields = ("cx", "cy", "x", "y", "z", "visibility", "presence", "wx", "wy", "wz")
        return {p.name: {f: getattr(p, f) for f in fields} for p in self.points}


PIXEL_FIELDS = ("cx", "cy")


def displacement(
    current: KeypointSnapshot | None,
    baseline: KeypointSnapshot | None,
    key: str | int,
    fields: tuple[str, ...] = PIXEL_FIELDS,
) -> tuple[float, ...] | None:
    """某個關鍵點目前相對初始點的位移（目前 - 初始點），fields 指定座標，預設為中心座標（像素）。

    例如 (+12.30, -8.05) 表示比坐正時往右 12.30 px、往下 8.05 px；中心座標取到小數點後 2 位。
    任一方沒有這個點或數值為 None 時回傳 None。
    """
    if current is None or baseline is None:
        return None
    now, base = current.get(key), baseline.get(key)
    if now is None or base is None:
        return None
    result = []
    for f in fields:
        a, b = getattr(now, f), getattr(base, f)
        if a is None or b is None:
            return None
        result.append(round(a - b, 2) if f in PIXEL_FIELDS else a - b)
    return tuple(result)


def to_center_pixels(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """正規化影像座標 -> 中心座標（像素）：畫面中心為 (0, 0)，x 向右、y 向上為正，取到小數點後 2 位。"""
    return round((x - 0.5) * width, 2), round((0.5 - y) * height, 2)


class NowPointGetter:
    """保存最新一幀的關鍵點。update() 在影像迴圈呼叫，latest 可在任何執行緒讀取。"""

    def __init__(self, pose_index: int = 0, indices: Iterable[int] | None = None) -> None:
        """
        pose_index：多人時取第幾個人（0 = 第一個）
        indices：只保留這些編號的關鍵點，None 表示全部 33 點
        """
        self.pose_index = pose_index
        self.indices = frozenset(indices) if indices is not None else None
        self._latest = KeypointSnapshot()
        self._baseline: KeypointSnapshot | None = None
        self._lock = threading.Lock()

    @property
    def latest(self) -> KeypointSnapshot:
        with self._lock:
            return self._latest

    def get(self, key: str | int) -> KeypointValue | None:
        return self.latest.get(key)

    @property
    def baseline(self) -> KeypointSnapshot | None:
        """初始點（使用者坐正時的關鍵點），尚未設定時為 None；辨識時可與 latest 比較。"""
        with self._lock:
            return self._baseline

    def set_baseline(self) -> KeypointSnapshot | None:
        """把目前這一幀設為初始點。畫面中沒有人時不設定，回傳 None。"""
        with self._lock:
            if not self._latest.detected:
                return None
            self._baseline = self._latest
            return self._baseline

    def low_visibility_points(self, snapshot: KeypointSnapshot | None = None) -> list[KeypointValue]:
        """坐姿判斷常用的上半身關鍵點中，可見度偏低（< 0.5）的點；用來提醒使用者調整位置。"""
        snap = snapshot if snapshot is not None else self.latest
        points = (snap.get(i) for i in POSTURE_KEY_POINTS)
        return [p for p in points if p is None or p.visibility < 0.5]

    def displacement(self, key: str | int, fields: tuple[str, ...] = PIXEL_FIELDS) -> tuple[float, ...] | None:
        """最新一幀相對初始點的位移；尚未設定初始點或沒有偵測到人時回傳 None。"""
        return displacement(self.latest, self.baseline, key, fields)

    def clear_baseline(self) -> None:
        with self._lock:
            self._baseline = None

    def reset(self) -> None:
        """清除數值與初始點（例如離開第 2 頁、可能更換相機時）。"""
        with self._lock:
            self._latest = KeypointSnapshot()
            self._baseline = None

    def update(self, frame: DetectionFrame) -> KeypointSnapshot:
        """用新的一幀更新。讀取失敗的幀（ok=False）不更新，保留上一次的數值。"""
        if not frame.ok:
            return self.latest
        snapshot = self._to_snapshot(frame)
        with self._lock:
            self._latest = snapshot
        return snapshot

    def _to_snapshot(self, frame: DetectionFrame) -> KeypointSnapshot:
        now = time.time()
        # frame.frame 為旋轉後的影像，鏡像不改變大小，因此可直接用來換算中心座標
        size = (frame.frame.shape[1], frame.frame.shape[0]) if frame.frame is not None else None
        poses = frame.result.pose_landmarks if frame.result is not None else []
        if self.pose_index >= len(poses):
            return KeypointSnapshot(now, size, len(poses))
        pose = poses[self.pose_index]
        worlds = frame.result.pose_world_landmarks
        world = worlds[self.pose_index] if self.pose_index < len(worlds) else [None] * len(pose)
        points = tuple(
            KeypointValue(
                i, LANDMARK_NAMES[i], LANDMARK_NAMES_ZH[i], lm.x, lm.y, lm.z,
                lm.visibility if lm.visibility is not None else 1.0,
                lm.presence if lm.presence is not None else 1.0,
                *((w.x, w.y, w.z) if w is not None else (None, None, None)),
                *(to_center_pixels(lm.x, lm.y, *size) if size else (None, None)),
            )
            for i, (lm, w) in enumerate(zip(pose, world, strict=False))
            if self.indices is None or i in self.indices
        )
        return KeypointSnapshot(now, size, len(poses), self.pose_index, points)
