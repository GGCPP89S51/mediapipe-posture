"""辨識規則：用關鍵點數值定義姿勢條件，判斷目前姿勢是否符合（例如低頭、駝背、肩膀傾斜）。

一條 Rule 由一個或多個 Condition 組成：
- 指標（Metric）：點的位置／位移、兩點距離、兩點連線角度、三點夾角、畫面中沒有人
- 比較基準（Reference）：目前數值、與初始點的差、與初始點的比例（%）
- 範圍：下限 / 上限（可不限），在區間內或區間外時條件成立
條件依 match（全部 / 任一）組合，持續成立 hold_s 秒才算「觸發」，避免短暫動作造成誤判。

座標一律使用中心座標（像素，畫面中心為原點、y 向上為正），角度單位為度。
除了 33 個關鍵點，也可選擇虛擬點：兩肩中點、兩耳中點、兩髖中點。

    engine = RuleEngine()                        # 自動載入 data/rules.json
    rule = engine.add(preset_rules()[0])         # 或自行建立 Rule / Condition
    for result in engine.evaluate(getter.latest, getter.baseline):
        if result.status is RuleStatus.TRIGGERED:
            print(result.rule.name, result.rule.message)
    engine.save()

辨識期間用 RecognitionTracker 統計每條規則的觸發次數與累計秒數。
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from camera_setting import ErrorLog, ErrorReporter
from now_point_get import LANDMARK_NAMES_ZH, KeypointSnapshot
from read_point_data import LANDMARK_NAMES

RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "rules.json"
MIN_VISIBILITY = 0.5  # 可見度低於此值的點不參與判斷

VIRTUAL_POINTS = {
    "mid_shoulder": ("兩肩中點", ("left_shoulder", "right_shoulder")),
    "mid_ear": ("兩耳中點", ("left_ear", "right_ear")),
    "mid_hip": ("兩髖中點", ("left_hip", "right_hip")),
}
# 介面下拉選單用：(名稱, 中文)，虛擬點排在前面
POINT_CHOICES = [(k, v[0]) for k, v in VIRTUAL_POINTS.items()] + list(
    zip(LANDMARK_NAMES, LANDMARK_NAMES_ZH, strict=True))
POINT_LABELS = dict(POINT_CHOICES)


class Metric(Enum):
    POSITION = "點的位置／位移"
    DISTANCE = "兩點距離"
    LINE_ANGLE = "兩點連線角度"
    JOINT_ANGLE = "三點夾角"
    NO_PERSON = "畫面中沒有人"


class Axis(Enum):
    X = "水平 x（向右為正）"
    Y = "垂直 y（向上為正）"
    DIST = "直線距離"


class Reference(Enum):
    ABSOLUTE = "目前數值"
    DIFF = "與初始點的差"
    RATIO = "與初始點的比例（%）"


class RangeMode(Enum):
    INSIDE = "在區間內"
    OUTSIDE = "在區間外"


# 各指標需要的點（介面標籤）、預設點、可用的比較基準、說明
METRIC_POINT_LABELS = {
    Metric.POSITION: ("關鍵點",),
    Metric.DISTANCE: ("點 A", "點 B"),
    Metric.LINE_ANGLE: ("起點", "終點"),
    Metric.JOINT_ANGLE: ("端點 A", "頂點", "端點 C"),
    Metric.NO_PERSON: (),
}
METRIC_DEFAULT_POINTS = {
    Metric.POSITION: ["nose"],
    Metric.DISTANCE: ["left_shoulder", "right_shoulder"],
    Metric.LINE_ANGLE: ["right_shoulder", "left_shoulder"],
    Metric.JOINT_ANGLE: ["left_ear", "left_shoulder", "left_hip"],
    Metric.NO_PERSON: [],
}
METRIC_REFERENCES = {
    Metric.POSITION: (Reference.DIFF, Reference.ABSOLUTE),
    Metric.DISTANCE: (Reference.ABSOLUTE, Reference.DIFF, Reference.RATIO),
    Metric.LINE_ANGLE: (Reference.ABSOLUTE, Reference.DIFF),
    Metric.JOINT_ANGLE: (Reference.ABSOLUTE, Reference.DIFF),
    Metric.NO_PERSON: (Reference.ABSOLUTE,),
}
METRIC_HELP = {
    Metric.POSITION: "與初始點的差即「位移」，例如鼻子垂直位移 ≤ -30 px 代表頭比坐正時低 30 px 以上",
    Metric.DISTANCE: "兩點之間的距離；與初始點比例 > 100% 代表距離變大（例如靠近鏡頭時兩肩距離變大）",
    Metric.LINE_ANGLE: "起點到終點連線與水平線的角度（-180° ~ 180°），0° 為水平，例如右肩→左肩可判斷肩膀傾斜",
    Metric.JOINT_ANGLE: "以頂點為中心、兩端點形成的夾角（0° ~ 180°），180° 為一直線",
    Metric.NO_PERSON: "畫面中偵測不到人時成立，搭配持續時間可判斷離座",
}


@dataclass
class Condition:
    metric: Metric = Metric.POSITION
    points: list[str] = field(default_factory=lambda: ["nose"])
    axis: Axis = Axis.Y  # 只有「點的位置／位移」使用
    reference: Reference = Reference.DIFF
    low: float | None = None  # 下限，None 表示不限
    high: float | None = -30.0  # 上限，None 表示不限
    mode: RangeMode = RangeMode.INSIDE

    def set_metric(self, metric: Metric) -> None:
        """切換指標，並把點與比較基準重設為該指標的預設值。"""
        self.metric = metric
        self.points = list(METRIC_DEFAULT_POINTS[metric])
        self.reference = METRIC_REFERENCES[metric][0]

    @property
    def unit(self) -> str:
        if self.metric is Metric.NO_PERSON:
            return ""
        if self.reference is Reference.RATIO:
            return "%"
        return "°" if self.metric in (Metric.LINE_ANGLE, Metric.JOINT_ANGLE) else "px"

    def value_label(self) -> str:
        """被比較的數值名稱，例如「鼻子 垂直位移」。"""
        names = [POINT_LABELS.get(p, p) for p in self.points]
        if self.metric is Metric.NO_PERSON:
            return "畫面中沒有人"
        if self.metric is Metric.POSITION:
            axis = {Axis.X: "水平", Axis.Y: "垂直", Axis.DIST: "直線"}[self.axis]
            kind = "位移" if self.reference is Reference.DIFF else "位置"
            if self.axis is Axis.DIST and self.reference is Reference.ABSOLUTE:
                return f"{names[0]} 與畫面中心的距離"
            return f"{names[0]} {axis}{kind}"
        if self.metric is Metric.DISTANCE:
            text = f"{names[0]}—{names[1]} 距離"
        elif self.metric is Metric.LINE_ANGLE:
            text = f"{names[0]}→{names[1]} 連線角度"
        else:
            text = f"{names[0]}-{names[1]}-{names[2]} 夾角"
        suffix = {Reference.ABSOLUTE: "", Reference.DIFF: "變化量", Reference.RATIO: "比例"}
        return text + suffix[self.reference]

    def range_text(self) -> str:
        if self.metric is Metric.NO_PERSON:
            return ""
        u = self.unit
        low, high = self.low, self.high
        if self.mode is RangeMode.INSIDE:
            if low is not None and high is not None:
                return f"介於 {low:g} ~ {high:g} {u}"
            if high is not None:
                return f"≤ {high:g} {u}"
            if low is not None:
                return f"≥ {low:g} {u}"
            return "任何數值"
        if low is not None and high is not None:
            return f"< {low:g} 或 > {high:g} {u}"
        if high is not None:
            return f"> {high:g} {u}"
        if low is not None:
            return f"< {low:g} {u}"
        return "永不成立"

    def describe(self) -> str:
        return f"{self.value_label()} {self.range_text()}".strip()

    def in_range(self, value: float) -> bool:
        inside = (self.low is None or value >= self.low) and (self.high is None or value <= self.high)
        if self.mode is RangeMode.INSIDE:
            return inside
        return (self.low is not None and value < self.low) or (self.high is not None and value > self.high)

    def validate(self) -> list[str]:
        errors = []
        if len(self.points) != len(METRIC_POINT_LABELS[self.metric]):
            errors.append(f"{self.metric.value} 需要 {len(METRIC_POINT_LABELS[self.metric])} 個點")
        errors += [f"未知的關鍵點：{p}" for p in self.points if p not in POINT_LABELS]
        if self.reference not in METRIC_REFERENCES[self.metric]:
            errors.append(f"{self.metric.value} 不能使用「{self.reference.value}」")
        return errors

    def warnings(self) -> list[str]:
        """不影響存檔、但可能不是使用者本意的設定。"""
        if self.metric is Metric.NO_PERSON:
            return []
        if self.low is not None and self.high is not None and self.low > self.high:
            return ["下限大於上限，" + ("此條件永遠不會成立" if self.mode is RangeMode.INSIDE
                                        else "此條件永遠成立")]
        return []

    def to_dict(self) -> dict:
        return {"metric": self.metric.name, "points": list(self.points), "axis": self.axis.name,
                "reference": self.reference.name, "low": self.low, "high": self.high,
                "mode": self.mode.name}

    @classmethod
    def from_dict(cls, data: dict) -> Condition:
        return cls(Metric[data["metric"]], list(data.get("points", [])), Axis[data.get("axis", "Y")],
                   Reference[data.get("reference", "ABSOLUTE")], data.get("low"), data.get("high"),
                   RangeMode[data.get("mode", "INSIDE")])


@dataclass
class Rule:
    name: str = "新規則"
    conditions: list[Condition] = field(default_factory=list)
    match: str = "all"  # "all"：全部條件成立；"any"：任一條件成立
    hold_s: float = 3.0  # 持續成立幾秒才觸發
    message: str = ""  # 觸發時的提醒文字
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def needs_baseline(self) -> bool:
        return any(c.reference is not Reference.ABSOLUTE for c in self.conditions)

    def validate(self) -> list[str]:
        errors = [] if self.name.strip() else ["規則名稱不能空白"]
        if self.match not in ("all", "any"):
            errors.append(f"未知的判斷方式：{self.match}")
        for i, c in enumerate(self.conditions, start=1):
            errors += [f"條件 {i}：{e}" for e in c.validate()]
        return errors

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "enabled": self.enabled, "match": self.match,
                "hold_s": self.hold_s, "message": self.message,
                "conditions": [c.to_dict() for c in self.conditions]}

    @classmethod
    def from_dict(cls, data: dict) -> Rule:
        return cls(data["name"], [Condition.from_dict(c) for c in data.get("conditions", [])],
                   data.get("match", "all"), float(data.get("hold_s", 3.0)), data.get("message", ""),
                   bool(data.get("enabled", True)), data.get("id") or uuid.uuid4().hex[:8])


# ---------- 計算 ----------


class RuleStatus(Enum):
    DISABLED = "未啟用"
    NO_CONDITION = "尚未設定條件"
    NEED_BASELINE = "需要初始點"
    UNKNOWN = "無法判斷"
    NORMAL = "正常"
    PENDING = "條件成立中"
    TRIGGERED = "觸發"


@dataclass
class ConditionResult:
    condition: Condition
    value: float | None  # 計算出的數值（依 condition.unit）；無法計算時為 None
    met: bool | None  # 是否成立；無法判斷時為 None
    reason: str = ""  # 無法判斷的原因


@dataclass
class RuleResult:
    rule: Rule
    status: RuleStatus
    conditions: list[ConditionResult]
    held_s: float = 0.0  # 條件已持續成立的秒數

    @property
    def triggered(self) -> bool:
        return self.status is RuleStatus.TRIGGERED


def resolve_point(snap: KeypointSnapshot, name: str,
                  min_visibility: float = MIN_VISIBILITY) -> tuple[float, float] | str:
    """取得點的中心座標；虛擬點取平均。無法使用時回傳原因文字。"""
    names = VIRTUAL_POINTS[name][1] if name in VIRTUAL_POINTS else (name,)
    points = [snap.get(n) for n in names]
    label = POINT_LABELS.get(name, name)
    if any(p is None or p.cx is None for p in points):
        return f"{label}沒有資料"
    if any(p.visibility < min_visibility for p in points):
        return f"{label}可見度不足"
    return (sum(p.cx for p in points) / len(points), sum(p.cy for p in points) / len(points))


def _measure(cond: Condition, snap: KeypointSnapshot, min_vis: float) -> float | tuple | str:
    pts = []
    for name in cond.points:
        p = resolve_point(snap, name, min_vis)
        if isinstance(p, str):
            return p
        pts.append(p)
    if cond.metric is Metric.POSITION:
        return pts[0]
    if cond.metric is Metric.DISTANCE:
        (ax, ay), (bx, by) = pts
        return math.hypot(bx - ax, by - ay)
    if cond.metric is Metric.LINE_ANGLE:
        (ax, ay), (bx, by) = pts
        if (ax, ay) == (bx, by):
            return "兩點重疊，無法計算角度"
        return math.degrees(math.atan2(by - ay, bx - ax))
    (ax, ay), (bx, by), (cx, cy) = pts  # JOINT_ANGLE，頂點為 b
    v1, v2 = (ax - bx, ay - by), (cx - bx, cy - by)
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return "點重疊，無法計算夾角"
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _position_value(cond: Condition, dx: float, dy: float) -> float:
    return {Axis.X: dx, Axis.Y: dy, Axis.DIST: math.hypot(dx, dy)}[cond.axis]


def compute_value(cond: Condition, current: KeypointSnapshot, baseline: KeypointSnapshot | None,
                  min_visibility: float = MIN_VISIBILITY) -> tuple[float | None, str]:
    """計算條件的數值（取到小數點後 2 位）。回傳 (數值, 無法計算的原因)。"""
    if not current.detected:
        return None, "未偵測到人"
    if cond.reference is not Reference.ABSOLUTE and baseline is None:
        return None, "需要先設定初始點"
    now = _measure(cond, current, min_visibility)
    if isinstance(now, str):
        return None, now
    if cond.reference is Reference.ABSOLUTE:
        value = _position_value(cond, *now) if cond.metric is Metric.POSITION else now
        return round(value, 2), ""
    base = _measure(cond, baseline, 0.0)  # 初始點已在設定時檢查過可見度
    if isinstance(base, str):
        return None, f"初始點：{base}"
    if cond.metric is Metric.POSITION:
        value = _position_value(cond, now[0] - base[0], now[1] - base[1])
    elif cond.reference is Reference.RATIO:
        if base == 0:
            return None, "初始點數值為 0，無法計算比例"
        value = now / base * 100
    else:
        value = now - base
        if cond.metric is Metric.LINE_ANGLE:  # 角度差換算到 -180° ~ 180°
            value = (value + 180) % 360 - 180
    return round(value, 2), ""


def evaluate_condition(cond: Condition, current: KeypointSnapshot, baseline: KeypointSnapshot | None,
                       min_visibility: float = MIN_VISIBILITY) -> ConditionResult:
    if cond.metric is Metric.NO_PERSON:
        return ConditionResult(cond, None, not current.detected)
    value, reason = compute_value(cond, current, baseline, min_visibility)
    return ConditionResult(cond, value, None if value is None else cond.in_range(value), reason)


# ---------- 辨識統計 ----------


@dataclass
class RuleStats:
    trigger_count: int = 0  # 觸發次數（每次由未觸發變為觸發算一次）
    triggered_s: float = 0.0  # 累計觸發秒數
    last_triggered_at: float | None = None


class RecognitionTracker:
    """統計一次辨識（開始 ~ 結束）期間，每條規則觸發幾次、累計多久。

    每次 evaluate 後呼叫 update(results)；stats 以 rule.id 為鍵。
    """

    def __init__(self) -> None:
        self.stats: dict[str, RuleStats] = {}
        self.started_at: float | None = None
        self._triggered: dict[str, bool] = {}
        self._last_time: float | None = None

    def start(self, now: float | None = None) -> None:
        self.stats.clear()
        self._triggered.clear()
        self.started_at = self._last_time = time.time() if now is None else now

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None or self._last_time is None:
            return 0.0
        return self._last_time - self.started_at

    def update(self, results: list[RuleResult], now: float | None = None) -> None:
        now = time.time() if now is None else now
        dt = 0.0 if self._last_time is None else max(now - self._last_time, 0.0)
        for result in results:
            rid = result.rule.id
            stats = self.stats.setdefault(rid, RuleStats())
            was = self._triggered.get(rid, False)
            if was:  # 上一次到這一次之間處於觸發狀態
                stats.triggered_s = round(stats.triggered_s + dt, 3)
            if result.triggered and not was:
                stats.trigger_count += 1
                stats.last_triggered_at = now
            self._triggered[rid] = result.triggered
        self._last_time = now

    def get(self, rule_id: str) -> RuleStats:
        return self.stats.get(rule_id, RuleStats())


# ---------- 規則引擎 ----------


class RuleErrorCode(Enum):
    LOAD_FAILED = "讀取辨識規則失敗"
    SAVE_FAILED = "儲存辨識規則失敗"
    INVALID_RULE = "辨識規則內容有誤，已略過"


class RuleEngine(ErrorReporter):
    """管理規則、存檔，並依最新關鍵點判斷每條規則的狀態（含持續時間計時）。"""

    def __init__(self, path: Path | None = RULES_PATH, min_visibility: float = MIN_VISIBILITY,
                 error_log: ErrorLog | None = None, autoload: bool = True) -> None:
        super().__init__(error_log)
        self.path = Path(path) if path is not None else None
        self.min_visibility = min_visibility
        self.rules: list[Rule] = []
        self._met_since: dict[str, float] = {}
        if autoload:
            self.load()

    def add(self, rule: Rule) -> Rule:
        self.rules.append(rule)
        return rule

    def remove(self, rule_id: str) -> None:
        self.rules = [r for r in self.rules if r.id != rule_id]
        self._met_since.pop(rule_id, None)

    def get(self, rule_id: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rule_id), None)

    def reset_timers(self) -> None:
        self._met_since.clear()

    def evaluate(self, current: KeypointSnapshot, baseline: KeypointSnapshot | None,
                 now: float | None = None) -> list[RuleResult]:
        now = time.time() if now is None else now
        return [self.evaluate_rule(rule, current, baseline, now) for rule in self.rules]

    def evaluate_rule(self, rule: Rule, current: KeypointSnapshot, baseline: KeypointSnapshot | None,
                      now: float | None = None) -> RuleResult:
        now = time.time() if now is None else now
        if not rule.enabled or not rule.conditions:
            self._met_since.pop(rule.id, None)
            status = RuleStatus.DISABLED if not rule.enabled else RuleStatus.NO_CONDITION
            return RuleResult(rule, status, [])
        results = [evaluate_condition(c, current, baseline, self.min_visibility) for c in rule.conditions]
        if rule.needs_baseline and baseline is None:
            self._met_since.pop(rule.id, None)
            return RuleResult(rule, RuleStatus.NEED_BASELINE, results)
        mets = [r.met for r in results]
        if rule.match == "any":
            met = True if True in mets else (None if None in mets else False)
        else:
            met = False if False in mets else (None if None in mets else True)
        if not met:
            self._met_since.pop(rule.id, None)
            return RuleResult(rule, RuleStatus.UNKNOWN if met is None else RuleStatus.NORMAL, results)
        held = now - self._met_since.setdefault(rule.id, now)
        status = RuleStatus.TRIGGERED if held >= rule.hold_s else RuleStatus.PENDING
        return RuleResult(rule, status, results, round(held, 2))

    # ---------- 存檔 ----------

    def save(self) -> bool:
        if self.path is None:
            return True
        data = {"version": 1, "rules": [r.to_dict() for r in self.rules]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")  # 先寫暫存檔再取代，避免寫到一半損壞
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:
            self._report(RuleErrorCode.SAVE_FAILED, f"{self.path}：{e}")
            return False
        return True

    def load(self) -> bool:
        """讀取規則檔；檔案不存在時為空清單。格式錯誤的規則會略過並記錄錯誤。"""
        if self.path is None or not self.path.exists():
            return True
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            items = data["rules"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            self._report(RuleErrorCode.LOAD_FAILED, f"{self.path}：{e}")
            return False
        self.rules = []
        for item in items:
            try:
                rule = Rule.from_dict(item)
                errors = rule.validate()
            except (KeyError, ValueError, TypeError) as e:
                rule, errors = None, [str(e)]
            if errors:
                self._report(RuleErrorCode.INVALID_RULE, f"{item.get('name', '?')}：{'；'.join(errors)}")
                continue
            self.rules.append(rule)
        return True


# ---------- 建議範本 ----------


def _cond(metric: Metric, points: list[str], reference: Reference, low=None, high=None,
          axis: Axis = Axis.Y, mode: RangeMode = RangeMode.INSIDE) -> Condition:
    return Condition(metric, points, axis, reference, low, high, mode)


def preset_rules() -> list[Rule]:
    """建議的規則範本（每次呼叫都回傳新的物件）。門檻為參考值，請依實際鏡頭位置調整。"""
    pos, dist, line = Metric.POSITION, Metric.DISTANCE, Metric.LINE_ANGLE
    diff, absolute, ratio = Reference.DIFF, Reference.ABSOLUTE, Reference.RATIO
    return [
        Rule("低頭", [_cond(pos, ["nose"], diff, high=-30)], hold_s=3,
             message="頭部低於坐正時，請抬頭"),
        Rule("駝背", [_cond(pos, ["mid_shoulder"], diff, high=-15),
                      _cond(pos, ["nose"], diff, high=-25)], hold_s=5,
             message="肩膀與頭部都下沉，請挺直背部"),
        Rule("身體前傾（靠近螢幕）", [_cond(dist, ["left_shoulder", "right_shoulder"], ratio, low=115)],
             hold_s=3, message="離螢幕太近，請往後坐"),
        Rule("肩膀傾斜", [_cond(line, ["right_shoulder", "left_shoulder"], absolute, -8, 8,
                             mode=RangeMode.OUTSIDE)], hold_s=5, message="肩膀一高一低，請坐正"),
        Rule("頭部側傾", [_cond(line, ["right_ear", "left_ear"], absolute, -10, 10,
                             mode=RangeMode.OUTSIDE)], hold_s=3, message="頭部歪向一側"),
        Rule("身體左右偏移", [_cond(pos, ["mid_shoulder"], diff, -40, 40, axis=Axis.X,
                               mode=RangeMode.OUTSIDE)], hold_s=5, message="身體偏離坐正時的位置"),
        Rule("手撐下巴", [_cond(dist, ["nose", "left_wrist"], absolute, high=90),
                         _cond(dist, ["nose", "right_wrist"], absolute, high=90)],
             match="any", hold_s=5, message="手撐下巴容易造成頸部歪斜"),
        Rule("離座", [Condition(Metric.NO_PERSON, [], Axis.Y, absolute, None, None)], hold_s=10,
             message="偵測不到人，是否已離座？"),
    ]
