"""第 1 頁（camera_setting）、第 2 頁（Skelenton_detection）、資料儲存（read_point_data）
目前關鍵點（now_point_get）辨識規則（recognition_rules）與操作介面（user_GUI）整合測試。

自動測試：pytest       （專案根目錄，設定在 pyproject.toml）
          沒有相機的電腦會自動略過需要相機的測試。
手動測試（需要相機，會開視窗）：
    python run/test.py camera [--preview]      第 1 頁：列出相機，--preview 開啟預覽
    python run/test.py skeleton                第 2 頁：即時骨架，用按鍵切換參數
    python run/test.py record [秒數] [--csv]   錄製幾秒關鍵點資料並讀回（預設 5 秒、SQLite）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from camera_setting import CameraErrorCode, CameraSetting, ErrorLog
from now_point_get import (
    LANDMARK_NAMES_ZH,
    NowPointGetter,
    displacement,
    to_center_pixels,
)
from read_point_data import (
    COLUMNS,
    LANDMARK_NAMES,
    CsvStore,
    SqliteStore,
    StorageErrorCode,
)
from recognition_rules import (
    Axis,
    Condition,
    Metric,
    RangeMode,
    RecognitionTracker,
    Reference,
    Rule,
    RuleEngine,
    RuleErrorCode,
    RuleStatus,
    evaluate_condition,
    preset_rules,
)
from Skelenton_detection import (
    MODEL_VARIANTS,
    MODELS_DIR,
    DetectionErrorCode,
    DetectionFrame,
    ParamKind,
    SkeletonDetection,
)


def codes(reporter) -> list:
    return [rec.code for rec in reporter.now_error_log.records()]


def new_camera() -> CameraSetting:
    return CameraSetting(error_log=ErrorLog(log_file=None))  # 測試不寫入 logs/


@pytest.fixture(scope="module")
def camera():
    """第 1 頁：掃描並開啟第一台相機，整個模組共用。"""
    cam = new_camera()
    cameras = cam.scan_cameras()
    if not cameras:
        pytest.skip("這台電腦沒有可用的相機")
    assert cam.select_camera(next(iter(cameras)))
    assert cam.open(), cam.now_error_log.records()
    cam.now_error_log.clear()
    yield cam
    cam.close()


@pytest.fixture(scope="module")
def page(camera):
    """第 2 頁：共用同一個物件以省去解析度探測時間，每個測試結束後恢復預設。"""
    p = SkeletonDetection(camera)
    assert codes(p) == []
    yield p
    p.close()


@pytest.fixture(autouse=True)
def _reset(request):
    yield
    if "page" in request.fixturenames:
        p = request.getfixturevalue("page")
        p.reset()
        p.now_error_log.clear()
        p.camera.now_error_log.clear()


# ---------- 第 1 頁 ----------


def test_camera_page_invalid_index_does_not_raise():
    cam = new_camera()
    assert cam.select_camera(99) is False
    assert codes(cam) == [CameraErrorCode.INVALID_INDEX]
    assert len(cam.error_log) == 1


def test_camera_page_open_without_select():
    cam = new_camera()
    assert cam.open() is False
    assert cam.read_frame() == (False, None)
    assert codes(cam) == [CameraErrorCode.NOT_SELECTED, CameraErrorCode.NOT_OPENED]


def test_camera_page_lists_names(camera):
    assert all(isinstance(name, str) and name for name in camera.available_cameras.values())
    assert camera.camera_name == camera.available_cameras[camera.camera_index]


# ---------- 第 2 頁需要第 1 頁 ----------


def test_page2_without_page1_reports_once():
    cam = new_camera()  # 沒有開啟相機
    with SkeletonDetection(cam) as p:
        assert p.ready is False
        for _ in range(5):
            assert p.process_frame().ok is False
        assert codes(p) == [DetectionErrorCode.CAMERA_NOT_READY]
        # 相機參數無法設定，骨架參數仍可調整
        assert p.set_param("fps", 15) is False
        assert p.set_param("model", "lite") is True


def test_page2_shares_total_log_but_not_now_log(camera, page):
    before = len(camera.error_log)
    page.set_param("num_poses", 99)
    assert codes(page) == [DetectionErrorCode.INVALID_VALUE]
    assert codes(camera) == []  # 不干擾第 1 頁
    assert len(camera.error_log) == before + 1  # 總紀錄共用


def test_camera_errors_forwarded_to_page2_until_closed():
    cam = new_camera()
    p = SkeletonDetection(cam)
    p.now_error_log.clear()
    cam.read_frame()  # 相機未開啟 -> NOT_OPENED
    assert CameraErrorCode.NOT_OPENED in codes(p)
    p.close()
    p.now_error_log.clear()
    cam.read_frame()
    assert codes(p) == []


# ---------- 參數描述（介面元件） ----------


def test_param_specs_are_renderable(page):
    assert page.params.groups() == ["骨架模型", "相機", "顯示"]
    for spec in page.params.specs():
        assert spec.label and spec.group
        if spec.is_dropdown:
            assert spec.options, spec.key
            assert spec.value in [v for v, _ in spec.options], spec.key
        else:
            assert spec.min < spec.max and spec.step > 0, spec.key
            assert spec.min <= spec.value <= spec.max, spec.key
        assert spec.coerce(spec.default) == spec.default


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("num_poses", "3", 3),  # 數字輸入框傳來的字串
        ("num_poses", 2.0, 2),
        ("min_tracking_confidence", "0.75", 0.75),
        ("model", "Heavy（最準）", "heavy"),  # 下拉選單傳顯示文字
        ("mirror", False, False),
    ],
)
def test_set_param_accepts_ui_values(page, key, raw, expected):
    assert page.set_param(key, raw) is True
    assert page.params[key] == expected
    assert codes(page) == []


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("num_poses", 0),
        ("num_poses", 1.5),
        ("num_poses", "abc"),
        ("num_poses", True),
        ("min_pose_detection_confidence", 1.2),
        ("model", "ultra"),
        ("rotation", 45),
        ("brightness", 500),
    ],
)
def test_set_param_rejects_invalid_values(page, key, raw):
    before = page.params[key]
    assert page.set_param(key, raw) is False
    assert page.params[key] == before
    assert codes(page) == [DetectionErrorCode.INVALID_VALUE]


def test_unknown_param(page):
    assert page.set_param("no_such_param", 1) is False
    assert codes(page) == [DetectionErrorCode.UNKNOWN_PARAM]


# ---------- 即時偵測 ----------


def test_process_frame(page):
    f = page.process_frame()
    assert f.ok and f.result is not None
    assert f.frame.ndim == 3 and f.inference_ms > 0


@pytest.mark.parametrize("variant", [v for v, _ in MODEL_VARIANTS])
def test_switch_model_live(page, variant):
    if not (MODELS_DIR / f"pose_landmarker_{variant}.task").exists():
        pytest.skip(f"未下載 {variant} 模型")
    assert page.set_param("model", variant) is True
    for _ in range(3):
        assert page.process_frame().ok


def test_change_detector_params_live(page):
    for key, value in [("num_poses", 2), ("min_pose_detection_confidence", 0.8),
                       ("min_pose_presence_confidence", 0.3), ("min_tracking_confidence", 0.9)]:
        assert page.set_param(key, value) is True
        assert page.process_frame().ok


def test_missing_model_keeps_previous(camera, tmp_path):
    with SkeletonDetection(camera, models_dir=tmp_path) as p:
        assert codes(p) == [DetectionErrorCode.MODEL_NOT_FOUND]
        assert p.set_param("model", "lite") is False
        assert p.params["model"] == "full"
        f = p.process_frame()  # 沒有模型仍可顯示影像，只是沒有骨架
        assert f.ok and f.result is None and f.frame is not None


def test_change_resolution(page):
    spec = page.params.spec("resolution")
    for (w, h), _ in spec.options:
        assert page.set_param("resolution", (w, h)) is True, page.now_error_log.records()
        f = page.process_frame()
        assert f.ok and f.frame.shape[:2] == (h, w)


def test_rotation_and_mirror(page):
    w, h = page.params["resolution"]
    page.set_param("rotation", 90)
    assert page.process_frame().frame.shape[:2] == (w, h)
    page.set_param("rotation", 0)
    page.set_param("mirror", False)
    page.set_param("show_skeleton", False)
    assert page.process_frame().frame.shape[:2] == (h, w)


def test_brightness_changes_image(page):
    page.set_param("show_skeleton", False)
    page.set_param("mirror", False)
    dark = page.process_frame().frame.mean()
    page.set_param("brightness", 100)
    bright = page.process_frame().frame.mean()
    assert bright > dark


def test_fps_setting(page):
    fps = page.params["fps"]
    assert page.set_param("fps", fps) is True
    for option, _ in page.params.spec("fps").options:
        ok = page.set_param("fps", option)
        # 不支援的 FPS 必須被拒絕並保留原值，不能讓程式中斷
        if not ok:
            assert DetectionErrorCode.CAMERA_PROP_UNSUPPORTED in codes(page)
            assert page.params["fps"] != option
        else:
            assert round(page.camera.get_property(cv2.CAP_PROP_FPS)) == option


def test_page2_close_keeps_camera_open(camera):
    with SkeletonDetection(camera):
        pass
    assert camera.is_opened
    ok, _ = camera.read_frame()
    assert ok


def test_param_kinds_cover_ui_widgets(page):
    kinds = {spec.kind for spec in page.params.specs()}
    assert kinds == set(ParamKind)


# ---------- 資料儲存 ----------


def fake_frame(num_poses: int = 1, base: float = 0.1, size=(640, 480)) -> DetectionFrame:
    """不需相機的假偵測結果：第 i 個關鍵點 x = base + i/100、y = 0.25，影像大小為 size。"""
    def lm(i):
        return SimpleNamespace(x=base + i / 100, y=0.25, z=-0.1, visibility=0.9, presence=0.8)

    def wlm(i):
        return SimpleNamespace(x=i / 1000, y=-0.2, z=0.05)

    result = SimpleNamespace(
        pose_landmarks=[[lm(i) for i in range(33)] for _ in range(num_poses)],
        pose_world_landmarks=[[wlm(i) for i in range(33)] for _ in range(num_poses)],
    )
    image = np.zeros((size[1], size[0], 3), np.uint8)
    return DetectionFrame(True, image, result, 12.5)


def new_store(cls, tmp_path, **kwargs):
    return cls(data_dir=tmp_path, error_log=ErrorLog(log_file=None), **kwargs)


STORES = [SqliteStore, CsvStore]


@pytest.mark.parametrize("cls", STORES)
def test_store_roundtrip(cls, tmp_path):
    with new_store(cls, tmp_path) as store:
        sid = store.start_session("TestCam", {"model": "full", "resolution": (640, 480)})
        assert sid
        assert store.record(fake_frame(1, 0.1))
        assert store.record(fake_frame(0))  # 沒有人
        assert store.record(fake_frame(2, 0.3))  # 兩個人 -> 兩列
        assert store.record(DetectionFrame(False)) is False  # 讀取失敗的幀不記錄
        store.end_session()

        df = store.load_session(sid)
        assert list(df.columns) == list(COLUMNS)
        assert len(df) == 4 and df["frame_index"].tolist() == [0, 1, 2, 2]
        assert df["pose_index"].isna().tolist() == [False, True, False, False]
        first = df.iloc[0]
        assert first["nose_x"] == pytest.approx(0.1)
        assert first["right_foot_index_x"] == pytest.approx(0.1 + 32 / 100)
        assert first["left_shoulder_wx"] == pytest.approx(11 / 1000)
        assert first["left_shoulder_vis"] == pytest.approx(0.9)
        assert df.iloc[3]["nose_x"] == pytest.approx(0.3)

        sessions = store.list_sessions()
        row = sessions.set_index("session_id").loc[sid]
        assert row["camera"] == "TestCam" and row["frame_count"] == 3 and row["ended_at"]
        assert row["params"]["resolution"] == [640, 480]
        assert codes(store) == []


@pytest.mark.parametrize("cls", STORES)
def test_store_batches_and_reads_while_recording(cls, tmp_path):
    with new_store(cls, tmp_path, batch_size=10) as store:
        sid = store.start_session()
        for _ in range(25):
            store.record(fake_frame())
        assert len(store._buffer) == 5  # 已寫入兩批，剩 5 列暫存
        assert len(store.load_session(sid)) == 25  # 讀取時會先寫入暫存


@pytest.mark.parametrize("cls", STORES)
def test_store_min_interval(cls, tmp_path):
    with new_store(cls, tmp_path, min_interval_s=10) as store:
        store.start_session()
        assert store.record(fake_frame()) is True
        assert store.record(fake_frame()) is False
        assert store.frame_count == 1


@pytest.mark.parametrize("cls", STORES)
def test_store_errors_do_not_raise(cls, tmp_path):
    with new_store(cls, tmp_path) as store:
        for _ in range(3):
            assert store.record(fake_frame()) is False
        assert codes(store) == [StorageErrorCode.NO_SESSION]  # 只記錄一次
        store.now_error_log.clear()
        assert store.load_session("no_such_session").empty
        assert codes(store) == [StorageErrorCode.SESSION_NOT_FOUND]


@pytest.mark.parametrize("cls", STORES)
def test_store_open_failed(cls, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")  # 資料夾位置已被檔案占用 -> 無法建立
    with new_store(cls, blocker / "records") as store:
        assert store.start_session() is None
        assert store.recording is False
        assert codes(store) == [StorageErrorCode.OPEN_FAILED]


def test_store_write_failure_keeps_rows_for_retry(tmp_path, monkeypatch):
    with new_store(SqliteStore, tmp_path, batch_size=1, max_buffer=3) as store:
        sid = store.start_session()

        def broken(rows):
            raise OSError("磁碟已滿")

        monkeypatch.setattr(store, "_write_rows", broken)
        for _ in range(5):
            store.record(fake_frame())
        assert codes(store).count(StorageErrorCode.WRITE_FAILED) == 1
        assert StorageErrorCode.BUFFER_OVERFLOW in codes(store)
        assert len(store._buffer) == 3
        monkeypatch.undo()  # 恢復後重試成功
        assert store.flush() is True
        assert len(store.load_session(sid)) == 3


@pytest.mark.parametrize("cls", STORES)
def test_store_records_live_camera(cls, page, tmp_path):
    with new_store(cls, tmp_path) as store:
        sid = store.start_session(page.camera.camera_name, page.params.values())
        for _ in range(10):
            assert store.record(page.process_frame())
        store.end_session()
        df = store.load_session(sid)
        assert df["frame_index"].nunique() == 10
        assert (df["inference_ms"] > 0).all()



# ---------- 目前關鍵點 ----------


def test_now_point_initially_empty():
    getter = NowPointGetter()
    assert not getter.latest.detected and getter.get("nose") is None


def test_now_point_values():
    getter = NowPointGetter()
    snap = getter.update(fake_frame(1, 0.1))
    assert snap.detected and snap.num_poses == 1 and snap.pose_index == 0
    assert len(snap.points) == 33 and getter.latest is snap
    shoulder = snap.get("left_shoulder")
    assert shoulder == snap.get(11)
    assert shoulder.name_zh == "左肩" and shoulder.x == pytest.approx(0.1 + 11 / 100)
    assert shoulder.wx == pytest.approx(11 / 1000) and shoulder.visibility == pytest.approx(0.9)
    assert snap.to_dict()["nose"]["x"] == pytest.approx(0.1)
    assert len(LANDMARK_NAMES_ZH) == 33


def test_now_point_center_pixels():
    snap = NowPointGetter().update(fake_frame(1, 0.1))  # 640x480
    assert snap.image_size == (640, 480)
    shoulder = snap.get("left_shoulder")  # x = 0.21、y = 0.25
    assert (shoulder.cx, shoulder.cy) == (-185.6, 120.0)  # 在中心左方、上方
    assert snap.to_dict()["left_shoulder"]["cx"] == -185.6
    assert to_center_pixels(0.5, 0.5, 640, 480) == (0.0, 0.0)
    assert to_center_pixels(1.0, 1.0, 640, 480) == (320.0, -240.0)  # 右下角
    assert to_center_pixels(0.123456, 0.5, 640, 480) == (-240.99, 0.0)  # 小數點後 2 位

    hd = NowPointGetter().update(fake_frame(1, 0.1, size=(1280, 720))).get("left_shoulder")
    assert (hd.cx, hd.cy) == (-371.2, 180.0)  # 依實際影像大小換算

    no_image = DetectionFrame(True, None, fake_frame().result)
    assert NowPointGetter().update(no_image).get("nose").cx is None


def test_now_point_no_person_and_failed_frame():
    getter = NowPointGetter()
    getter.update(fake_frame(1))
    assert getter.update(DetectionFrame(False)).detected  # 讀取失敗：保留上一次
    snap = getter.update(fake_frame(0))  # 沒有人：清空
    assert not snap.detected and snap.num_poses == 0 and snap.pose_index is None
    getter.update(fake_frame(1))
    getter.reset()
    assert not getter.latest.detected


def test_now_point_pose_index_and_filter():
    second = NowPointGetter(pose_index=1)
    assert not second.update(fake_frame(1)).detected  # 只有一人
    assert second.update(fake_frame(2)).pose_index == 1
    upper = NowPointGetter(indices=range(25))
    snap = upper.update(fake_frame(1))
    assert len(snap.points) == 25 and snap.get("left_knee") is None


def test_now_point_baseline():
    getter = NowPointGetter()
    assert getter.baseline is None
    assert getter.set_baseline() is None  # 沒有人：不設定
    getter.update(fake_frame(1, 0.1))
    base = getter.set_baseline()
    assert base is getter.baseline and base.get("left_shoulder").cx == -185.6
    getter.update(fake_frame(1, 0.3))  # 之後的幀不影響初始點
    assert getter.baseline.get("left_shoulder").cx == -185.6
    assert getter.latest.get("left_shoulder").cx != -185.6
    assert getter.low_visibility_points() == []
    getter.clear_baseline()
    assert getter.baseline is None
    getter.set_baseline()
    getter.reset()  # 離開第 2 頁：一併清除
    assert getter.baseline is None


def test_now_point_displacement():
    getter = NowPointGetter()
    getter.update(fake_frame(1, 0.1))
    assert getter.displacement("nose") is None  # 尚未設定初始點
    base = getter.set_baseline()
    assert getter.displacement("left_shoulder") == (0.0, 0.0)
    getter.update(fake_frame(1, 0.3))  # x 往右 0.2 -> 640 * 0.2 = 128 px
    assert getter.displacement("left_shoulder") == (128.0, 0.0)
    assert getter.displacement(11) == (128.0, 0.0)
    dx, dy, dz = getter.displacement("left_shoulder", ("x", "y", "z"))
    assert dx == pytest.approx(0.2) and dy == dz == 0
    assert displacement(getter.latest, base, "left_shoulder", ("wx", "wy", "wz")) == (0, 0, 0)
    getter.update(fake_frame(0))  # 沒有人
    assert getter.displacement("left_shoulder") is None
    assert displacement(None, base, "nose") is None


def test_now_point_low_visibility_warning():
    frame = fake_frame(1)
    frame.result.pose_landmarks[0][7].visibility = 0.2  # 左耳被擋住
    getter = NowPointGetter()
    getter.update(frame)
    assert [p.index for p in getter.low_visibility_points()] == [7]


def test_now_point_live_camera(page):
    getter = NowPointGetter()
    for _ in range(5):
        snap = getter.update(page.process_frame())
    assert snap.timestamp > 0 and snap.num_poses in (0, 1)
    assert snap.image_size == page.params["resolution"]


# ---------- 辨識規則 ----------


def frame_with(points: dict, visibility: dict | None = None):
    """指定關鍵點的正規化座標 {名稱: (x, y)}，其餘沿用 fake_frame；640x480。"""
    frame = fake_frame(1)
    pose = frame.result.pose_landmarks[0]
    for name, (x, y) in points.items():
        pose[LANDMARK_NAMES.index(name)].x, pose[LANDMARK_NAMES.index(name)].y = x, y
    for name, vis in (visibility or {}).items():
        pose[LANDMARK_NAMES.index(name)].visibility = vis
    return frame


# 鼻子在畫面中心 (0, 0)，兩肩在 (±160, -120)
UPRIGHT = {"nose": (0.5, 0.5), "left_shoulder": (0.75, 0.75), "right_shoulder": (0.25, 0.75)}


def snapshots(current_points, baseline_points=None, visibility=None):
    getter = NowPointGetter()
    if baseline_points is not None:
        getter.update(frame_with(baseline_points))
        getter.set_baseline()
    getter.update(frame_with(current_points, visibility))
    return getter.latest, getter.baseline


def cond(metric, points, reference=Reference.ABSOLUTE, low=None, high=None, axis=Axis.Y,
         mode=RangeMode.INSIDE):
    return Condition(metric, points, axis, reference, low, high, mode)


def test_rule_metric_values():
    cur, _ = snapshots(UPRIGHT)
    pos, dist, line, joint = Metric.POSITION, Metric.DISTANCE, Metric.LINE_ANGLE, Metric.JOINT_ANGLE
    cases = [
        (cond(pos, ["nose"], axis=Axis.X), 0.0),
        (cond(pos, ["left_shoulder"], axis=Axis.Y), -120.0),
        (cond(pos, ["left_shoulder"], axis=Axis.DIST), 200.0),  # 與畫面中心的距離
        (cond(pos, ["mid_shoulder"], axis=Axis.Y), -120.0),  # 虛擬點取平均
        (cond(pos, ["mid_shoulder"], axis=Axis.X), 0.0),
        (cond(dist, ["left_shoulder", "right_shoulder"]), 320.0),
        (cond(line, ["right_shoulder", "left_shoulder"]), 0.0),  # 兩肩水平
        (cond(joint, ["right_shoulder", "nose", "left_shoulder"]), 106.26),
    ]
    for c, expected in cases:
        assert evaluate_condition(c, cur, None).value == pytest.approx(expected), c.describe()

    tilted, _ = snapshots({**UPRIGHT, "left_shoulder": (0.75, 0.70)})  # 左肩高 24 px
    tilt = evaluate_condition(cond(line, ["right_shoulder", "left_shoulder"]), tilted, None)
    assert tilt.value == pytest.approx(4.29)


def test_rule_baseline_references():
    lowered = {**UPRIGHT, "nose": (0.5, 0.6)}  # 鼻子往下 48 px
    cur, base = snapshots(lowered, UPRIGHT)
    dy = cond(Metric.POSITION, ["nose"], Reference.DIFF, high=-30)
    result = evaluate_condition(dy, cur, base)
    assert result.value == -48.0 and result.met
    moved = cond(Metric.POSITION, ["nose"], Reference.DIFF, axis=Axis.DIST)
    assert evaluate_condition(moved, cur, base).value == 48.0

    wider, base = snapshots({**UPRIGHT, "left_shoulder": (0.8, 0.75), "right_shoulder": (0.2, 0.75)},
                            UPRIGHT)  # 兩肩距離 320 -> 384
    shoulders = ["left_shoulder", "right_shoulder"]
    assert evaluate_condition(cond(Metric.DISTANCE, shoulders, Reference.RATIO), wider, base).value == 120.0
    assert evaluate_condition(cond(Metric.DISTANCE, shoulders, Reference.DIFF), wider, base).value == 64.0

    tilted, base = snapshots({**UPRIGHT, "left_shoulder": (0.75, 0.65)}, UPRIGHT)
    angle = cond(Metric.LINE_ANGLE, ["right_shoulder", "left_shoulder"], Reference.DIFF)
    assert evaluate_condition(angle, tilted, base).value == pytest.approx(8.53)

    no_base = evaluate_condition(dy, cur, None)  # 沒有初始點
    assert no_base.value is None and no_base.met is None and "初始點" in no_base.reason


def test_rule_unknown_when_not_visible():
    cur, _ = snapshots(UPRIGHT, visibility={"left_shoulder": 0.2})
    result = evaluate_condition(cond(Metric.POSITION, ["mid_shoulder"]), cur, None)
    assert result.met is None and "兩肩中點" in result.reason
    empty = NowPointGetter().update(fake_frame(0))
    assert evaluate_condition(cond(Metric.POSITION, ["nose"]), empty, None).reason == "未偵測到人"


def test_condition_range_and_text():
    c = cond(Metric.POSITION, ["nose"], Reference.DIFF, high=-30)
    assert c.in_range(-31) and not c.in_range(-29)
    assert c.describe() == "鼻子 垂直位移 ≤ -30 px"
    c.mode = RangeMode.OUTSIDE
    assert c.in_range(-29) and not c.in_range(-31)
    band = cond(Metric.LINE_ANGLE, ["right_ear", "left_ear"], low=-10, high=10, mode=RangeMode.OUTSIDE)
    assert band.in_range(11) and band.in_range(-11) and not band.in_range(5)
    assert band.range_text() == "< -10 或 > 10 °"
    ratio = cond(Metric.DISTANCE, ["nose", "left_wrist"], Reference.RATIO, low=115)
    assert ratio.range_text() == "≥ 115 %"
    assert cond(Metric.POSITION, ["nose"]).in_range(12345)  # 上下限都不限
    assert cond(Metric.POSITION, ["nose"], low=10, high=5).warnings()

    c.set_metric(Metric.JOINT_ANGLE)  # 切換指標會重設點與比較基準
    assert len(c.points) == 3 and c.reference is Reference.ABSOLUTE and not c.validate()
    assert Condition(Metric.DISTANCE, ["nose"]).validate()  # 點數不對
    assert Condition(Metric.LINE_ANGLE, ["nose", "left_ear"], reference=Reference.RATIO).validate()


def test_rule_engine_status_and_hold():
    engine = RuleEngine(path=None)
    rule = engine.add(Rule("低頭", [cond(Metric.POSITION, ["nose"], Reference.DIFF, high=-30)], hold_s=3))
    upright, base = snapshots(UPRIGHT, UPRIGHT)
    lowered, _ = snapshots({**UPRIGHT, "nose": (0.5, 0.6)})

    assert engine.evaluate_rule(rule, upright, None, now=0).status is RuleStatus.NEED_BASELINE
    assert engine.evaluate_rule(rule, upright, base, now=0).status is RuleStatus.NORMAL
    first = engine.evaluate_rule(rule, lowered, base, now=10)
    assert first.status is RuleStatus.PENDING and first.held_s == 0
    assert engine.evaluate_rule(rule, lowered, base, now=12).status is RuleStatus.PENDING
    done = engine.evaluate_rule(rule, lowered, base, now=13.1)
    assert done.triggered and done.held_s == pytest.approx(3.1)
    engine.evaluate_rule(rule, upright, base, now=14)  # 恢復正常：計時歸零
    assert engine.evaluate_rule(rule, lowered, base, now=15).status is RuleStatus.PENDING

    rule.enabled = False
    assert engine.evaluate_rule(rule, lowered, base).status is RuleStatus.DISABLED
    assert engine.evaluate_rule(Rule("空"), lowered, base).status is RuleStatus.NO_CONDITION


def test_rule_engine_match_any_all():
    cur, _ = snapshots(UPRIGHT, visibility={"left_wrist": 0.1})
    near = cond(Metric.DISTANCE, ["nose", "right_shoulder"], high=500)  # 成立
    hidden = cond(Metric.DISTANCE, ["nose", "left_wrist"], high=500)  # 看不到：無法判斷
    far = cond(Metric.DISTANCE, ["nose", "right_shoulder"], high=10)  # 不成立
    engine = RuleEngine(path=None)

    def status(conditions, match):
        return engine.evaluate_rule(Rule("r", conditions, match, 0), cur, None).status

    assert status([near, hidden], "all") is RuleStatus.UNKNOWN
    assert status([near, hidden], "any") is RuleStatus.TRIGGERED
    assert status([far, hidden], "all") is RuleStatus.NORMAL
    assert status([far, hidden], "any") is RuleStatus.UNKNOWN
    assert status([near, far], "all") is RuleStatus.NORMAL


def test_rule_no_person():
    engine = RuleEngine(path=None)
    away = engine.add(Rule("離座", [Condition(Metric.NO_PERSON, [], reference=Reference.ABSOLUTE)],
                           hold_s=10))
    empty = NowPointGetter().update(fake_frame(0))
    assert engine.evaluate_rule(away, empty, None, now=0).status is RuleStatus.PENDING
    assert engine.evaluate_rule(away, empty, None, now=10).triggered
    present, _ = snapshots(UPRIGHT)
    assert engine.evaluate_rule(away, present, None, now=11).status is RuleStatus.NORMAL


def test_rule_engine_save_load(tmp_path):
    path = tmp_path / "rules.json"
    engine = RuleEngine(path, error_log=ErrorLog(log_file=None))
    assert engine.rules == [] and codes(engine) == []  # 檔案不存在：空清單
    for rule in preset_rules():
        engine.add(rule)
    engine.rules[0].enabled = False
    assert engine.save()
    loaded = RuleEngine(path, error_log=ErrorLog(log_file=None))
    assert [r.to_dict() for r in loaded.rules] == [r.to_dict() for r in engine.rules]
    assert not loaded.rules[0].enabled and codes(loaded) == []


def test_rule_engine_load_errors(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text("{not json", encoding="utf-8")
    broken = RuleEngine(path, error_log=ErrorLog(log_file=None))
    assert broken.rules == [] and codes(broken) == [RuleErrorCode.LOAD_FAILED]

    good = preset_rules()[0].to_dict()
    bad = {"name": "壞規則", "conditions": [{"metric": "DISTANCE", "points": ["nose"]}]}
    path.write_text(json.dumps({"version": 1, "rules": [good, bad]}), encoding="utf-8")
    partial = RuleEngine(path, error_log=ErrorLog(log_file=None))
    assert [r.name for r in partial.rules] == ["低頭"]  # 壞的略過，好的保留
    assert codes(partial) == [RuleErrorCode.INVALID_RULE]


def test_rule_presets():
    presets = preset_rules()
    assert len({r.name for r in presets}) == len(presets) >= 8
    cur, base = snapshots(UPRIGHT, UPRIGHT)
    engine = RuleEngine(path=None)
    for rule in presets:
        assert rule.validate() == [] and rule.message, rule.name
        assert all(c.describe() for c in rule.conditions)
        engine.add(rule)
    statuses = {r.rule.name: r.status for r in engine.evaluate(cur, base)}
    assert statuses["低頭"] is RuleStatus.NORMAL and statuses["肩膀傾斜"] is RuleStatus.NORMAL
    assert preset_rules()[0].id != presets[0].id  # 每次都是新物件

def test_recognition_tracker():
    engine = RuleEngine(path=None)
    rule = engine.add(Rule("低頭", [cond(Metric.POSITION, ["nose"], Reference.DIFF, high=-30)], hold_s=0))
    upright, base = snapshots(UPRIGHT, UPRIGHT)
    lowered, _ = snapshots({**UPRIGHT, "nose": (0.5, 0.6)})
    tracker = RecognitionTracker()
    tracker.start(now=100)
    for now, snap in [(101, upright), (102, lowered), (103, lowered), (104, lowered),
                      (105, upright), (106, lowered), (107, upright)]:
        tracker.update(engine.evaluate(snap, base, now=now), now=now)
    stats = tracker.get(rule.id)
    assert stats.trigger_count == 2  # 102 開始一次、106 再一次
    assert stats.triggered_s == pytest.approx(4.0)  # 102~105 共 3 秒 + 106~107 共 1 秒
    assert stats.last_triggered_at == 106 and tracker.elapsed_s == 7
    assert tracker.get("unknown").trigger_count == 0
    tracker.start(now=200)  # 重新開始：歸零
    assert tracker.get(rule.id).trigger_count == 0

# ---------- 操作介面 ----------


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 測試不實際開視窗
    import gui_style
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    gui_style.apply_theme(app)  # 一併測試樣式表
    return app


@pytest.fixture
def window(qapp, camera, tmp_path):
    from user_GUI import MainWindow

    w = MainWindow(camera=camera, store_factory=lambda: new_store(SqliteStore, tmp_path),
                   rules_path=tmp_path / "rules.json")  # 測試不寫入 data/rules.json
    yield w
    w.detection.leave()  # 不呼叫 close()：會關閉模組共用的相機
    w.store.close()
    w.deleteLater()


def enter_detection(window):
    window.cover.scan()
    window.cover.confirm()
    window.detection.timer.stop()  # 改由測試手動呼叫 tick()
    return window.detection


def test_gui_cover_lists_cameras(window):
    window.cover.scan()
    assert window.stack.currentWidget() is window.cover
    assert window.cover.combo.count() == len(window.camera.available_cameras) >= 1
    assert window.cover.btn_ok.isEnabled()


def test_gui_param_widgets_follow_specs(window):
    from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

    det = enter_detection(window)
    assert window.stack.currentWidget() is det
    expected = {ParamKind.CHOICE: QComboBox, ParamKind.BOOL: QComboBox,
                ParamKind.INT: QSpinBox, ParamKind.FLOAT: QDoubleSpinBox}
    for spec in det.page.params.specs():
        assert type(det.params.widgets[spec.key]) is expected[spec.kind], spec.key

    det.params.widgets["model"].setCurrentIndex(0)
    assert det.page.params["model"] == "lite"
    det.params.widgets["num_poses"].setValue(2)
    assert det.page.params["num_poses"] == 2
    det.params.widgets["min_tracking_confidence"].setValue(0.7)
    assert det.page.params["min_tracking_confidence"] == pytest.approx(0.7)

    det.params._changed("num_poses", 99)  # 被拒絕：元件還原、跳出錯誤視窗
    assert det.params.widgets["num_poses"].value() == 2
    assert window.notifier.visible


def test_gui_video_and_status(window):
    det = enter_detection(window)
    for _ in range(3):
        det.tick()
    assert det.view.pixmap() is not None and not det.view.pixmap().isNull()
    assert det.chip_fps.text().endswith("FPS") and "ms" in det.chip_infer.text()
    assert det.chip_model.text() == "Full（平衡）"
    assert window.camera.camera_name in det.camera_chip.text()


def test_gui_recognition_flow(window):
    det = enter_detection(window)
    panel = det.recognition
    assert panel.btn_start.isEnabled() and not panel.btn_end.isEnabled()

    panel.btn_start.click()
    assert window.store.recording and det.dialog.isVisible()
    assert not det.params.isEnabled() and not det.btn_back.isEnabled()
    assert not panel.btn_start.isEnabled() and not panel.btn_reset.isEnabled()
    session_id = window.store.session_id
    for _ in range(5):
        det.tick()
    assert "記錄中" in det.rec_pill.text() and det.rec_pill.property("state") == "recording"

    panel.btn_end.click()
    assert not window.store.recording and not det.dialog.isVisible()
    assert det.params.isEnabled() and panel.btn_start.isEnabled()
    assert det.rec_pill.property("state") == "saved" and "5 幀" in det.rec_pill.text()
    assert window.store.load_session(session_id)["frame_index"].nunique() == 5
    row = window.store.list_sessions().set_index("session_id").loc[session_id]
    assert row["frame_count"] == 5 and row["camera"] == window.camera.camera_name


def test_gui_closing_dialog_ends_recording(window):
    det = enter_detection(window)
    det.recognition.btn_start.click()
    det.dialog.close()
    assert not window.store.recording
    assert det.recognition.btn_start.isEnabled()


def test_gui_reset(window):
    det = enter_detection(window)
    det.params.widgets["model"].setCurrentIndex(2)
    det.params.widgets["mirror"].setCurrentIndex(1)
    det.recognition.btn_reset.click()
    assert det.page.params["model"] == "full" and det.page.params["mirror"] is True
    assert det.params.widgets["model"].currentText() == "Full（平衡）"


def test_gui_back_keeps_camera_open(window):
    det = enter_detection(window)
    det.btn_back.click()
    assert window.stack.currentWidget() is window.cover
    assert det.page is None and det.params is None
    assert window.camera.is_opened
    enter_detection(window)  # 可以再次進入第 2 頁
    assert det.page is not None



def test_gui_keypoint_window(window):
    det = enter_detection(window)
    win = det.keypoint_window
    assert win.isVisible()  # 進入第 2 頁自動開啟
    assert win.table.rowCount() == 33 and win.table.item(11, 1).text() == "左肩"

    det.point_getter.update(fake_frame(1, 0.1))  # 用固定數值檢查顯示
    win.refresh()
    # 預設為中心座標（像素）：只有 x、y，小數點後 2 位
    assert win.coord.currentText() == "中心座標（像素）"
    assert win.table.horizontalHeaderItem(2).text() == "X（px）"
    assert (win.table.item(11, 2).text(), win.table.item(11, 3).text()) == ("-185.60", "+120.00")
    assert win.table.isColumnHidden(4)
    assert win.table.item(11, 5).text() == "0.90"
    assert "偵測到 1 人" in win.status.text() and "640×480" in win.status.text()

    win.coord.setCurrentIndex(1)  # 影像座標
    assert not win.table.isColumnHidden(4)
    assert win.table.item(11, 2).text() == f"{0.1 + 11 / 100:.3f}"

    win.coord.setCurrentIndex(2)  # 世界座標
    assert win.table.horizontalHeaderItem(2).text() == "X（公尺）"
    assert win.table.item(11, 2).text() == f"{11 / 1000:+.3f}"

    win.btn_pause.setChecked(True)  # 暫停後不更新
    det.point_getter.update(fake_frame(0))
    win.refresh()
    assert win.table.item(11, 2).text() != "—"
    win.btn_pause.setChecked(False)
    win.refresh()
    assert win.table.item(11, 2).text() == "—" and win.status.text() == "未偵測到人"

    det.tick()  # 實際影像也會更新 getter
    assert det.point_getter.latest.timestamp > 0

    win.close()
    det.btn_points.click()  # 關閉後可從標題列按鈕再開啟
    assert win.isVisible()
    det.btn_back.click()  # 離開第 2 頁時關閉並清空
    assert not win.isVisible() and not det.point_getter.latest.detected


def test_gui_set_baseline(window):
    det = enter_detection(window)
    win = det.keypoint_window
    received = []
    win.baseline_changed.connect(received.append)
    assert win.baseline_pill.property("state") == "idle"

    det.point_getter.reset()
    win.btn_baseline.click()  # 沒有人
    assert win.baseline_pill.property("state") == "warning" and "未偵測到人" in win.baseline_pill.text()
    assert received == []

    det.point_getter.update(fake_frame(1, 0.1))
    win.btn_baseline.click()
    assert win.baseline_pill.property("state") == "saved" and "已設定初始點" in win.baseline_pill.text()
    assert win.btn_baseline.text().endswith("重新設定初始點")
    assert received == [det.point_getter.baseline]

    frame = fake_frame(1)
    frame.result.pose_landmarks[0][11].visibility = 0.1  # 左肩不清楚：仍設定但提醒
    det.point_getter.update(frame)
    win.btn_baseline.click()
    assert win.baseline_pill.property("state") == "warning" and "左肩" in win.baseline_pill.text()
    assert len(received) == 2

    det.btn_back.click()  # 離開後初始點清除，再進入時顯示未設定
    enter_detection(window)
    assert det.point_getter.baseline is None
    assert win.baseline_pill.property("state") == "idle"


def test_gui_baseline_window(window):
    det = enter_detection(window)
    win = det.keypoint_window
    base_win = win.baseline_window
    assert not win.btn_view_baseline.isEnabled()

    det.point_getter.reset()
    win.btn_baseline.click()  # 沒有人：不彈出
    assert not base_win.isVisible()

    det.point_getter.update(fake_frame(1, 0.1))
    win.btn_baseline.click()  # 設定成功：自動彈出並顯示儲存的數值
    assert base_win.isVisible() and win.btn_view_baseline.isEnabled()
    assert (base_win.table.item(11, 2).text(), base_win.table.item(11, 3).text()) == ("-185.60", "+120.00")
    assert "640×480" in base_win.info.text() and base_win.warning.isHidden()

    assert base_win.table.horizontalHeaderItem(6).text() == "目前位移（px）"
    assert base_win.table.item(11, 6).text() == "+0.00, +0.00"

    det.point_getter.update(fake_frame(1, 0.3))  # 初始點數值不變，位移欄即時更新
    win.refresh()
    base_win.refresh_delta()
    assert win.table.item(11, 2).text() != "-185.60"
    assert base_win.table.item(11, 2).text() == "-185.60"
    assert base_win.table.item(11, 6).text() == "+128.00, +0.00"
    assert "即時更新" in base_win.live.text()

    base_win.coord.setCurrentIndex(2)  # 可切換座標，位移單位跟著改變
    assert base_win.table.item(11, 2).text() == f"{11 / 1000:+.3f}"
    assert base_win.table.horizontalHeaderItem(6).text() == "目前位移（公尺）"
    assert base_win.table.item(11, 6).text() == "+0.000, +0.000, +0.000"
    base_win.coord.setCurrentIndex(0)

    det.point_getter.update(fake_frame(0))  # 沒有人：無法計算位移
    base_win.refresh_delta()
    assert base_win.table.item(11, 6).text() == "—" and "未偵測到人" in base_win.live.text()
    det.point_getter.update(fake_frame(1, 0.1, size=(1280, 720)))  # 解析度改變：提醒
    base_win.refresh_delta()
    assert "畫面大小" in base_win.live.text()

    base_win.close()
    win.btn_view_baseline.click()  # 關閉後可再開啟
    assert base_win.isVisible()

    frame = fake_frame(1)
    frame.result.pose_landmarks[0][8].visibility = 0.1  # 右耳不清楚：顯示警告
    det.point_getter.update(frame)
    win.btn_baseline.click()
    assert not base_win.warning.isHidden() and "右耳" in base_win.warning.text()

    det.btn_back.click()  # 離開第 2 頁時一併關閉
    assert not base_win.isVisible()



def test_gui_rules_page(window):
    det = enter_detection(window)
    btn = det.recognition.btn_rules
    assert btn.text().endswith("添加辨識規則") and btn.isEnabled()

    btn.click()  # 開啟辨識規則頁
    assert window.stack.currentWidget() is window.rules
    det.tick()  # 規則頁期間相機持續運作
    assert det.page is not None and det.point_getter.latest.timestamp > 0

    window.rules.btn_back.click()  # 返回第 2 頁，不需重新載入
    assert window.stack.currentWidget() is det and det.page is not None

    det.recognition.btn_start.click()  # 記錄期間不能修改規則
    assert not btn.isEnabled()
    det.recognition.btn_end.click()
    assert btn.isEnabled()


def test_gui_rules_editor(window, tmp_path):
    page = window.rules
    assert page.rule_list.count() == 0 and page.editor_stack.currentIndex() == 0
    assert not page.btn_delete.isEnabled()

    page.btn_add.click()
    rule = page.current_rule
    assert page.rule_list.count() == 1 and page.editor_stack.currentIndex() == 1
    assert len(page.condition_editors) == 1 and (tmp_path / "rules.json").exists()

    page.name_edit.setText("低頭測試")
    page.name_edit.textEdited.emit("低頭測試")
    page.hold_spin.setValue(2.5)
    page.match_combo.setCurrentIndex(1)
    page.message_edit.setText("請抬頭")
    page.message_edit.textEdited.emit("請抬頭")
    assert (rule.name, rule.hold_s, rule.match, rule.message) == ("低頭測試", 2.5, "any", "請抬頭")
    assert "低頭測試" in page.rule_list.item(0).text()

    editor = page.condition_editors[0]
    editor.metric.setCurrentIndex(list(Metric).index(Metric.JOINT_ANGLE))
    assert rule.conditions[0].metric is Metric.JOINT_ANGLE
    assert [not c.isHidden() for c in editor.point_combos] == [True, True, True]
    assert editor.axis_row.isHidden()
    editor.metric.setCurrentIndex(list(Metric).index(Metric.NO_PERSON))
    assert editor.low_row.isHidden() and all(c.isHidden() for c in editor.point_combos)
    editor.metric.setCurrentIndex(list(Metric).index(Metric.POSITION))
    assert not editor.axis_row.isHidden()
    assert [not c.isHidden() for c in editor.point_combos] == [True, False, False]
    editor.point_combos[0].setCurrentIndex(0)  # 兩肩中點
    editor.high_none.setChecked(True)
    editor.low_none.setChecked(False)
    editor.low.setValue(10)
    c = rule.conditions[0]
    assert (c.points, c.high, c.low) == (["mid_shoulder"], None, 10.0)
    assert "兩肩中點" in editor.summary.text() and "≥ 10" in editor.summary.text()

    page.btn_add_condition.click()
    assert len(rule.conditions) == len(page.condition_editors) == 2
    page.condition_editors[0].btn_remove.click()
    assert len(rule.conditions) == 1 and rule.conditions[0].points == ["nose"]  # 刪掉的是第一個

    action = next(a for a in page.template_menu.actions() if a.text() == "肩膀傾斜")
    action.trigger()  # 從範本新增
    assert page.rule_list.count() == 2 and page.name_edit.text() == "肩膀傾斜"
    action.trigger()  # 同一範本再加一次：自動編號
    assert page.name_edit.text() == "肩膀傾斜 (2)"
    page.btn_delete.click()

    saved = RuleEngine(tmp_path / "rules.json", error_log=ErrorLog(log_file=None))
    assert [r.name for r in saved.rules] == ["低頭測試", "肩膀傾斜"]  # 自動儲存

    page.btn_delete.click()
    assert page.rule_list.count() == 1 and page.name_edit.text() == "低頭測試"


def test_gui_rules_preview(window):
    page, getter = window.rules, window.detection.point_getter
    next(a for a in page.template_menu.actions() if a.text() == "低頭").trigger()
    getter.update(frame_with(UPRIGHT))
    page.refresh_preview()
    assert "需要初始點" in page.preview_pill.text() and "尚未設定" in page.baseline_label.text()

    getter.set_baseline()
    page.refresh_preview()
    assert page.preview_pill.property("state") == "saved"  # 正常
    assert page.condition_editors[0].live.text().startswith("目前數值：+0.00 px")

    getter.update(frame_with({**UPRIGHT, "nose": (0.5, 0.6)}))  # 低頭 48 px
    page.refresh_preview()
    assert "條件成立中" in page.preview_pill.text()
    assert "-48.00 px" in page.condition_editors[0].live.text()
    assert page.condition_editors[0].live.property("state") == "met"
    assert "低頭" in page.all_rules_label.text()

def test_gui_recognition_results(window):
    det = enter_detection(window)
    panel, dialog, getter = det.recognition, det.dialog, det.point_getter
    assert panel.result_stack.currentIndex() == 0  # 尚未開始：預留畫面

    det.recognition.btn_start.click()  # 沒有規則：提示建立規則
    assert panel.result_stack.currentIndex() == 1 and not panel.results.empty.isHidden()
    det.recognition.btn_end.click()

    low = window.rule_engine.add(preset_rules()[0])  # 低頭
    low.hold_s = 0
    tilt = window.rule_engine.add(preset_rules()[3])  # 肩膀傾斜
    off = window.rule_engine.add(preset_rules()[7])  # 離座（停用：不顯示）
    off.enabled = False
    getter.update(frame_with(UPRIGHT))
    det.recognition.btn_start.click()
    assert dialog.isVisible() and list(panel.results.rows) == [low.id, tilt.id]
    assert "需要初始點" in panel.results.rows[low.id][1].text()
    assert not panel.result_hint.isHidden() and "初始點" in panel.result_hint.text()

    getter.set_baseline()
    det.update_recognition(force=True)
    assert "正常" in panel.results.rows[low.id][1].text() and panel.result_hint.isHidden()
    assert "目前姿勢正常" in dialog.summary.text()

    getter.update(frame_with({**UPRIGHT, "nose": (0.5, 0.6)}))  # 低頭 -> 立即觸發
    det.update_recognition(force=True)
    frame, title, detail = panel.results.rows[low.id]
    assert "觸發" in title.text() and frame.property("state") == "recording"
    assert low.message in detail.text() and "已觸發 1 次" in detail.text()
    assert "觸發中：低頭" in dialog.summary.text()
    assert dialog.results.rows[low.id][1].text() == title.text()  # 兩處顯示一致

    det.recognition.btn_end.click()  # 結束：顯示本次統計
    assert not dialog.isVisible() and "本次辨識結果" in panel.result_title.text()
    assert "觸發 1 次" in panel.results.rows[low.id][2].text()
    assert panel.results.rows[tilt.id][2].text() == "未觸發"

    det.btn_back.click()  # 離開第 2 頁：回到預留畫面
    assert panel.result_stack.currentIndex() == 0

# ---------- 手動測試 ----------


def print_errors(*reporters) -> None:
    for reporter in reporters:
        for rec in reporter.now_error_log.records():
            print("[彈窗]", rec)
        reporter.now_error_log.clear()  # 介面顯示完彈窗後清空


def open_first_camera() -> CameraSetting:
    """模擬第 1 頁：掃描並開啟第一台相機，失敗時印出錯誤並結束。"""
    cam = CameraSetting()
    cameras = cam.scan_cameras()
    print("可用相機：")
    for index, name in cameras.items():
        print(f"  {index}: {name}")
    if not (cameras and cam.select_camera(next(iter(cameras))) and cam.open()):
        print_errors(cam)
        sys.exit(1)
    print(f"已開啟：{cam.camera_index} - {cam.camera_name}")
    return cam


def demo_camera(preview: bool) -> None:
    cam = open_first_camera()
    while preview:
        ok, frame = cam.read_frame()
        if not ok:
            break
        cv2.imshow(f"{cam.camera_name} (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    cam.close()
    print_errors(cam)


def demo_skeleton() -> None:
    """按鍵：1/2/3 切換模型、u 全身/上半身、m 鏡像、r 旋轉、+/- 亮度、0 恢復預設、q 離開。"""
    camera = open_first_camera()
    with SkeletonDetection(camera) as page:
        for group in page.params.groups():  # 正式介面依這些描述產生元件
            print(f"[{group}]")
            for spec in page.params.specs(group):
                if spec.is_dropdown:
                    widget = "下拉：" + " / ".join(label for _, label in spec.options)
                else:
                    widget = f"數字：{spec.min} ~ {spec.max}，間隔 {spec.step}"
                print(f"  {spec.label:<12} = {spec.display_value:<14} {widget}")
        print(demo_skeleton.__doc__)

        toggles = {ord("u"): ("landmark_set", {"full": "upper", "upper": "full"}),
                   ord("m"): ("mirror", {True: False, False: True}),
                   ord("r"): ("rotation", {0: 90, 90: 180, 180: 270, 270: 0})}
        while True:
            f = page.process_frame()
            if f.frame is not None:
                text = f"{page.params.spec('model').display_value}  {f.inference_ms:.0f} ms"
                cv2.putText(f.frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Skeleton detection (q to quit)", f.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("1"), ord("2"), ord("3")):
                page.set_param("model", MODEL_VARIANTS[key - ord("1")][0])
            elif key in toggles:
                name, cycle = toggles[key]
                page.set_param(name, cycle[page.params[name]])
            elif key in (ord("+"), ord("-")):
                page.set_param("brightness", page.params["brightness"] + (10 if key == ord("+") else -10))
            elif key == ord("0"):
                page.reset()
            print_errors(page)
        cv2.destroyAllWindows()
    camera.close()


def demo_record(seconds: float, use_csv: bool) -> None:
    store_cls = CsvStore if use_csv else SqliteStore
    camera = open_first_camera()
    with SkeletonDetection(camera) as page, store_cls() as store:
        session_id = store.start_session(camera.camera_name, page.params.values())
        print(f"開始記錄 {session_id}（{store_cls.__name__}，{seconds} 秒）")
        end = time.time() + seconds
        while time.time() < end:
            store.record(page.process_frame())
        store.end_session()

        df = store.load_session(session_id)
        detected = df["pose_index"].notna()
        print(f"共 {df['frame_index'].nunique()} 幀，偵測到人的幀 {detected.sum()}，欄位 {len(df.columns)} 個")
        if detected.any():
            print(df.loc[detected, ["elapsed_s", "nose_x", "nose_y", "left_shoulder_wx",
                                    "left_shoulder_wy"]].head())
        print("資料位置：", store.data_dir)
        print_errors(page, store)
    camera.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手動測試（需要相機）")
    sub = parser.add_subparsers(dest="demo", required=True)
    p_camera = sub.add_parser("camera", help="第 1 頁：列出並開啟相機")
    p_camera.add_argument("--preview", action="store_true", help="開啟預覽視窗")
    sub.add_parser("skeleton", help="第 2 頁：即時骨架偵測")
    p_record = sub.add_parser("record", help="錄製關鍵點資料並讀回")
    p_record.add_argument("seconds", nargs="?", type=float, default=5)
    p_record.add_argument("--csv", action="store_true", help="改存 CSV（預設 SQLite）")
    args = parser.parse_args()

    if args.demo == "camera":
        demo_camera(args.preview)
    elif args.demo == "skeleton":
        demo_skeleton()
    else:
        demo_record(args.seconds, args.csv)
