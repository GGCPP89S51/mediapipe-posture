"""檢查環境是否可正常執行姿勢偵測與操作介面。

用法：
    python scripts/verify_env.py           # 檢查套件版本、介面套件、模型檔與模型推論
    python scripts/verify_env.py --camera  # 另外開啟攝影機即時預覽（按 q 離開）
"""
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL = MODELS_DIR / "pose_landmarker_full.task"


def build_landmarker(mode):
    options = vision.PoseLandmarkerOptions(
        # 用 buffer 讀入，避免路徑含中文時 MediaPipe 開檔失敗
        base_options=mp_tasks.BaseOptions(model_asset_buffer=MODEL.read_bytes()),
        running_mode=mode,
    )
    return vision.PoseLandmarker.create_from_options(options)


def check_versions() -> None:
    print(f"python    {sys.version.split()[0]}")
    print(f"mediapipe {mp.__version__}")
    print(f"opencv    {cv2.__version__}")
    print(f"numpy     {np.__version__}")


def check_gui() -> None:
    """操作介面需要 PyQt5；Windows 取得相機名稱需要 pygrabber。"""
    try:
        from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
    except ImportError as e:
        sys.exit(f"PyQt5 無法載入：{e}\n請執行 pip install -r requirements.txt")
    print(f"PyQt5     {PYQT_VERSION_STR}（Qt {QT_VERSION_STR}）")
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            names = FilterGraph().get_input_devices()
            print(f"pygrabber 可用，偵測到相機：{', '.join(names) if names else '無'}")
        except Exception as e:
            print(f"pygrabber 無法使用（相機會顯示為「相機 N」，不影響其他功能）：{e}")


def check_model_files() -> None:
    missing = [v for v in ("lite", "full", "heavy")
               if not (MODELS_DIR / f"pose_landmarker_{v}.task").exists()]
    if missing:
        print(f"缺少模型：{', '.join(missing)}（介面選到時會跳出錯誤），"
              f"請執行 python scripts/download_models.py {' '.join(missing)}")
    else:
        print("模型檔：lite / full / heavy 皆已下載")


def check_model() -> None:
    if not MODEL.exists():
        sys.exit(f"找不到模型：{MODEL}\n請先執行 python scripts/download_models.py")
    with build_landmarker(vision.RunningMode.IMAGE) as landmarker:
        blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((256, 256, 3), np.uint8))
        landmarker.detect(blank)
    print("模型載入與推論：OK")


def run_camera() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("無法開啟攝影機 0")
    with build_landmarker(vision.RunningMode.VIDEO) as landmarker:
        ts = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
            )
            ts += 33
            h, w = frame.shape[:2]
            for pose in result.pose_landmarks:
                for lm in pose:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            cv2.imshow("verify_env (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    check_versions()
    check_gui()
    check_model_files()
    check_model()
    if "--camera" in sys.argv:
        run_camera()
