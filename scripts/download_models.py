"""下載 MediaPipe Pose Landmarker 模型檔到 models/。

第 2 頁（run/Skelenton_detection.py）的模型下拉選單提供 lite / full / heavy，
因此預設全部下載。

用法：
    python scripts/download_models.py            # 下載全部變體
    python scripts/download_models.py lite heavy # 只下載指定變體
"""
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)
VARIANTS = ("lite", "full", "heavy")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    variants = sys.argv[1:] or list(VARIANTS)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        sys.exit(f"未知的模型變體：{', '.join(unknown)}（可用：{', '.join(VARIANTS)}）")
    MODELS_DIR.mkdir(exist_ok=True)
    for v in variants:
        dest = MODELS_DIR / f"pose_landmarker_{v}.task"
        if dest.exists():
            print(f"已存在，略過：{dest}")
            continue
        # 先寫入暫存檔，完整下載後才改名，避免中斷時留下損壞的模型被當成已存在
        part = dest.with_suffix(".task.part")
        print(f"下載 {v} -> {dest}")
        try:
            urllib.request.urlretrieve(URL.format(v=v), part)
        except Exception as e:
            part.unlink(missing_ok=True)
            sys.exit(f"下載 {v} 失敗：{e}")
        part.replace(dest)
    print("完成")


if __name__ == "__main__":
    main()
