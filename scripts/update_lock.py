"""重新產生 requirements.lock.txt（目前環境已驗證可用的完整套件版本）。

用法（在已啟用的 posture 環境中）：
    python scripts/update_lock.py

不直接用 `pip freeze > requirements.lock.txt` 的原因：
- conda 安裝的套件（例如 packaging）會被記成 `套件 @ file:///...` 的本機路徑，
  其他電腦無法安裝；`pip list --format=freeze` 一律輸出 `套件==版本`
- PowerShell 5.1 的 `>` 會寫成 UTF-16，這裡固定寫成 UTF-8
"""
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

LOCK = Path(__file__).resolve().parent.parent / "requirements.lock.txt"
EXCLUDE = ("pip", "setuptools", "wheel")  # 由 conda 環境提供


def main() -> None:
    cmd = [sys.executable, "-m", "pip", "list", "--format=freeze"]
    for name in EXCLUDE:
        cmd += ["--exclude", name]
    packages = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.split()
    header = (
        f"# 由 scripts/update_lock.py 產生（{date.today()}，{platform.system()} {platform.version()} / "
        f"Python {platform.python_version()}），用於完全重現環境\n"
        "# pip install -r requirements.lock.txt\n"
    )
    LOCK.write_text(header + "\n".join(sorted(packages, key=str.lower)) + "\n", encoding="utf-8")
    print(f"已寫入 {LOCK}（{len(packages)} 個套件）")


if __name__ == "__main__":
    main()
