# 開發環境建置說明

本專案用 MediaPipe Pose Landmarker 做坐姿分析，並以 PyQt5 提供操作介面。這份文件說明怎麼從零建好開發環境，並確認環境可以使用。系統功能與架構請看 [README.md](../README.md)。

## 1. 版本一覽

| 項目 | 版本 | 備註 |
|---|---|---|
| OS | Windows 10/11 x64（已驗證 Windows 11） | 相機名稱需要 Windows（DirectShow）；其他系統可執行，但相機會顯示為「相機 N」 |
| Conda | Miniconda / Anaconda 皆可 | 只用 `conda-forge` 通道 |
| Python | **3.11**（已驗證 3.11.16） | mediapipe 0.10.21 支援 3.9–3.12 |
| mediapipe | **0.10.21**（固定） | 本專案只使用 Tasks API（`PoseLandmarker`） |
| numpy | 1.26.x（`<2`） | mediapipe 0.10.21 不相容 numpy 2 |
| opencv-contrib-python | 4.11.x（`<4.12`） | 由 mediapipe 帶入；4.12 以上需要 numpy 2 |
| PyQt5 | 5.15.x（已驗證 5.15.11 / Qt 5.15.2） | 操作介面 |
| pygrabber | 0.2（僅 Windows） | 取得相機名稱；取不到時不影響其他功能 |
| pandas / matplotlib | 2.x / 3.x | 讀取紀錄與分析 |
| pytest / ruff | 8 以上 / 0.6 以上 | 測試與靜態檢查 |
| 姿勢模型 | `pose_landmarker_{lite,full,heavy}.task` | 另外下載，不進版控 |

相依套件定義在：

- `environment.yml`：conda 環境（Python 版本與通道），pip 套件會引用 `requirements.txt`
- `requirements.txt`：主要 pip 套件與版本範圍（**要新增套件就改這個檔案**）
- `requirements.lock.txt`：已驗證可用的完整版本清單，由 `scripts/update_lock.py` 產生，需要完全重現環境時使用

## 2. 建置步驟

以下指令在 **Anaconda Prompt** 或已執行過 `conda init powershell` 的 PowerShell 中，於專案根目錄執行。

### 2.1 建立環境

```powershell
# 做法 A：用 environment.yml（建議）
conda env create -f environment.yml
```

如果出現 `CondaToSNonInteractiveError: Terms of Service have not been accepted`，代表你的 conda 全域設定還包含 Anaconda 預設通道（`environment.yml` 裡的 `nodefaults` 無法略過這項檢查）。擇一處理：

```powershell
# 做法 B：不改全域設定，只從 conda-forge 建立（已驗證可用）
conda create -n posture python=3.11 pip --override-channels -c conda-forge -y
conda activate posture
pip install -r requirements.txt

# 做法 C：全域改成只用 conda-forge（之後做法 A 就能直接使用）
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict

# 做法 D：接受 Anaconda 服務條款（組織超過 200 人且為商業用途時需付費授權，請先確認）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

需要跟已驗證環境完全一致時，做法 B 的最後一步改用：

```powershell
pip install -r requirements.lock.txt
```

環境已經存在時，`conda env create` 會失敗。要依最新的設定檔更新，請改用：

```powershell
conda env update -f environment.yml --prune
```

### 2.2 啟用環境

```powershell
conda activate posture
```

使用 VS Code 時，按 `Ctrl+Shift+P` → `Python: Select Interpreter`，選擇 `posture` 環境。否則編輯器會顯示「Import "cv2" could not be resolved」之類的警告。

### 2.3 下載模型

```powershell
python scripts/download_models.py            # 下載全部（lite / full / heavy，約 45 MB）
python scripts/download_models.py full       # 只下載指定變體
```

模型會存到 `models/`，這個資料夾已經列入 `.gitignore`。第 2 頁的模型下拉選單三種都會用到，所以請全部下載；缺少的變體在介面上選擇時會跳出「找不到骨架模型檔」。下載中途中斷不會留下損壞的檔案，重新執行即可。

| 變體 | 大小 | 特性 | 推論時間（參考，筆電 CPU） |
|---|---|---|---|
| lite | 約 6 MB | 最快，精度最低，適合低階裝置 | 約 17 ms/幀 |
| full | 約 9 MB | 預設，速度與精度較平衡 | 約 26 ms/幀 |
| heavy | 約 31 MB | 精度最高，CPU 上較慢（介面約 10 FPS） | 約 96 ms/幀 |

### 2.4 驗證環境

```powershell
python scripts/verify_env.py           # 檢查套件版本、介面套件、模型檔，並做一次模型推論
python scripts/verify_env.py --camera  # 另外開攝影機即時畫出關鍵點，按 q 離開
```

預期輸出（`absl` / `TensorFlow Lite` 的 INFO、WARNING 訊息可以忽略）：

```
python    3.11.16
mediapipe 0.10.21
opencv    4.11.0
numpy     1.26.4
PyQt5     5.15.11（Qt 5.15.2）
pygrabber 可用，偵測到相機：USB2.0 HD UVC WebCam
模型檔：lite / full / heavy 皆已下載
模型載入與推論：OK
```

最後啟動介面與自動測試，確認整套系統可以運作：

```powershell
python run/user_GUI.py   # 操作介面
pytest                   # 自動測試（約 2 分鐘；沒有相機時會略過需要相機的測試）
```

## 3. 日常開發

```powershell
conda activate posture
python run/user_GUI.py       # 執行介面
pytest                       # 執行測試（設定在 pyproject.toml）
pytest -k rule               # 只跑名稱含 rule 的測試
ruff check run scripts       # 靜態檢查
```

- pytest 的設定在 `pyproject.toml`：測試檔為 `run/test.py`，並把 `run/` 加入匯入路徑，所以程式之間以 `from camera_setting import ...` 的方式互相匯入。
- 介面測試使用 Qt 的 offscreen 模式（`QT_QPA_PLATFORM=offscreen`），不會開出視窗。
- 程式執行時會自動建立 `data/`（規則與紀錄）和 `logs/`（錯誤紀錄），不需要手動建立。

### 新增或升級套件

1. 把套件加進 `requirements.txt`，並寫上版本範圍
2. `pip install -r requirements.txt`
3. 跑 `python scripts/verify_env.py` 和 `pytest`，確認沒有東西壞掉
4. 重新產生鎖定檔：`python scripts/update_lock.py`
5. `requirements.txt` 與 `requirements.lock.txt` 一起 commit

> 請不要用 `pip freeze > requirements.lock.txt`：conda 安裝的套件會被記成 `套件 @ file:///...` 的本機路徑，其他電腦無法安裝；而且 PowerShell 5.1 的 `>` 會把檔案寫成 UTF-16。`update_lock.py` 會避開這兩個問題。

### 移除或重建環境

```powershell
conda deactivate
conda env remove -n posture
```

## 4. 套件使用注意

### MediaPipe

- **只使用 Tasks API**（`mediapipe.tasks.python.vision.PoseLandmarker`），不要使用舊版 `mp.solutions.pose`。新版 mediapipe 已經移除舊版 API，依賴它之後會很難升級。
- 本專案以 `RunningMode.VIDEO` 搭配 `detect_for_video(image, timestamp_ms)` 處理即時影像，時間戳必須嚴格遞增（見 `SkeletonDetection.process_frame()`）。
- 模型以 `model_asset_buffer=path.read_bytes()` 載入，而不是 `model_asset_path`，避免專案路徑含中文時 MediaPipe 開檔失敗。
- OpenCV 讀進來的影像是 **BGR**，交給 MediaPipe 前要先轉成 RGB（`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`）。
- 輸出有兩組座標：`pose_landmarks`（正規化影像座標）與 `pose_world_landmarks`（以髖部中心為原點的公尺座標）。本專案另外換算出「中心座標」（像素）供介面與辨識規則使用，說明見 README 的「座標系統」。
- 總共 33 個關鍵點，坐姿分析常用的編號：鼻 0、左右耳 7/8、左右肩 11/12、左右髖 23/24。

### OpenCV

- 在 Windows 上用 `cv2.CAP_DSHOW`（DirectShow）開啟相機，開啟速度比預設的 MSMF 快。pygrabber 列出的裝置順序與 DirectShow 的相機編號一致，所以可以對應出相機名稱。
- DirectShow 回報的 FPS 不一定是相機實際的幀率。

### PyQt5

- 使用 Fusion 風格加上自訂樣式表（`run/gui_style.py`），各台電腦的外觀會一致。
- **不要使用 `QMessageBox`**：它在 offscreen 模式顯示時會讓程式崩潰，導致自動測試失敗。需要彈出訊息時請改用 `QDialog`（參考 `user_GUI.py` 的 `ErrorDialog`）。
- 影像更新目前在主執行緒上以 `QTimer` 進行。如果之後要改用執行緒，注意 `cv2.VideoCapture` 不是執行緒安全的，讀影像和設定相機屬性要在同一個執行緒上進行。

## 5. 常見問題

| 狀況 | 原因與解法 |
|---|---|
| `conda` 不是可辨識的指令 | 改用 Anaconda Prompt，或在 PowerShell 執行 `conda init powershell` 後重開終端機 |
| `CondaToSNonInteractiveError` | 參考 2.1 的做法 B、C 或 D |
| `conda env create` 顯示環境已存在 | 改用 `conda env update -f environment.yml --prune` |
| `pip install -r requirements.lock.txt` 出現 `file:///...` 找不到 | 鎖定檔是用 `pip freeze` 產生的，請改用 `python scripts/update_lock.py` 重新產生 |
| `numpy.core.multiarray failed to import` 或其他跟 numpy 2 有關的錯誤 | 有東西把 numpy 升到 2.x，執行 `pip install "numpy<2"` |
| `cv2` 相關錯誤或 `imshow` 失效 | 同時裝了 `opencv-python` 和 `opencv-contrib-python`。先執行 `pip uninstall opencv-python opencv-contrib-python -y`，再執行 `pip install -r requirements.txt` |
| `qt.qpa.plugin: Could not load the Qt platform plugin "windows"` | PyQt5 安裝不完整，執行 `pip install --force-reinstall "PyQt5>=5.15,<5.16"` |
| VS Code 顯示 `Import "cv2" could not be resolved` | 編輯器沒有使用 `posture` 環境，參考 2.2 選擇直譯器 |
| 終端機的中文變成亂碼 | Windows 主控台編碼問題。先執行 `chcp 65001`，或設定環境變數 `PYTHONIOENCODING=utf-8` |
| 找不到模型 / 介面選模型時跳出「找不到骨架模型檔」 | 執行 `python scripts/download_models.py` |
| 相機打不開 | 確認 Windows「設定 → 隱私權 → 相機」有允許桌面應用程式使用相機，並關閉其他正在使用鏡頭的程式 |
| 相機顯示為「相機 0」而不是裝置名稱 | pygrabber 未安裝或無法使用（非 Windows 系統也是如此），不影響其他功能 |
| 從封面進入第 2 頁要等好幾秒 | 正常現象，需要偵測相機支援的解析度並載入模型，約 6 秒 |
