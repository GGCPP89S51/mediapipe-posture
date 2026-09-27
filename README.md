# 姿態辨識系統（MediaPipe 坐姿分析）

以 MediaPipe Pose Landmarker 即時偵測人體 33 個關鍵點。使用者先設定「坐正」時的姿勢作為初始點，再用可自訂的辨識規則（例如低頭、駝背、肩膀傾斜）判斷目前的坐姿，並記錄關鍵點資料供後續分析。

## 功能

- **相機選擇**：自動掃描相機並顯示裝置名稱；設備異常時跳出「請檢查設備」視窗，程式不中斷
- **即時骨架偵測**：可在介面即時切換模型（Lite / Full / Heavy）、信心門檻、解析度、旋轉、亮度等參數
- **關鍵點數值**：以「中心座標」（像素，畫面中心為原點）即時顯示 33 個點，也可切換影像座標或世界座標（公尺）
- **初始點**：記錄坐正時的姿勢，並即時顯示每個點相對初始點的位移
- **辨識規則**：在介面建立規則，條件可用點的位移、兩點距離、連線角度、三點夾角或「沒有人」，並附 8 個常見坐姿範本
- **姿態辨識**：依規則即時判斷，顯示觸發提醒，結束後統計每條規則的觸發次數與累計秒數
- **資料記錄**：辨識期間的關鍵點存成 SQLite（或 CSV），可用 SQL / pandas 分析

## 快速開始

```powershell
conda env create -f environment.yml    # 出現 CondaToSNonInteractiveError 時請看 docs/ENVIRONMENT.md 2.1
conda activate posture
python scripts/download_models.py      # 下載 lite / full / heavy 三種模型（約 45 MB）
python run/user_GUI.py                 # 啟動操作介面
```

環境版本限制、各種建置方式與常見問題：[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)

## 使用流程

1. **封面**：從下拉選單選擇相機，按「進入系統」。進入時會偵測相機支援的解析度並載入模型，約需 6 秒。
2. **第 2 頁**：左側是即時影像與骨架，中間可以調整參數；同時會彈出「目前關鍵點數值」視窗。
3. **設定初始點**：坐正後，在「目前關鍵點數值」視窗下方按「設為初始點」。會彈出「已儲存的初始點」視窗，其中「目前位移」欄即時顯示每個點相對坐正時的位移。
4. **建立規則**：按「姿態辨識」卡片右上角的「添加辨識規則」，可從範本新增，或自行設定條件。每個條件都會顯示目前數值，方便決定門檻。規則會自動存到 `data/rules.json`。
5. **開始辨識**：回到第 2 頁按「開始辨識」，右側與「目前辨識結果」視窗會即時顯示各規則狀態，觸發時顯示提醒。按「結束」停止並存檔，右側會顯示本次統計。

> 記錄期間會鎖定參數、規則與「更換相機」，確保同一段資料的設定一致。離開第 2 頁時初始點會清除，請重新設定。

## 系統架構

```
┌───────────────────────────── user_GUI.py（PyQt5 介面）─────────────────────────────┐
│  CoverPage          DetectionPage                       RulesPage                  │
│  選擇相機      →    影像 / 參數 / 辨識結果        ⇄     規則列表 / 編輯 / 即時預覽   │
│                     KeypointWindow、BaselineWindow、RecognitionDialog（彈出視窗）  │
└──────┬───────────────────┬──────────────────┬──────────────────┬──────────────────┘
       │                   │                  │                  │
 camera_setting.py   Skelenton_detection.py  now_point_get.py   recognition_rules.py
 CameraSetting       SkeletonDetection        NowPointGetter      RuleEngine
 開啟相機、讀影像    參數 + MediaPipe 偵測    最新關鍵點、初始點  判斷規則、統計、存檔
       │                   │                  │
       └──── 影像 ────→    └─ DetectionFrame ─┴──→ read_point_data.py（SqliteStore / CsvStore）
                                                     關鍵點資料存檔
```

每一幀的資料流：

```
CameraSetting.read_frame() → SkeletonDetection.process_frame() → DetectionFrame
    ├→ 畫面顯示（已畫骨架）
    ├→ NowPointGetter.update()   → 關鍵點數值視窗、初始點位移
    └→（辨識中）PointDataStore.record() 存檔
                RuleEngine.evaluate(latest, baseline) → 辨識結果 + RecognitionTracker 統計
```

設計原則：

- **邏輯與介面分離**：`run/` 裡除了 `user_GUI.py`、`gui_style.py` 以外的模組都不依賴 Qt，可以單獨測試，也可以給其他程式使用。
- **錯誤不中斷程式**：各模組繼承 `ErrorReporter`，發生錯誤時不拋出例外，而是回傳 `False` / `None` 並寫入兩種紀錄：
  - `NowErrorLog`：該頁目前尚未處理的錯誤。介面讀取後跳出「請檢查設備」視窗，然後清空。
  - `ErrorLog`：所有錯誤的總紀錄，各頁共用，並寫入 `logs/camera_error.log`。
- **參數描述驅動介面**：第 2 頁的參數以 `Param` 描述（類型、選項、範圍、步進），介面依描述自動產生下拉選單或數字輸入框。新增參數不需修改介面程式。

## 模組說明

| 檔案 | 主要類別 | 職責 |
|---|---|---|
| `run/camera_setting.py` | `CameraSetting`、`ErrorReporter`、`NowErrorLog`、`ErrorLog` | 掃描、選擇、開啟相機，讀取影像；共用的錯誤紀錄機制 |
| `run/Skelenton_detection.py` | `SkeletonDetection`、`Param`、`ParamSet`、`DetectionFrame` | 參數管理（即時套用）、MediaPipe 偵測、影像前處理與骨架繪製 |
| `run/now_point_get.py` | `NowPointGetter`、`KeypointSnapshot`、`KeypointValue` | 保存最新一幀的關鍵點與初始點，計算中心座標與位移 |
| `run/recognition_rules.py` | `RuleEngine`、`Rule`、`Condition`、`RecognitionTracker` | 規則定義、判斷（含持續時間）、統計、存檔與範本 |
| `run/read_point_data.py` | `SqliteStore`、`CsvStore` | 關鍵點資料記錄（批次寫入、失敗重試）與讀取 |
| `run/user_GUI.py` | `MainWindow` 與各頁面、視窗 | PyQt5 操作介面 |
| `run/gui_style.py` | `apply_theme()` | 介面顏色與樣式表 |
| `run/test.py` | — | 整合測試（pytest）與手動測試指令 |

### 常用 API

```python
from camera_setting import CameraSetting
from Skelenton_detection import SkeletonDetection
from now_point_get import NowPointGetter
from recognition_rules import RuleEngine, RuleStatus

camera = CameraSetting()
camera.scan_cameras()                          # {0: "USB2.0 HD UVC WebCam"}
camera.select_camera(0) and camera.open()

page = SkeletonDetection(camera)
page.set_param("model", "heavy")               # 不合法的值會被拒絕並寫入 NowErrorLog

getter = NowPointGetter()
getter.update(page.process_frame())
getter.set_baseline()                          # 坐正時呼叫
getter.latest.get("left_shoulder").cx          # 中心座標（像素）
getter.displacement("nose")                    # (dx, dy)：相對初始點的位移

engine = RuleEngine()                          # 載入 data/rules.json
for result in engine.evaluate(getter.latest, getter.baseline):
    if result.status is RuleStatus.TRIGGERED:
        print(result.rule.name, result.rule.message)
```

## 座標系統

| 座標 | 欄位 | 原點與方向 | 單位 |
|---|---|---|---|
| **中心座標**（介面預設、規則使用） | `cx`, `cy` | 畫面中心為 (0, 0)，x 向右、**y 向上**為正 | 像素，取到小數點後 2 位 |
| 影像座標 | `x`, `y`, `z` | 左上角 (0, 0)、右下角 (1, 1)；z 越小越靠近鏡頭 | 0 ~ 1 |
| 世界座標 | `wx`, `wy`, `wz` | 髖部中心，為 MediaPipe 估計值 | 公尺 |

- 左右以**被拍攝者本人**為準。座標以鏡像前的影像計算，所以開啟「鏡像顯示」時，畫面上看到的左右與 `cx` 的正負相反。
- 中心座標依實際影像大小換算。設定初始點後若更換解析度或旋轉角度，像素位移就不準確，請重新設定初始點。
- 可見度（`visibility`）低於 0.5 的點，在表格中顯示為灰色，也不參與規則判斷。

## 辨識規則

一條規則由一個或多個條件組成，依「全部成立 / 任一成立」判斷，並需持續成立設定的秒數才算**觸發**。

| 指標 | 需要的點 | 比較基準 | 例子 |
|---|---|---|---|
| 點的位置／位移 | 1 | 目前數值、與初始點的差 | 鼻子垂直位移 ≤ -30 px（低頭） |
| 兩點距離 | 2 | 目前數值、差、比例（%） | 兩肩距離比例 ≥ 115%（靠近螢幕） |
| 兩點連線角度 | 2 | 目前數值、差 | 右肩→左肩角度在 ±8° 以外（肩膀傾斜） |
| 三點夾角 | 3 | 目前數值、差 | 耳-肩-髖夾角 |
| 畫面中沒有人 | 0 | — | 持續 10 秒（離座） |

- 除了 33 個關鍵點，也可以選擇虛擬點：兩肩中點、兩耳中點、兩髖中點。
- 規則狀態：正常、條件成立中、觸發、無法判斷（點看不清楚）、需要初始點、未啟用。
- 內建範本（門檻為參考值，請依鏡頭位置調整）：低頭、駝背、身體前傾（靠近螢幕）、肩膀傾斜、頭部側傾、身體左右偏移、手撐下巴、離座。

## 資料與檔案

| 路徑 | 內容 | 進版控 |
|---|---|---|
| `data/records/posture.db` | 關鍵點紀錄（SQLite） | 否 |
| `data/records/<session_id>.csv` / `.json` | 關鍵點紀錄（使用 `CsvStore` 時） | 否 |
| `data/rules.json` | 辨識規則 | 是（可共用規則設定） |
| `logs/camera_error.log` | 錯誤總紀錄 | 否 |
| `models/*.task` | MediaPipe 模型 | 否，執行 `scripts/download_models.py` 下載 |

SQLite 資料表：

- `sessions`：每次「開始辨識 ~ 結束」一列，包含 `session_id`、開始 / 結束時間、相機名稱、幀數，以及 `params`（當時第 2 頁所有參數的 JSON）。
- `frames`：每一幀每個人一列，包含 `frame_index`、`unix_time`、`elapsed_s`、`inference_ms`、`pose_index`，以及每個關鍵點的 8 個欄位：`<名稱>_x`、`_y`、`_z`、`_vis`、`_pres`、`_wx`、`_wy`、`_wz`（例如 `left_shoulder_wy`）。畫面中沒有人的幀也會記錄一列，關鍵點欄位為空。

```python
from read_point_data import SqliteStore
store = SqliteStore()
store.list_sessions()                          # 所有紀錄摘要
df = store.load_session("20260924_132501_551972")   # pandas DataFrame
```

資料量：每秒 30 幀約為每分鐘 8 MB。長時間記錄可使用 `SqliteStore(min_interval_s=0.2)` 降低記錄頻率。

## 測試

```powershell
pytest                                     # 全部自動測試（約 2 分鐘）
pytest -k rule                             # 只跑名稱含 rule 的測試
python run/test.py camera --preview        # 手動：列出相機並預覽
python run/test.py skeleton                # 手動：即時骨架，用按鍵切換參數
python run/test.py record 10 [--csv]       # 手動：錄 10 秒並讀回
```

- 需要相機的測試在沒有相機的電腦上會自動略過。
- 介面測試使用 Qt offscreen 模式，不會開出視窗。**不要在介面使用 `QMessageBox`**，它在 offscreen 模式顯示時會讓程式崩潰；請改用 `QDialog`（參考 `ErrorDialog`）。
- 測試資料一律寫入暫存資料夾，不會動到 `data/` 與 `logs/`。

## 開發指引

- **新增第 2 頁參數**：在 `SkeletonDetection._build_params()` 加一個 `Param`，並在 `_apply()` 或 `process_frame()` 中使用它。介面會自動產生對應的元件。
- **新增規則指標**：在 `recognition_rules.py` 的 `Metric` 加一個項目，補上 `METRIC_POINT_LABELS`、`METRIC_DEFAULT_POINTS`、`METRIC_REFERENCES`、`METRIC_HELP`，並在 `_measure()` 實作計算方式。規則頁的欄位會依這些設定自動調整。
- **新增規則範本**：在 `preset_rules()` 加入一條 `Rule`。
- **調整外觀**：顏色集中在 `gui_style.py` 開頭。按鈕樣式用 `button.setProperty("variant", ...)` 指定，可用 `primary`、`outline`、`danger`、`ghost`。
- **新增錯誤代碼**：在各模組自己的 `*ErrorCode` Enum 加入，透過 `self._report(...)` 回報，介面會自動顯示。
- **新增套件**：修改 `requirements.txt`，再執行 `python scripts/update_lock.py` 更新鎖定檔（不要用 `pip freeze`），流程見 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) 第 3 節。

## 已知限制與待辦

- 從封面進入第 2 頁約需 6 秒，主要花在偵測相機支援的解析度。可改為快取偵測結果或在背景執行。
- 影像處理在主執行緒進行，使用 Heavy 模型（約 96 ms/幀）時畫面約 10 FPS，操作會稍微卡頓。
- 使用 DirectShow 時，相機回報的 FPS 不一定是實際值。
- 規則觸發紀錄目前只保存在記憶體（本次統計），尚未寫入資料庫。
- 初始點只保存在記憶體，離開第 2 頁或關閉程式後需重新設定。
- 刪除規則沒有確認視窗，也不能復原。

## 目錄結構

```
environment.yml           conda 環境定義
requirements.txt          pip 相依套件（版本範圍）
requirements.lock.txt     已驗證的完整版本鎖定
pyproject.toml            pytest 設定
run/
  user_GUI.py             操作介面（PyQt5）
  gui_style.py            介面主題（顏色、樣式表）
  camera_setting.py       相機選擇 / 開啟、錯誤紀錄機制
  Skelenton_detection.py  骨架偵測與可即時調整的參數
  now_point_get.py        最新關鍵點數值、初始點與位移
  recognition_rules.py    辨識規則：條件、判斷、統計、存檔、範本
  read_point_data.py      關鍵點資料儲存（SQLite / CSV）
  test.py                 整合測試與手動測試
scripts/
  download_models.py      下載 MediaPipe 模型
  verify_env.py           檢查環境（--camera 開啟相機預覽）
  update_lock.py          重新產生 requirements.lock.txt
docs/
  ENVIRONMENT.md          環境建置說明
data/                     規則與紀錄（records/ 不進版控）
models/                   模型檔（不進版控）
logs/                     錯誤紀錄（不進版控）
```
