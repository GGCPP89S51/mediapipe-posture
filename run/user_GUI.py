"""姿態辨識系統操作介面（PyQt5）。

封面：選擇相機 -> 第 2 頁：
┌─────────────────────────────────────────────────────────┐
│ 標題列：Logo、系統名稱、相機名稱、記錄狀態、更換相機         │
├──────────────────┬────────────┬──────────────┤
│ 即時影像 + 骨架   │ 參數設定    │ 姿態辨識（待設計）│
│                  │（依 Param   │ [ 開始辨識 ]   │
│                  │  自動產生）  │ [結束] [重置]  │
└──────────────────┴────────────┴──────────────┘
開始辨識：開始記錄關鍵點資料，並依辨識規則即時判斷姿勢，結果顯示在右側辨識區與「目前辨識結果」視窗；
記錄期間鎖定參數，確保整段資料參數一致。結束後辨識區顯示本次各規則的觸發次數與累計秒數。
結束：停止記錄並存檔。重置：參數恢復預設。
添加辨識規則（姿態辨識卡片右上角）：切換到辨識規則頁，規則邏輯在 recognition_rules.py。
各頁的 NowErrorLog 會以「請檢查設備」視窗顯示，程式不中斷。
外觀（顏色、樣式）集中在 gui_style.py。

執行：python run/user_GUI.py
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import gui_style as theme
import numpy as np
from camera_setting import CameraSetting, ErrorReporter
from now_point_get import (
    LANDMARK_NAMES_ZH,
    KeypointSnapshot,
    NowPointGetter,
    displacement,
)
from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from read_point_data import LANDMARK_NAMES, PointDataStore, SqliteStore
from recognition_rules import (
    METRIC_HELP,
    METRIC_POINT_LABELS,
    METRIC_REFERENCES,
    POINT_CHOICES,
    RULES_PATH,
    Axis,
    Condition,
    ConditionResult,
    Metric,
    RangeMode,
    RecognitionTracker,
    Reference,
    Rule,
    RuleEngine,
    RuleResult,
    RuleStatus,
    preset_rules,
)
from Skelenton_detection import ParamKind, SkeletonDetection

APP_TITLE = "姿態辨識系統"
APP_SUBTITLE = "MediaPipe 即時骨架偵測 · 坐姿分析"
CONFIRM_TEXT = "進入系統  →"


# ---------- 共用元件 ----------


def make_button(text: str, variant: str | None = None, height: int = 40) -> QPushButton:
    """variant：primary（主要）/ outline（外框）/ danger（危險）/ ghost（次要），樣式在 gui_style.py。"""
    button = QPushButton(text)
    button.setMinimumHeight(height)
    button.setCursor(Qt.PointingHandCursor)
    if variant:
        button.setProperty("variant", variant)
    return button


def make_label(text: str, object_name: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    return label


def make_card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    """圓角卡片，回傳 (卡片, 內容 layout)。"""
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(10)
    if title:
        layout.addWidget(make_label(title, "CardTitle"))
    return frame, layout


def make_header_bar(title: str = APP_TITLE) -> tuple[QFrame, QHBoxLayout]:
    """頁面上方的標題列（Logo + 標題），回傳 (標題列, layout)，可再加入其他元件。"""
    header = QFrame()
    header.setObjectName("Header")
    header.setFixedHeight(60)
    bar = QHBoxLayout(header)
    bar.setContentsMargins(20, 0, 16, 0)
    bar.setSpacing(12)
    bar.addWidget(LogoMark(34), 0, Qt.AlignVCenter)
    bar.addWidget(make_label(title, "HeaderTitle"), 0, Qt.AlignVCenter)
    return header, bar


class Pill(QLabel):
    """狀態膠囊：idle（待機）/ recording（記錄中）/ saved（已儲存）。"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Pill")
        self.set_state("待機中", "idle")

    def set_state(self, text: str, state: str) -> None:
        self.setText(text)
        if self.property("state") != state:
            self.setProperty("state", state)
            self.style().unpolish(self)  # 屬性改變後重新套用樣式
            self.style().polish(self)


# Logo 坐姿骨架的正規化座標：頭、肩、髖、膝、腳、手
LOGO_JOINTS = {"head": (0.42, 0.25), "neck": (0.43, 0.38), "hip": (0.40, 0.62),
               "knee": (0.68, 0.62), "foot": (0.68, 0.84), "hand": (0.62, 0.52)}
LOGO_BONES = (("neck", "hip"), ("hip", "knee"), ("knee", "foot"), ("neck", "hand"))


class LogoMark(QWidget):
    """以程式繪製的 Logo：漸層圓角方塊中的坐姿骨架。"""

    def __init__(self, size: int = 88) -> None:
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:
        s = self.width()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, s, s)
        gradient.setColorAt(0, QColor(theme.ACCENT))
        gradient.setColorAt(1, QColor(theme.ACCENT_2))
        p.setPen(Qt.NoPen)
        p.setBrush(gradient)
        p.drawRoundedRect(0, 0, s, s, s * 0.28, s * 0.28)

        def pt(name):
            x, y = LOGO_JOINTS[name]
            return QPointF(x * s, y * s)

        ink = QColor(theme.ON_ACCENT)
        p.setPen(QPen(ink, s * 0.07, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for a, b in LOGO_BONES:
            p.drawLine(pt(a), pt(b))
        p.setPen(Qt.NoPen)
        p.setBrush(ink)
        p.drawEllipse(pt("head"), s * 0.09, s * 0.09)
        p.setBrush(QColor("#ffffff"))
        for name in ("hip", "knee", "hand"):
            p.drawEllipse(pt(name), s * 0.035, s * 0.035)


# ---------- 錯誤視窗 ----------


class ErrorDialog(QDialog):
    """「請檢查設備」視窗。非強制回應，開著時新錯誤會附加到清單。

    不用 QMessageBox：它在 Qt offscreen 平台（自動測試）顯示時會當掉。
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("請檢查設備")
        self.setModal(False)  # 不阻塞畫面更新
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(500, 260)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.addWidget(make_label("⚠  請檢查設備", "ErrorTitle"))
        layout.addWidget(make_label("發生以下錯誤，請確認相機連線或參數設定。", "Hint"))
        layout.addWidget(self.details)
        layout.addWidget(buttons)

    def add_records(self, records: list) -> None:
        if not self.isVisible():
            self.details.clear()
        for rec in records:
            self.details.appendPlainText(str(rec))
        self.show()
        self.raise_()


class ErrorNotifier:
    """把各頁 NowErrorLog 的錯誤顯示在 ErrorDialog，顯示後清空。"""

    def __init__(self, parent: QWidget) -> None:
        self.dialog = ErrorDialog(parent)

    def check(self, *reporters: ErrorReporter) -> None:
        records = []
        for reporter in reporters:
            records += reporter.now_error_log.records()
            reporter.now_error_log.clear()
        if records:
            self.dialog.add_records(records)

    @property
    def visible(self) -> bool:
        return self.dialog.isVisible()


# ---------- 封面 ----------


class CoverPage(QWidget):
    camera_ready = pyqtSignal(object)  # 已開啟的 CameraSetting

    def __init__(self, camera: CameraSetting, notifier: ErrorNotifier) -> None:
        super().__init__()
        self.setObjectName("Cover")
        self.setAttribute(Qt.WA_StyledBackground)  # 讓 QSS 背景作用在 QWidget 上
        self.camera = camera
        self._notifier = notifier

        self.combo = QComboBox()
        self.btn_scan = make_button("↻  重新掃描")
        self.btn_ok = make_button(CONFIRM_TEXT, "primary", height=48)
        self.btn_scan.clicked.connect(self.scan)
        self.btn_ok.clicked.connect(self.confirm)

        card = QFrame()
        card.setObjectName("CoverCard")
        card.setFixedWidth(540)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(60)
        shadow.setOffset(0, 16)
        shadow.setColor(QColor(0, 0, 0, 170))
        card.setGraphicsEffect(shadow)

        title = make_label(APP_TITLE, "AppTitle")
        subtitle = make_label(APP_SUBTITLE, "Subtitle")
        for label in (title, subtitle):
            label.setAlignment(Qt.AlignCenter)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.combo, 1)
        row.addWidget(self.btn_scan)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(44, 40, 44, 36)
        inner.setSpacing(0)
        inner.addWidget(LogoMark(88), 0, Qt.AlignHCenter)
        inner.addSpacing(20)
        inner.addWidget(title)
        inner.addSpacing(6)
        inner.addWidget(subtitle)
        inner.addSpacing(36)
        inner.addWidget(make_label("選擇相機", "FieldLabel"))
        inner.addSpacing(8)
        inner.addLayout(row)
        inner.addSpacing(20)
        inner.addWidget(self.btn_ok)
        inner.addSpacing(12)
        hint = make_label("進入後會偵測相機解析度並載入模型，約需數秒", "Hint")
        hint.setAlignment(Qt.AlignCenter)
        inner.addWidget(hint)

        footer = make_label("Powered by MediaPipe Pose Landmarker · PyQt5", "Footer")
        footer.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 16)
        layout.addStretch(2)
        layout.addWidget(card, 0, Qt.AlignHCenter)
        layout.addStretch(3)
        layout.addWidget(footer)

    def scan(self) -> None:
        self.combo.clear()
        for index, name in self.camera.scan_cameras().items():
            self.combo.addItem(f"{index}: {name}", index)
        self.btn_ok.setEnabled(self.combo.count() > 0)
        self._notifier.check(self.camera)

    def confirm(self) -> None:
        index = self.combo.currentData()
        if index is not None and self.camera.select_camera(index) and self.camera.open():
            self.camera_ready.emit(self.camera)
        self._notifier.check(self.camera)


# ---------- 第 2 頁元件 ----------


class VideoView(QLabel):
    """等比例縮放顯示 BGR 影像。"""

    def __init__(self) -> None:
        super().__init__("等待相機畫面…")
        self.setObjectName("VideoView")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)

    def show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(self.size(), Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation)
        self.setPixmap(pixmap)


class ParamPanel(QScrollArea):
    """依 page.params 的描述自動產生元件：下拉選單 / 整數 / 小數，變動時即時套用。"""

    def __init__(self, page: SkeletonDetection, notifier: ErrorNotifier) -> None:
        super().__init__()
        self.page = page
        self._notifier = notifier
        self.widgets: dict[str, QWidget] = {}

        container = QWidget()
        container.setObjectName("ScrollContent")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(4)
        for group in page.params.groups():
            box = QGroupBox(group)
            form = QFormLayout(box)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(6)
            form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for spec in page.params.specs(group):
                widget = self._make_widget(spec)
                label = make_label(spec.label, "FieldLabel")
                if spec.help:
                    label.setToolTip(spec.help)
                    widget.setToolTip(spec.help)
                form.addRow(label, widget)
                self.widgets[spec.key] = widget
            layout.addWidget(box)
        layout.addStretch(1)
        self.setWidget(container)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.refresh()

    def _make_widget(self, spec) -> QWidget:
        key = spec.key
        if spec.is_dropdown:
            widget = QComboBox()
            for _, label in spec.options:
                widget.addItem(label)
            # 用索引對回選項值，避免 tuple 等型別經過 Qt 轉換後變形
            widget.currentIndexChanged.connect(
                lambda i, k=key: self._changed(k, self.page.params.spec(k).options[i][0]))
            return widget
        if spec.kind is ParamKind.INT:
            widget = QSpinBox()
            widget.setRange(int(spec.min), int(spec.max))
            widget.setSingleStep(int(spec.step))
        else:
            widget = QDoubleSpinBox()
            decimals = len(f"{spec.step:g}".partition(".")[2])
            widget.setDecimals(max(decimals, 1))
            widget.setRange(spec.min, spec.max)
            widget.setSingleStep(spec.step)
        widget.setKeyboardTracking(False)  # 輸入完成（Enter / 離開欄位）才套用
        widget.valueChanged.connect(lambda v, k=key: self._changed(k, v))
        return widget

    def _changed(self, key: str, value) -> None:
        if not self.page.set_param(key, value):
            self.refresh(key)  # 被拒絕：元件還原成目前實際的值
        self._notifier.check(self.page)

    def refresh(self, key: str | None = None) -> None:
        """把元件同步成 page.params 的值（不觸發變動事件）。"""
        keys = [key] if key else list(self.widgets)
        for k in keys:
            spec, widget = self.page.params.spec(k), self.widgets[k]
            widget.blockSignals(True)
            if spec.is_dropdown:
                values = [v for v, _ in spec.options]
                if spec.value in values:
                    widget.setCurrentIndex(values.index(spec.value))
            else:
                widget.setValue(spec.value)
            widget.blockSignals(False)


def make_placeholder(icon: str, title: str, hint: str) -> QFrame:
    """虛線框的預留區域。"""
    frame = QFrame()
    frame.setObjectName("Placeholder")
    layout = QVBoxLayout(frame)
    layout.addStretch(1)
    for text, name in ((icon, "PlaceholderIcon"), (title, "FieldLabel"), (hint, "Hint")):
        label = make_label(text, name)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)
    layout.addStretch(1)
    return frame


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class RuleResultList(QWidget):
    """每條啟用中的規則一列：狀態、名稱、進度或提醒訊息。辨識區與「目前辨識結果」視窗共用。"""

    def __init__(self) -> None:
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self.empty = make_label("沒有啟用中的辨識規則\n請點右上角「添加辨識規則」建立", "Hint")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)
        self._layout.addWidget(self.empty)
        self._layout.addStretch(1)
        self.rows: dict[str, tuple[QFrame, QLabel, QLabel]] = {}

    def _ensure_rows(self, rules: list[Rule]) -> None:
        """規則有增減時才重建列，平時只更新文字。"""
        if [r.id for r in rules] == list(self.rows):
            return
        for frame, _, _ in self.rows.values():
            frame.setParent(None)
            frame.deleteLater()
        self.rows = {}
        for rule in rules:
            frame = QFrame()
            frame.setObjectName("ResultRow")
            title = make_label("", "ResultName")
            title.setWordWrap(True)
            detail = make_label("", "Hint")
            detail.setWordWrap(True)
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 8, 12, 8)
            box.setSpacing(2)
            box.addWidget(title)
            box.addWidget(detail)
            self._layout.insertWidget(self._layout.count() - 1, frame)
            self.rows[rule.id] = (frame, title, detail)
        self.empty.setVisible(not rules)

    def show_results(self, results: list[RuleResult], tracker: RecognitionTracker | None = None) -> None:
        shown = [r for r in results if r.rule.enabled]
        self._ensure_rows([r.rule for r in shown])
        for result in shown:
            frame, title, detail = self.rows[result.rule.id]
            state, symbol = status_style(result.status)
            text = f"{symbol}  {result.rule.name}　{result.status.value}"
            if result.status is RuleStatus.PENDING:
                text += f" {result.held_s:.1f} / {result.rule.hold_s:g} 秒"
            title.setText(text)
            lines = []
            if result.triggered:
                lines.append(result.rule.message or "條件已持續成立")
            elif result.status in (RuleStatus.UNKNOWN, RuleStatus.NEED_BASELINE):
                reason = next((c.reason for c in result.conditions if c.reason), "")
                lines.append(reason or result.status.value)
            if tracker is not None and (stats := tracker.get(result.rule.id)).trigger_count:
                lines.append(f"已觸發 {stats.trigger_count} 次 · 累計 {stats.triggered_s:.1f} 秒")
            detail.setText("\n".join(lines))
            detail.setVisible(bool(lines))
            set_state(frame, state)

    def show_summary(self, rules: list[Rule], tracker: RecognitionTracker) -> None:
        """辨識結束後的統計：每條規則觸發次數與累計秒數。"""
        shown = [r for r in rules if r.enabled]
        self._ensure_rows(shown)
        for rule in shown:
            frame, title, detail = self.rows[rule.id]
            stats = tracker.get(rule.id)
            triggered = stats.trigger_count > 0
            title.setText(f"{'⚠' if triggered else '✓'}  {rule.name}")
            detail.setText(f"觸發 {stats.trigger_count} 次 · 累計 {stats.triggered_s:.1f} 秒"
                           if triggered else "未觸發")
            detail.setVisible(True)
            set_state(frame, "recording" if triggered else "saved")


def make_results_view(results: RuleResultList) -> QScrollArea:
    container = QWidget()
    container.setObjectName("ScrollContent")
    box = QVBoxLayout(container)
    box.setContentsMargins(0, 0, 6, 0)
    box.addWidget(results)
    scroll = QScrollArea()
    scroll.setWidget(container)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    return scroll


def baseline_hint(results: list[RuleResult], baseline_missing: bool) -> str:
    needs = any(r.rule.enabled and r.rule.needs_baseline for r in results)
    return "⚠  尚未設定初始點，與初始點比較的規則無法判斷；請在「關鍵點數值」視窗設定" \
        if baseline_missing and needs else ""


class RecognitionPanel(QFrame):
    """右側辨識區：辨識中即時顯示各規則狀態，結束後顯示本次統計；下方為操作按鈕。"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        self.btn_rules = make_button("＋  添加辨識規則", "outline", height=30)
        self.btn_rules.setToolTip("開啟辨識規則設定頁面")
        title_row = QHBoxLayout()
        title_row.addWidget(make_label("姿態辨識", "CardTitle"))
        title_row.addStretch(1)
        title_row.addWidget(self.btn_rules)
        layout.addLayout(title_row)

        self.result_stack = QStackedWidget()
        self.result_stack.addWidget(make_placeholder(
            "◎", "辨識結果將顯示於此", "按下「開始辨識」後，會依辨識規則即時判斷姿勢"))
        result_page = QWidget()
        self.result_title = make_label("", "SectionTitle")
        self.result_hint = make_label("", "BaselineWarn")
        self.result_hint.setWordWrap(True)
        self.results = RuleResultList()
        page_box = QVBoxLayout(result_page)
        page_box.setContentsMargins(0, 0, 0, 0)
        page_box.setSpacing(8)
        page_box.addWidget(self.result_title)
        page_box.addWidget(self.result_hint)
        page_box.addWidget(make_results_view(self.results), 1)
        self.result_stack.addWidget(result_page)

        self.btn_start = make_button("▶  開始辨識", "primary", height=56)
        self.btn_start.setProperty("size", "large")
        self.btn_end = make_button("■  結束", "danger")
        self.btn_reset = make_button("↺  重置", "ghost")
        self.btn_reset.setToolTip("參數恢復預設值")

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.btn_end)
        buttons.addWidget(self.btn_reset)
        layout.addWidget(self.result_stack, 1)
        layout.addSpacing(4)
        layout.addWidget(self.btn_start)
        layout.addLayout(buttons)

    def show_idle(self) -> None:
        self.result_stack.setCurrentIndex(0)

    def show_live(self, results: list[RuleResult], tracker: RecognitionTracker,
                  baseline_missing: bool) -> None:
        self.result_stack.setCurrentIndex(1)
        self.result_title.setText(f"●  辨識中 {format_duration(tracker.elapsed_s)}")
        hint = baseline_hint(results, baseline_missing)
        self.result_hint.setText(hint)
        self.result_hint.setVisible(bool(hint))
        self.results.show_results(results, tracker)

    def show_summary(self, rules: list[Rule], tracker: RecognitionTracker) -> None:
        self.result_stack.setCurrentIndex(1)
        self.result_title.setText(f"本次辨識結果（{format_duration(tracker.elapsed_s)}）")
        self.result_hint.hide()
        self.results.show_summary(rules, tracker)


class RecognitionDialog(QDialog):
    """目前辨識結果：辨識期間即時顯示每條規則的狀態、觸發提醒與次數。"""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("目前辨識結果")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(420, 480)
        self.summary = make_label("", "Hint")
        self.summary.setWordWrap(True)
        self.baseline_warning = make_label("", "BaselineWarn")
        self.baseline_warning.setWordWrap(True)
        self.results = RuleResultList()
        self.content = QVBoxLayout(self)
        self.content.setContentsMargins(20, 18, 20, 20)
        self.content.setSpacing(8)
        self.content.addWidget(make_label("目前辨識結果", "CardTitle"))
        self.content.addWidget(self.summary)
        self.content.addWidget(self.baseline_warning)
        self.content.addWidget(make_results_view(self.results), 1)

    def update_result(self, results: list[RuleResult], tracker: RecognitionTracker,
                      baseline_missing: bool = False) -> None:
        active = [r for r in results if r.triggered]
        names = "、".join(r.rule.name for r in active)
        self.summary.setText(f"辨識中 {format_duration(tracker.elapsed_s)} · "
                             + (f"觸發中：{names}" if active else "目前姿勢正常"))
        hint = baseline_hint(results, baseline_missing)
        self.baseline_warning.setText(hint)
        self.baseline_warning.setVisible(bool(hint))
        self.results.show_results(results, tracker)


# (選單文字, 欄位標題, KeypointValue 欄位, 數值格式, 說明, 位移欄標題)；第一個為預設
COORD_MODES = (
    ("中心座標（像素）", ("X（px）", "Y（px）"), ("cx", "cy"), "{:+.2f}",
     "原點為畫面中心，x 向右、y 向上為正", "目前位移（px）"),
    ("影像座標（0~1）", ("x", "y", "z"), ("x", "y", "z"), "{:.3f}",
     "原點為畫面左上角，右下角為 (1, 1)", "目前位移"),
    ("世界座標（公尺）", ("X（公尺）", "Y（公尺）", "Z（公尺）"), ("wx", "wy", "wz"), "{:+.3f}",
     "原點為髖部中心，MediaPipe 估計值", "目前位移（公尺）"),
)
DELTA_COLUMN = 6
LOW_VISIBILITY = 0.5
SIDE_NOTE = f"左右以被拍攝者本人為準（鏡像顯示時與畫面左右相反）；灰色表示可見度低於 {LOW_VISIBILITY}"


def make_coord_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItems([mode[0] for mode in COORD_MODES])
    return combo


def format_clock(timestamp: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp))


class KeypointTable(QTableWidget):
    """33 個關鍵點的數值表格，「目前數值」與「初始點」兩個視窗共用。

    show_delta=True 時多一欄「目前位移」（目前 - 初始點），由 show_delta() 更新。
    """

    def __init__(self, show_delta: bool = False) -> None:
        super().__init__(len(LANDMARK_NAMES), 7 if show_delta else 6)
        self.has_delta = show_delta
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # 預先建立所有儲存格，更新時只改文字，避免每次重建
        for row, (name, name_zh) in enumerate(zip(LANDMARK_NAMES, LANDMARK_NAMES_ZH, strict=True)):
            for col, text in enumerate((str(row), name_zh, *["—"] * (self.columnCount() - 2))):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter if col != 1 else Qt.AlignLeft | Qt.AlignVCenter)
                if col == 1:
                    item.setToolTip(name)
                self.setItem(row, col, item)
        self.mode = COORD_MODES[0]
        self.set_mode(0)

    def set_mode(self, index: int) -> None:
        self.mode = COORD_MODES[index]
        headers = self.mode[1]
        labels = ["#", "部位", *headers, *[""] * (3 - len(headers)), "可見度"]
        self.setHorizontalHeaderLabels([*labels, self.mode[5]] if self.has_delta else labels)
        self.setColumnHidden(4, len(headers) < 3)  # 中心座標只有 x、y

    def clear_values(self) -> None:
        for row in range(self.rowCount()):
            for col in range(2, 6):
                self.item(row, col).setText("—")

    def show_snapshot(self, snap: KeypointSnapshot | None) -> None:
        if snap is None or not snap.detected:
            self.clear_values()
            return
        _, _, fields, fmt, _, _ = self.mode
        low, normal = QColor(theme.DISABLED), QColor(theme.TEXT)
        for p in snap.points:
            values = (getattr(p, f) for f in fields)
            texts = ["—" if v is None else fmt.format(v) for v in values]
            texts += [""] * (3 - len(texts))
            color = low if p.visibility < LOW_VISIBILITY else normal
            for col, text in enumerate([*texts, f"{p.visibility:.2f}"], start=2):
                item = self.item(p.index, col)
                item.setText(text)
                item.setForeground(color)


    def show_delta(self, current: KeypointSnapshot | None, baseline: KeypointSnapshot | None) -> None:
        """更新「目前位移」欄：各點目前相對初始點的位移，單位同目前座標。"""
        fields, fmt = self.mode[2], self.mode[3]
        low, normal = QColor(theme.DISABLED), QColor(theme.TEXT)
        for row, name in enumerate(LANDMARK_NAMES):
            delta = displacement(current, baseline, name, fields)
            item = self.item(row, DELTA_COLUMN)
            item.setText("—" if delta is None else ", ".join(fmt.format(v) for v in delta))
            now = current.get(name) if current is not None else None
            item.setForeground(low if now is None or now.visibility < LOW_VISIBILITY else normal)


class BaselineWindow(QDialog):
    """已儲存的初始點。設定初始點後自動彈出；初始點數值固定，「目前位移」欄定時更新。"""

    def __init__(self, parent: QWidget, getter: NowPointGetter, refresh_ms: int = 100) -> None:
        super().__init__(parent)
        self.getter = getter
        self.setWindowTitle("已儲存的初始點")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(600, 700)
        self.snapshot: KeypointSnapshot | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(refresh_ms)
        self.timer.timeout.connect(self.refresh_delta)

        self.info = make_label("", "Hint")
        self.info.setWordWrap(True)
        self.warning = make_label("", "BaselineWarn")
        self.warning.setWordWrap(True)
        self.warning.hide()
        self.coord = make_coord_combo()
        self.coord.currentIndexChanged.connect(self._change_mode)
        self.table = KeypointTable(show_delta=True)
        self.mode_hint = make_label("", "Hint")
        self.live = make_label("", "Hint")  # 目前畫面狀態
        self.live.setWordWrap(True)
        btn_close = make_button("關閉", "ghost", height=34)
        btn_close.clicked.connect(self.close)

        tools = QHBoxLayout()
        tools.addWidget(self.info, 1)
        tools.addWidget(self.coord)
        bottom = QHBoxLayout()
        note = make_label(SIDE_NOTE, "Hint")
        note.setWordWrap(True)
        bottom.addWidget(note, 1)
        bottom.addWidget(btn_close)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.addWidget(make_label("◎  已儲存的初始點", "CardTitle"))
        layout.addLayout(tools)
        layout.addWidget(self.warning)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.live)
        layout.addWidget(self.mode_hint)
        layout.addLayout(bottom)
        self._change_mode(0)

    def _change_mode(self, index: int) -> None:
        self.table.set_mode(index)
        self.mode_hint.setText(f"{COORD_MODES[index][0]}：{COORD_MODES[index][4]}；"
                               "目前位移 = 目前位置 - 初始點")
        self.table.show_snapshot(self.snapshot)
        self.refresh_delta()

    def refresh_delta(self) -> None:
        current = self.getter.latest
        self.table.show_delta(current, self.snapshot)
        if not current.detected:
            self.live.setText("目前未偵測到人，無法計算位移")
        elif self.snapshot is not None and current.image_size != self.snapshot.image_size:
            self.live.setText("⚠  目前畫面大小與設定初始點時不同，像素位移僅供參考，建議重新設定初始點")
        else:
            self.live.setText(f"位移即時更新中 · {format_clock(current.timestamp)}")

    def showEvent(self, event) -> None:
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.timer.stop()
        super().hideEvent(event)

    def show_baseline(self, snap: KeypointSnapshot, unclear: list) -> None:
        self.snapshot = snap
        size = f" · 畫面 {snap.image_size[0]}×{snap.image_size[1]}" if snap.image_size else ""
        self.info.setText(f"設定時間 {format_clock(snap.timestamp)}{size}")
        names = "、".join(LANDMARK_NAMES_ZH[p.index] for p in unclear if p is not None)
        self.warning.setText(f"⚠  {names}可見度偏低，數值可能不準確，建議調整位置後重新設定")
        self.warning.setVisible(bool(names))
        self.table.show_snapshot(snap)
        self.refresh_delta()
        self.show()
        self.raise_()


class KeypointWindow(QDialog):
    """目前關鍵點數值。進入第 2 頁時自動開啟，定時從 NowPointGetter 讀取最新數值。

    下方可把目前姿勢「設為初始點」（正常坐姿），設定後以 BaselineWindow 顯示儲存的數值。
    """

    baseline_changed = pyqtSignal(object)  # 設定初始點後發出 KeypointSnapshot，給辨識模組使用

    def __init__(self, parent: QWidget, getter: NowPointGetter, refresh_ms: int = 100) -> None:
        super().__init__(parent)
        self.getter = getter
        self.setWindowTitle("目前關鍵點數值")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(480, 760)
        self.baseline_window = BaselineWindow(parent, getter)
        self._baseline_placed = False

        self.status = make_label("未偵測到人", "Hint")
        self.mode_hint = make_label("", "Hint")
        for label in (self.status, self.mode_hint):
            label.setWordWrap(True)  # 文字變長時換行，不把視窗撐寬
        self.coord = make_coord_combo()
        self.coord.currentIndexChanged.connect(self._change_mode)
        self.btn_pause = make_button("❚❚  暫停", "ghost", height=32)
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(
            lambda paused: self.btn_pause.setText("▶  繼續" if paused else "❚❚  暫停"))
        self.table = KeypointTable()

        tools = QHBoxLayout()
        tools.addWidget(self.status, 1)
        tools.addWidget(self.coord)
        tools.addWidget(self.btn_pause)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.addWidget(make_label("目前關鍵點數值", "CardTitle"))
        layout.addLayout(tools)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.mode_hint)
        note = make_label(SIDE_NOTE, "Hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addSpacing(6)
        layout.addWidget(self._build_baseline_box())

        self.timer = QTimer(self)
        self.timer.setInterval(refresh_ms)  # 表格不需跟影像一樣每幀更新
        self.timer.timeout.connect(self.refresh)
        self._change_mode(0)

    def _build_baseline_box(self) -> QFrame:
        box = QFrame()
        box.setObjectName("BaselineBox")
        tip = make_label("坐正之後，點擊下方按鈕定位正常點", "BaselineTip")
        detail = make_label("背部挺直、雙肩放鬆、眼睛平視螢幕，並確認上半身完整入鏡", "Hint")
        detail.setWordWrap(True)
        self.btn_baseline = make_button("◎  設為初始點", "primary", height=44)
        self.btn_baseline.clicked.connect(self.set_baseline)
        self.baseline_pill = Pill()
        self.baseline_pill.setWordWrap(True)  # 警告訊息較長時換行
        self.baseline_pill.set_state("尚未設定初始點", "idle")
        self.btn_view_baseline = make_button("▤  查看初始點", "ghost", height=30)
        self.btn_view_baseline.setEnabled(False)
        self.btn_view_baseline.clicked.connect(self.show_baseline_window)
        status_row = QHBoxLayout()
        status_row.addWidget(self.baseline_pill, 1)
        status_row.addWidget(self.btn_view_baseline, 0, Qt.AlignVCenter)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(tip)
        layout.addWidget(detail)
        layout.addWidget(self.btn_baseline)
        layout.addLayout(status_row)
        return box

    def set_baseline(self) -> None:
        snap = self.getter.set_baseline()
        if snap is None:
            self.baseline_pill.set_state("⚠  未偵測到人，請坐到鏡頭前再設定", "warning")
            return
        clock = format_clock(snap.timestamp)
        unclear = self.getter.low_visibility_points(snap)
        if unclear:
            names = "、".join(LANDMARK_NAMES_ZH[p.index] for p in unclear if p is not None)
            self.baseline_pill.set_state(f"⚠  已設定（{clock}），但{names}不清楚", "warning")
        else:
            self.baseline_pill.set_state(f"✓  已設定初始點（{clock}）", "saved")
        self.btn_baseline.setText("◎  重新設定初始點")
        self.btn_view_baseline.setEnabled(True)
        self.show_baseline_window()
        self.baseline_changed.emit(snap)

    def show_baseline_window(self) -> None:
        """彈出「已儲存的初始點」視窗；第一次放在本視窗左側，之後保留使用者拖曳的位置。"""
        snap = self.getter.baseline
        if snap is None:
            return
        win = self.baseline_window
        win.show_baseline(snap, self.getter.low_visibility_points(snap))  # 先顯示才能取得實際大小
        if not self._baseline_placed:
            self._baseline_placed = True
            screen = QApplication.desktop().availableGeometry(self)
            here = self.frameGeometry()
            win.move(max(here.left() - win.frameGeometry().width() - 12, screen.left()), here.top())

    def _change_mode(self, index: int) -> None:
        self.table.set_mode(index)
        self.mode_hint.setText(f"{COORD_MODES[index][0]}：{COORD_MODES[index][4]}")
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        if self.btn_pause.isChecked() and not force:
            return
        snap = self.getter.latest
        if snap.detected:
            size = f" · 畫面 {snap.image_size[0]}×{snap.image_size[1]}" if snap.image_size else ""
            self.status.setText(f"偵測到 {snap.num_poses} 人 · 第 {snap.pose_index + 1} 人{size}"
                                f" · {format_clock(snap.timestamp)}")
        else:
            self.status.setText("未偵測到人")
        self.table.show_snapshot(snap)

    def showEvent(self, event) -> None:
        if self.getter.baseline is None:  # 離開第 2 頁時初始點已清除
            self.baseline_pill.set_state("尚未設定初始點", "idle")
            self.btn_baseline.setText("◎  設為初始點")
            self.btn_view_baseline.setEnabled(False)
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.timer.stop()  # 關閉時不佔用資源
        super().hideEvent(event)


# ---------- 第 2 頁 ----------


class DetectionPage(QWidget):
    back_requested = pyqtSignal()
    rules_requested = pyqtSignal()  # 按下「添加辨識規則」

    def __init__(self, notifier: ErrorNotifier, store: PointDataStore,
                 rule_engine: RuleEngine | None = None) -> None:
        super().__init__()
        self.setObjectName("DetectionPage")
        self.setAttribute(Qt.WA_StyledBackground)
        self.rule_engine = rule_engine if rule_engine is not None else RuleEngine(path=None)
        self.tracker = RecognitionTracker()
        self._last_results: list[RuleResult] = []
        self._last_result_ui = 0.0
        self._notifier = notifier
        self.store = store
        self.page: SkeletonDetection | None = None
        self.params: ParamPanel | None = None
        self._fps = 0.0
        self._last_tick = 0.0
        self._ending = False

        # 標題列
        header, bar = make_header_bar()
        self.camera_chip = make_label("", "Chip")
        self.rec_pill = Pill()
        self.btn_back = make_button("←  更換相機", "ghost", height=34)
        self.btn_back.clicked.connect(self.back_requested)
        self.btn_points = make_button("⌗  關鍵點數值", "ghost", height=34)
        self.btn_points.setToolTip("開啟「目前關鍵點數值」視窗")
        self.btn_points.clicked.connect(self.show_keypoints)
        bar.addWidget(self.camera_chip, 0, Qt.AlignVCenter)
        bar.addStretch(1)
        bar.addWidget(self.rec_pill, 0, Qt.AlignVCenter)
        bar.addWidget(self.btn_points, 0, Qt.AlignVCenter)
        bar.addWidget(self.btn_back, 0, Qt.AlignVCenter)

        # 左：即時影像
        video_card, video_layout = make_card()
        self.chip_model = make_label("", "Chip")
        self.chip_infer = make_label("", "Chip")
        self.chip_fps = make_label("", "Chip")
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title_row.addWidget(make_label("即時影像", "CardTitle"))
        title_row.addStretch(1)
        for chip in (self.chip_model, self.chip_infer, self.chip_fps):
            title_row.addWidget(chip)
        self.view = VideoView()
        video_layout.addLayout(title_row)
        video_layout.addWidget(self.view, 1)

        # 中：參數（進入本頁時才依相機建立 ParamPanel）
        params_card, self.params_box = make_card("參數設定")
        params_card.setFixedWidth(360)

        # 右：辨識
        self.recognition = RecognitionPanel()
        self.recognition.setFixedWidth(280)
        self.recognition.btn_start.clicked.connect(self.start_recognition)
        self.recognition.btn_end.clicked.connect(self.end_recognition)
        self.recognition.btn_reset.clicked.connect(self.reset)
        self.recognition.btn_rules.clicked.connect(self.rules_requested)
        self.dialog = RecognitionDialog(self)
        self.dialog.finished.connect(self.end_recognition)  # 關閉結果視窗也視為結束
        self.point_getter = NowPointGetter()
        self.keypoint_window = KeypointWindow(self, self.point_getter)
        self._keypoint_placed = False

        body = QHBoxLayout()
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(16)
        body.addWidget(video_card, 1)
        body.addWidget(params_card)
        body.addWidget(self.recognition)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addLayout(body, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

    # ---------- 進入 / 離開 ----------

    def enter(self, camera: CameraSetting) -> None:
        self.page = SkeletonDetection(camera)
        self.params = ParamPanel(self.page, self._notifier)
        self.params_box.addWidget(self.params, 1)
        self.camera_chip.setText(f"●  {camera.camera_name}")
        self.rec_pill.set_state("待機中", "idle")
        self._set_recording(False)
        self._notifier.check(self.page)
        self._last_tick = time.perf_counter()
        self.timer.start(0)  # cap.read 會等待下一幀，因此實際頻率等於相機 FPS
        self.show_keypoints()

    def leave(self) -> None:
        self.timer.stop()
        self.end_recognition()
        self.recognition.show_idle()
        self.keypoint_window.close()
        self.keypoint_window.baseline_window.close()
        self.point_getter.reset()
        if self.params is not None:
            self.params.setParent(None)
            self.params.deleteLater()
            self.params = None
        if self.page is not None:
            self.page.close()  # 不會關閉相機
            self.page = None

    # ---------- 每幀 ----------

    def tick(self) -> None:
        if self.page is None:
            return
        f = self.page.process_frame()
        self.point_getter.update(f)
        now = time.perf_counter()
        dt, self._last_tick = now - self._last_tick, now
        if f.ok and dt > 0:
            self._fps = 1 / dt if self._fps == 0 else self._fps * 0.9 + 0.1 / dt
        if f.frame is not None:
            self.view.show_frame(f.frame)
        self.chip_model.setText(self.page.params.spec("model").display_value)
        self.chip_infer.setText(f"推論 {f.inference_ms:.0f} ms")
        self.chip_fps.setText(f"{self._fps:.0f} FPS")
        if self.store.recording:
            self.store.record(f)
            elapsed = int(self.store.elapsed_s)
            dot = "●" if elapsed % 2 == 0 else "○"  # 每秒閃爍
            self.rec_pill.set_state(
                f"{dot}  記錄中 {elapsed // 60:02d}:{elapsed % 60:02d} · {self.store.frame_count} 幀",
                "recording")
            self.update_recognition()
        self._notifier.check(self.page, self.store)

    def update_recognition(self, force: bool = False) -> list[RuleResult]:
        """依辨識規則判斷最新關鍵點。每幀都判斷（持續時間才準確），畫面每 0.2 秒更新一次。"""
        latest, baseline = self.point_getter.latest, self.point_getter.baseline
        results = self.rule_engine.evaluate(latest, baseline)
        self.tracker.update(results)
        self._last_results = results
        now = time.perf_counter()
        if force or now - self._last_result_ui >= 0.2:
            self._last_result_ui = now
            self.recognition.show_live(results, self.tracker, baseline is None)
            self.dialog.update_result(results, self.tracker, baseline is None)
        return results

    # ---------- 按鈕 ----------

    def show_keypoints(self) -> None:
        """開啟「目前關鍵點數值」視窗。只有第一次自動擺放，之後保留使用者拖曳後的位置。"""
        win = self.keypoint_window
        win.show()  # 先顯示才能取得實際大小
        win.raise_()
        if not self._keypoint_placed:
            self._keypoint_placed = True
            # 盡量放在主視窗右側；螢幕不夠寬時貼齊螢幕右緣，只遮住右側的辨識區
            main = self.window().frameGeometry()
            screen = QApplication.desktop().availableGeometry(self.window())
            x = min(main.right() + 12, screen.right() - win.frameGeometry().width())
            win.move(max(x, screen.left()), main.top() + 90)

    def start_recognition(self) -> None:
        if self.page is None or self.store.recording:
            return
        session_id = self.store.start_session(self.page.camera.camera_name, self.page.params.values())
        self._notifier.check(self.store)
        if session_id is None:
            return
        self.rec_pill.set_state("●  記錄中 00:00 · 0 幀", "recording")
        self._set_recording(True)
        self.rule_engine.reset_timers()  # 持續時間從這次辨識開始計算
        self.tracker.start()
        self.update_recognition(force=True)
        self.dialog.show()

    def end_recognition(self) -> None:
        if self._ending:
            return
        self._ending = True
        try:
            if self.store.recording:
                self.store.end_session()
                self.rec_pill.set_state(f"✓  已儲存 {self.store.frame_count} 幀", "saved")
                self.recognition.show_summary(self.rule_engine.rules, self.tracker)
            if self.dialog.isVisible():
                self.dialog.close()
            self._set_recording(False)
            self._notifier.check(self.store)
        finally:
            self._ending = False

    def reset(self) -> None:
        if self.page is None:
            return
        self.page.reset()
        self.params.refresh()
        self._notifier.check(self.page)

    def _set_recording(self, recording: bool) -> None:
        self.recognition.btn_start.setEnabled(not recording)
        self.recognition.btn_end.setEnabled(recording)
        self.recognition.btn_reset.setEnabled(not recording)
        self.btn_back.setEnabled(not recording)
        self.recognition.btn_rules.setEnabled(not recording)  # 記錄期間不能修改規則
        if self.params is not None:
            self.params.setEnabled(not recording)  # 記錄期間鎖定參數


# ---------- 辨識規則頁 ----------

# RuleStatus -> (Pill 狀態, 列表符號)
STATUS_STYLE = {
    RuleStatus.NORMAL: ("saved", "✓"),
    RuleStatus.PENDING: ("warning", "…"),
    RuleStatus.TRIGGERED: ("recording", "⚠"),
}


def status_style(status: RuleStatus | None) -> tuple[str, str]:
    return STATUS_STYLE.get(status, ("idle", "○"))


def set_state(widget: QWidget, state: str) -> None:
    """設定 QSS 用的 state 屬性並重新套用樣式。"""
    if widget.property("state") != state:
        widget.setProperty("state", state)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def make_range_spin() -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-100000, 100000)
    spin.setDecimals(2)
    spin.setKeyboardTracking(False)  # 輸入完成才套用
    return spin


class ConditionEditor(QFrame):
    """一個條件的編輯元件。依指標顯示需要的欄位，變更直接寫回 Condition 並發出 changed。"""

    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, condition: Condition, number: int) -> None:
        super().__init__()
        self.setObjectName("ConditionBox")
        self.condition = condition
        self._references: tuple[Reference, ...] = ()

        self.btn_remove = make_button("刪除條件", "ghost", height=28)
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))
        self.metric = QComboBox()
        self.metric.addItems([m.value for m in Metric])
        self.metric_help = make_label("", "Hint")
        self.metric_help.setWordWrap(True)
        self.point_labels = [make_label("", "FieldLabel") for _ in range(3)]
        self.point_combos = []
        for _ in range(3):
            combo = QComboBox()
            combo.addItems([label for _, label in POINT_CHOICES])
            combo.setMaxVisibleItems(15)
            self.point_combos.append(combo)
        self.axis = QComboBox()
        self.axis.addItems([a.value for a in Axis])
        self.reference = QComboBox()
        self.mode = QComboBox()
        self.mode.addItems([m.value for m in RangeMode])
        self.low, self.high = make_range_spin(), make_range_spin()
        self.low_none, self.high_none = QCheckBox("不限"), QCheckBox("不限")
        self.unit_low, self.unit_high = make_label("", "FieldLabel"), make_label("", "FieldLabel")
        self.summary = make_label("", "Hint")
        self.summary.setWordWrap(True)
        self.live = make_label("目前數值：—", "LiveValue")
        self.live.setWordWrap(True)

        head = QHBoxLayout()
        head.addWidget(make_label(f"條件 {number}", "ConditionTitle"))
        head.addStretch(1)
        head.addWidget(self.btn_remove)
        # 每個欄位是獨立的一列，隱藏時不留空白（QFormLayout 隱藏列仍會保留間距）
        points = QWidget()
        points_box = QHBoxLayout(points)
        points_box.setContentsMargins(0, 0, 0, 0)
        for label, combo in zip(self.point_labels, self.point_combos, strict=True):
            points_box.addWidget(label)
            points_box.addWidget(combo, 1)
        self.points_row = self._row("關鍵點", points)
        self.axis_row = self._row("方向", self.axis)
        self.reference_row = self._row("比較基準", self.reference)
        self.mode_row = self._row("成立條件", self.mode)
        self.low_row = self._row("下限", self._bound(self.low, self.unit_low, self.low_none))
        self.high_row = self._row("上限", self._bound(self.high, self.unit_high, self.high_none))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)
        layout.addLayout(head)
        layout.addWidget(self._row("指標", self.metric))
        layout.addWidget(self.metric_help)
        for row in (self.points_row, self.axis_row, self.reference_row, self.mode_row,
                    self.low_row, self.high_row):
            layout.addWidget(row)
        layout.addWidget(self.summary)
        layout.addWidget(self.live)

        self.metric.currentIndexChanged.connect(self._on_metric)
        for widget in (*self.point_combos, self.axis, self.reference, self.mode):
            widget.currentIndexChanged.connect(self._on_field)
        for spin in (self.low, self.high):
            spin.valueChanged.connect(self._on_field)
        for check in (self.low_none, self.high_none):
            check.toggled.connect(self._on_field)
        self._load()

    @staticmethod
    def _row(label: str, field: QWidget) -> QWidget:
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(12)
        title = make_label(label, "FieldLabel")
        title.setFixedWidth(64)
        box.addWidget(title)
        box.addWidget(field, 1)
        return row

    @staticmethod
    def _bound(spin: QDoubleSpinBox, unit: QLabel, none: QCheckBox) -> QWidget:
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(spin, 1)
        box.addWidget(unit)
        box.addWidget(none)
        return row

    def _widgets(self) -> list[QWidget]:
        return [self.metric, *self.point_combos, self.axis, self.reference, self.mode,
                self.low, self.high, self.low_none, self.high_none]

    def _load(self) -> None:
        """把 Condition 的內容填入元件（不觸發變更事件）。"""
        c = self.condition
        for w in self._widgets():
            w.blockSignals(True)
        self.metric.setCurrentIndex(list(Metric).index(c.metric))
        names = [name for name, _ in POINT_CHOICES]
        for combo, point in zip(self.point_combos, c.points, strict=False):
            combo.setCurrentIndex(names.index(point) if point in names else 0)
        self.axis.setCurrentIndex(list(Axis).index(c.axis))
        self._references = METRIC_REFERENCES[c.metric]
        self.reference.clear()
        self.reference.addItems([r.value for r in self._references])
        self.reference.setCurrentIndex(self._references.index(c.reference)
                                       if c.reference in self._references else 0)
        self.mode.setCurrentIndex(list(RangeMode).index(c.mode))
        for spin, none, value in ((self.low, self.low_none, c.low), (self.high, self.high_none, c.high)):
            none.setChecked(value is None)
            spin.setValue(value if value is not None else 0.0)
        for w in self._widgets():
            w.blockSignals(False)
        self._sync()

    def _sync(self) -> None:
        """依指標顯示 / 隱藏欄位，並更新單位與文字說明。"""
        c = self.condition
        labels = METRIC_POINT_LABELS[c.metric]
        for i, (label, combo) in enumerate(zip(self.point_labels, self.point_combos, strict=True)):
            combo.setVisible(i < len(labels))
            label.setVisible(len(labels) > 1 and i < len(labels))  # 只有一個點時不需小標籤
            if i < len(labels):
                label.setText(labels[i])
        self.points_row.setVisible(bool(labels))
        self.axis_row.setVisible(c.metric is Metric.POSITION)
        self.reference_row.setVisible(len(self._references) > 1)
        has_range = c.metric is not Metric.NO_PERSON
        for row in (self.mode_row, self.low_row, self.high_row):
            row.setVisible(has_range)
        self.low.setEnabled(not self.low_none.isChecked())
        self.high.setEnabled(not self.high_none.isChecked())
        self.unit_low.setText(c.unit)
        self.unit_high.setText(c.unit)
        self.metric_help.setText(METRIC_HELP[c.metric])
        warnings = c.warnings()
        self.summary.setText("⚠  " + "；".join(warnings) if warnings else f"成立條件：{c.describe()}")

    def _on_metric(self, index: int) -> None:
        self.condition.set_metric(list(Metric)[index])
        self._load()
        self.changed.emit()

    def _on_field(self) -> None:
        c = self.condition
        n = len(METRIC_POINT_LABELS[c.metric])
        c.points = [POINT_CHOICES[combo.currentIndex()][0] for combo in self.point_combos[:n]]
        c.axis = list(Axis)[self.axis.currentIndex()]
        if self._references:
            c.reference = self._references[max(self.reference.currentIndex(), 0)]
        c.mode = list(RangeMode)[self.mode.currentIndex()]
        c.low = None if self.low_none.isChecked() else round(self.low.value(), 2)
        c.high = None if self.high_none.isChecked() else round(self.high.value(), 2)
        self._sync()
        self.changed.emit()

    def set_live(self, result: ConditionResult | None) -> None:
        """顯示目前計算出的數值，協助使用者設定門檻。"""
        if result is None:
            text, state = "目前數值：—", "unknown"
        elif self.condition.metric is Metric.NO_PERSON:
            text, state = ("目前：畫面中沒有人（成立）", "met") if result.met else ("目前：畫面中有人", "clear")
        elif result.value is None:
            text, state = f"目前數值：—（{result.reason}）", "unknown"
        else:
            verdict = "成立" if result.met else "不成立"
            text = f"目前數值：{result.value:+.2f} {self.condition.unit}　{verdict}"
            state = "met" if result.met else "clear"
        self.live.setText(text)
        set_state(self.live, state)


class RulesPage(QWidget):
    """辨識規則設定（由第 2 頁「添加辨識規則」進入）。

    左：規則列表（新增 / 範本 / 刪除）；中：編輯名稱、判斷方式、持續時間、提醒訊息與條件；
    右：即時預覽，顯示每個條件目前的數值與規則狀態，方便設定門檻。變更會自動儲存。
    離開本頁回到第 2 頁時，相機與骨架偵測持續運作，不需重新載入。
    """

    back_requested = pyqtSignal()

    def __init__(self, engine: RuleEngine, getter: NowPointGetter, notifier: ErrorNotifier,
                 refresh_ms: int = 200) -> None:
        super().__init__()
        self.setObjectName("DetectionPage")  # 與第 2 頁相同的背景
        self.setAttribute(Qt.WA_StyledBackground)
        self.engine = engine
        self.getter = getter
        self._notifier = notifier
        self.condition_editors: list[ConditionEditor] = []

        header, bar = make_header_bar()
        bar.addWidget(make_label("辨識規則設定", "Chip"), 0, Qt.AlignVCenter)
        bar.addStretch(1)
        bar.addWidget(make_label("變更會自動儲存", "Hint"), 0, Qt.AlignVCenter)
        self.btn_back = make_button("←  返回辨識畫面", "ghost", height=34)
        self.btn_back.clicked.connect(self.back_requested)
        bar.addWidget(self.btn_back, 0, Qt.AlignVCenter)

        body = QHBoxLayout()
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(16)
        body.addWidget(self._build_list_card())
        body.addWidget(self._build_editor_card(), 1)
        body.addWidget(self._build_preview_card())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addLayout(body, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(refresh_ms)
        self.timer.timeout.connect(self.refresh_preview)
        self._reload_list()

    # ---------- 版面 ----------

    def _build_list_card(self) -> QFrame:
        card, layout = make_card("規則列表")
        card.setFixedWidth(280)
        self.rule_list = QListWidget()
        self.rule_list.currentRowChanged.connect(self._select)
        self.btn_add = make_button("＋  新增規則", "outline", height=36)
        self.btn_add.clicked.connect(self.add_rule)
        self.btn_template = make_button("☆  從範本新增", height=36)
        self.template_menu = QMenu(self)
        for preset in preset_rules():
            action = self.template_menu.addAction(preset.name)
            action.setToolTip("；".join(c.describe() for c in preset.conditions))
            action.triggered.connect(lambda _=False, name=preset.name: self.add_preset(name))
        self.template_menu.setToolTipsVisible(True)
        self.btn_template.setMenu(self.template_menu)
        self.btn_delete = make_button("刪除規則", "danger", height=36)
        self.btn_delete.clicked.connect(self.delete_rule)
        layout.addWidget(self.rule_list, 1)
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_template)
        layout.addWidget(self.btn_delete)
        return card

    def _build_editor_card(self) -> QFrame:
        card, layout = make_card("規則設定")
        self.editor_stack = QStackedWidget()
        self.editor_stack.addWidget(make_placeholder(
            "☰", "請從左側選擇或新增規則", "也可以「從範本新增」常見的坐姿規則，再調整門檻"))

        editor = QWidget()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：低頭、駝背")
        self.name_edit.textEdited.connect(self._on_name)
        self.enabled_check = QCheckBox("啟用此規則")
        self.enabled_check.toggled.connect(self._on_rule_field)
        self.match_combo = QComboBox()
        self.match_combo.addItems(["全部條件都成立", "任一條件成立"])
        self.match_combo.currentIndexChanged.connect(self._on_rule_field)
        self.hold_spin = QDoubleSpinBox()
        self.hold_spin.setRange(0, 600)
        self.hold_spin.setSingleStep(0.5)
        self.hold_spin.setDecimals(1)
        self.hold_spin.setSuffix(" 秒")
        self.hold_spin.setKeyboardTracking(False)
        self.hold_spin.setToolTip("條件要持續成立多久才算觸發，避免短暫動作造成誤判")
        self.hold_spin.valueChanged.connect(self._on_rule_field)
        self.message_edit = QLineEdit()
        self.message_edit.setPlaceholderText("觸發時顯示的提醒，例如：請抬頭挺胸")
        self.message_edit.textEdited.connect(self._on_rule_field)
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        for label, widget in (("名稱", self.name_edit), ("狀態", self.enabled_check),
                              ("判斷方式", self.match_combo), ("持續時間", self.hold_spin),
                              ("提醒訊息", self.message_edit)):
            form.addRow(make_label(label, "FieldLabel"), widget)

        self.btn_add_condition = make_button("＋  新增條件", "outline", height=30)
        self.btn_add_condition.clicked.connect(self.add_condition)
        cond_head = QHBoxLayout()
        cond_head.addWidget(make_label("條件", "SectionTitle"))
        cond_head.addWidget(make_label("可以選擇關鍵點，設定位移、距離或角度在什麼區間時成立", "Hint"), 1)
        cond_head.addWidget(self.btn_add_condition)
        container = QWidget()
        container.setObjectName("ScrollContent")
        self.conditions_box = QVBoxLayout(container)
        self.conditions_box.setContentsMargins(0, 0, 6, 0)
        self.conditions_box.setSpacing(10)
        self.conditions_box.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(10)
        editor_layout.addLayout(form)
        editor_layout.addSpacing(4)
        editor_layout.addLayout(cond_head)
        editor_layout.addWidget(scroll, 1)
        self.editor_stack.addWidget(editor)
        layout.addWidget(self.editor_stack, 1)
        return card

    def _build_preview_card(self) -> QFrame:
        card, layout = make_card("即時預覽")
        card.setFixedWidth(320)
        self.preview_name = make_label("", "SectionTitle")
        self.preview_pill = Pill()
        self.preview_pill.setWordWrap(True)
        self.preview_detail = make_label("", "Hint")
        self.preview_detail.setWordWrap(True)
        self.baseline_label = make_label("", "Hint")
        self.baseline_label.setWordWrap(True)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {theme.BORDER};")
        self.all_rules_label = make_label("", "Hint")
        self.all_rules_label.setWordWrap(True)
        self.all_rules_label.setTextFormat(Qt.RichText)
        layout.addWidget(self.preview_name)
        layout.addWidget(self.preview_pill, 0, Qt.AlignLeft)
        layout.addWidget(self.preview_detail)
        layout.addWidget(self.baseline_label)
        layout.addWidget(line)
        layout.addWidget(make_label("所有規則", "SectionTitle"))
        layout.addWidget(self.all_rules_label)
        layout.addStretch(1)
        return card

    # ---------- 規則列表 ----------

    @property
    def current_rule(self) -> Rule | None:
        row = self.rule_list.currentRow()
        return self.engine.rules[row] if 0 <= row < len(self.engine.rules) else None

    def _reload_list(self, select_id: str | None = None) -> None:
        self.rule_list.blockSignals(True)
        self.rule_list.clear()
        for rule in self.engine.rules:
            self.rule_list.addItem(QListWidgetItem(self._item_text(rule, None)))
        self.rule_list.blockSignals(False)
        ids = [r.id for r in self.engine.rules]
        row = ids.index(select_id) if select_id in ids else (0 if ids else -1)
        self.rule_list.setCurrentRow(row)
        self._select(row)

    @staticmethod
    def _item_text(rule: Rule, status: RuleStatus | None) -> str:
        suffix = "" if rule.enabled else "（停用）"
        return f"{status_style(status)[1]}  {rule.name or '（未命名）'}{suffix}"

    def add_rule(self) -> None:
        rule = self.engine.add(Rule(f"新規則 {len(self.engine.rules) + 1}", [Condition()]))
        self._save()
        self._reload_list(rule.id)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def add_preset(self, name: str) -> None:
        preset = next(r for r in preset_rules() if r.name == name)
        existing = {r.name for r in self.engine.rules}
        number = 2
        while preset.name in existing:  # 重複加入同一範本時自動編號，方便區分
            preset.name = f"{name} ({number})"
            number += 1
        self.engine.add(preset)
        self._save()
        self._reload_list(preset.id)

    def delete_rule(self) -> None:
        rule = self.current_rule
        if rule is None:
            return
        row = self.rule_list.currentRow()
        self.engine.remove(rule.id)
        self._save()
        rules = self.engine.rules
        self._reload_list(rules[min(row, len(rules) - 1)].id if rules else None)

    # ---------- 編輯 ----------

    def _select(self, row: int) -> None:
        rule = self.current_rule
        self.btn_delete.setEnabled(rule is not None)
        self.editor_stack.setCurrentIndex(0 if rule is None else 1)
        if rule is None:
            self._build_conditions(None)
            self.refresh_preview()
            return
        fields = (self.name_edit, self.enabled_check, self.match_combo, self.hold_spin, self.message_edit)
        for w in fields:
            w.blockSignals(True)
        self.name_edit.setText(rule.name)
        self.enabled_check.setChecked(rule.enabled)
        self.match_combo.setCurrentIndex(0 if rule.match == "all" else 1)
        self.hold_spin.setValue(rule.hold_s)
        self.message_edit.setText(rule.message)
        for w in fields:
            w.blockSignals(False)
        self._build_conditions(rule)
        self.refresh_preview()

    def _build_conditions(self, rule: Rule | None) -> None:
        for editor in self.condition_editors:
            editor.setParent(None)
            editor.deleteLater()
        self.condition_editors = []
        if rule is None:
            return
        for number, condition in enumerate(rule.conditions, start=1):
            editor = ConditionEditor(condition, number)
            editor.changed.connect(self._save)
            editor.remove_requested.connect(self._remove_condition)
            self.conditions_box.insertWidget(self.conditions_box.count() - 1, editor)
            self.condition_editors.append(editor)

    def add_condition(self) -> None:
        rule = self.current_rule
        if rule is None:
            return
        rule.conditions.append(Condition())
        self._build_conditions(rule)
        self._save()

    def _remove_condition(self, editor: ConditionEditor) -> None:
        rule = self.current_rule
        if rule is None:
            return
        rule.conditions = [c for c in rule.conditions if c is not editor.condition]
        self._build_conditions(rule)
        self._save()

    def _on_name(self, text: str) -> None:
        rule = self.current_rule
        if rule is None:
            return
        rule.name = text.strip()
        self._save()

    def _on_rule_field(self) -> None:
        rule = self.current_rule
        if rule is None:
            return
        rule.enabled = self.enabled_check.isChecked()
        rule.match = "all" if self.match_combo.currentIndex() == 0 else "any"
        rule.hold_s = round(self.hold_spin.value(), 1)
        rule.message = self.message_edit.text().strip()
        self._save()

    def _save(self) -> None:
        self.engine.save()
        self._notifier.check(self.engine)
        self.refresh_preview()

    # ---------- 即時預覽 ----------

    def refresh_preview(self) -> None:
        results = self.engine.evaluate(self.getter.latest, self.getter.baseline)
        lines = []
        for row, result in enumerate(results):
            item = self.rule_list.item(row)
            if item is not None:
                item.setText(self._item_text(result.rule, result.status))
                item.setForeground(QColor(theme.DANGER if result.triggered else theme.TEXT))
            color = {RuleStatus.TRIGGERED: theme.DANGER, RuleStatus.PENDING: theme.WARNING,
                     RuleStatus.NORMAL: theme.SUCCESS}.get(result.status, theme.MUTED)
            lines.append(f'<span style="color:{color}">{status_style(result.status)[1]}</span>&nbsp; '
                         f"{result.rule.name or '（未命名）'}：{result.status.value}")
        self.all_rules_label.setText("<br>".join(lines) or "尚未建立任何規則")
        base = self.getter.baseline
        self.baseline_label.setText(
            f"初始點：已設定（{format_clock(base.timestamp)}）" if base is not None
            else "初始點：尚未設定。與初始點比較的條件需要先在「目前關鍵點數值」視窗設定初始點")

        rule = self.current_rule
        result = next((r for r in results if rule is not None and r.rule is rule), None)
        if result is None:
            self.preview_name.setText("")
            self.preview_pill.set_state("未選擇規則", "idle")
            self.preview_detail.setText("")
            return
        self.preview_name.setText(rule.name or "（未命名）")
        state, symbol = status_style(result.status)
        text = f"{symbol}  {result.status.value}"
        if result.status is RuleStatus.PENDING:
            text += f" {result.held_s:.1f} / {rule.hold_s:g} 秒"
        self.preview_pill.set_state(text, state)
        detail = {
            RuleStatus.TRIGGERED: rule.message or "條件已持續成立",
            RuleStatus.NEED_BASELINE: "此規則有與初始點比較的條件，請先設定初始點",
            RuleStatus.UNKNOWN: "部分關鍵點看不清楚或畫面中沒有人，暫時無法判斷",
            RuleStatus.NO_CONDITION: "請新增至少一個條件",
            RuleStatus.DISABLED: "此規則已停用，不會參與辨識",
        }.get(result.status, "")
        self.preview_detail.setText(detail)
        for editor, cond_result in zip(self.condition_editors, result.conditions, strict=False):
            editor.set_live(cond_result)
        if not result.conditions:
            for editor in self.condition_editors:
                editor.set_live(None)

    def showEvent(self, event) -> None:
        self.refresh_preview()
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.timer.stop()
        super().hideEvent(event)


# ---------- 主視窗 ----------


class MainWindow(QMainWindow):
    def __init__(
        self,
        camera: CameraSetting | None = None,
        store_factory: Callable[[], PointDataStore] = SqliteStore,
        rules_path: Path | None = RULES_PATH,
    ) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 860)
        self.notifier = ErrorNotifier(self)
        self.camera = camera if camera is not None else CameraSetting()
        self.store = store_factory()

        self.cover = CoverPage(self.camera, self.notifier)
        self.rule_engine = RuleEngine(rules_path)
        self.detection = DetectionPage(self.notifier, self.store, self.rule_engine)
        self.rules = RulesPage(self.rule_engine, self.detection.point_getter, self.notifier)
        self.stack = QStackedWidget()
        for page in (self.cover, self.detection, self.rules):
            self.stack.addWidget(page)
        self.setCentralWidget(self.stack)

        self.cover.camera_ready.connect(self.show_detection)
        self.detection.back_requested.connect(self.show_cover)
        self.detection.rules_requested.connect(lambda: self.stack.setCurrentWidget(self.rules))
        self.rules.back_requested.connect(lambda: self.stack.setCurrentWidget(self.detection))
        QTimer.singleShot(0, self.cover.scan)  # 視窗顯示後再掃描，避免啟動時卡住
        QTimer.singleShot(0, lambda: self.notifier.check(self.rule_engine))  # 規則檔讀取錯誤

    def show_detection(self, camera: CameraSetting) -> None:
        # 探測解析度與載入模型需要數秒，先給使用者回饋
        self.cover.btn_ok.setText("載入中…")
        self.cover.btn_ok.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self.detection.enter(camera)
            self.stack.setCurrentWidget(self.detection)
        finally:
            QApplication.restoreOverrideCursor()
            self.cover.btn_ok.setText(CONFIRM_TEXT)
            self.cover.btn_ok.setEnabled(True)

    def show_cover(self) -> None:
        self.detection.leave()
        self.stack.setCurrentWidget(self.cover)
        self.cover.scan()

    def closeEvent(self, event) -> None:
        self.detection.leave()
        self.store.close()
        self.camera.close()
        super().closeEvent(event)


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    theme.apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
