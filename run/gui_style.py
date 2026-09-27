"""操作介面的主題：顏色、字型與 Qt 樣式表（QSS）。換主題只需修改本檔上方的顏色。

元件用 objectName 或 variant 屬性套用樣式，例如：
    button.setProperty("variant", "primary")   # primary / outline / danger / ghost
    label.setObjectName("Chip")
"""
from __future__ import annotations

from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication

FONT_FAMILY = "Microsoft JhengHei UI"

BG = "#0f1419"  # 視窗底色
SURFACE = "#171d25"  # 卡片
SURFACE_2 = "#1f2630"  # 輸入框、次要按鈕
BORDER = "#2a3340"
TEXT = "#e6edf3"
MUTED = "#8b98a5"
DISABLED = "#56616d"
ACCENT = "#2dd4bf"  # 主色（青綠）
ACCENT_2 = "#38bdf8"  # 漸層第二色（天藍）
ON_ACCENT = "#06201c"  # 主色上的文字
DANGER = "#f87171"
SUCCESS = "#4ade80"
WARNING = "#fbbf24"

STYLE = f"""
QWidget {{ color: {TEXT}; font-family: "{FONT_FAMILY}"; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; }}

/* ---------- 頁面與卡片 ---------- */
#Cover {{
    background: qradialgradient(cx:0.5, cy:0.3, radius:0.85, fx:0.5, fy:0.25,
                                stop:0 #123a40, stop:0.5 #0f1c24, stop:1 {BG});
}}
#DetectionPage {{ background: {BG}; }}
#Header {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
#CoverCard {{ background: rgba(23, 29, 37, 235); border: 1px solid {BORDER}; border-radius: 22px; }}

/* ---------- 文字 ---------- */
#AppTitle {{ font-size: 30pt; font-weight: 700; letter-spacing: 6px; }}
#Subtitle {{ color: {MUTED}; font-size: 11pt; letter-spacing: 1px; }}
#HeaderTitle {{ font-size: 14pt; font-weight: 700; letter-spacing: 2px; }}
#CardTitle {{ font-size: 12pt; font-weight: 700; }}
#FieldLabel, #Hint {{ color: {MUTED}; }}
#Hint {{ font-size: 9pt; }}
#Footer {{ color: {DISABLED}; font-size: 9pt; }}

/* ---------- 按鈕 ---------- */
QPushButton {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 16px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {DISABLED}; background: #151a21; border-color: #222a34; }}
QPushButton[variant="primary"] {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT}, stop:1 {ACCENT_2});
    color: {ON_ACCENT}; border: none; font-weight: 700;
}}
QPushButton[variant="primary"]:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5eead4, stop:1 #7dd3fc);
}}
QPushButton[size="large"] {{ font-size: 14pt; letter-spacing: 2px; }}
QPushButton[variant="primary"]:pressed {{ background: {ACCENT}; }}
QPushButton[variant="primary"]:disabled {{ background: #1c3533; color: #4e716c; }}
QPushButton[variant="danger"] {{ background: transparent; color: {DANGER}; border: 1px solid {DANGER}; }}
QPushButton[variant="danger"]:hover {{ background: rgba(248, 113, 113, 30); }}
QPushButton[variant="danger"]:disabled {{ color: #5b3d3d; border-color: #3a2a2a; background: transparent; }}
QPushButton[variant="outline"] {{
    background: transparent; color: {ACCENT}; border: 1px solid {ACCENT}; padding: 4px 12px;
}}
QPushButton[variant="outline"]:hover {{ background: rgba(45, 212, 191, 30); }}
QPushButton[variant="outline"]:disabled {{ color: {DISABLED}; border-color: {BORDER}; background: transparent; }}
QPushButton[variant="ghost"] {{ background: transparent; border: 1px solid transparent; color: {MUTED}; }}
QPushButton[variant="ghost"]:hover {{ color: {TEXT}; border-color: {BORDER}; }}
QPushButton[variant="ghost"]:disabled {{ color: {DISABLED}; background: transparent; border-color: transparent; }}

/* ---------- 輸入元件 ---------- */
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 4px 8px; min-height: 20px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover,
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {DISABLED}; background: #151a21; }}
QComboBox QAbstractItemView {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; outline: none;
    selection-background-color: {ACCENT}; selection-color: {ON_ACCENT};
}}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 10px; margin-top: 12px; padding: 12px 10px 6px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {ACCENT}; font-weight: 700;
}}
QGroupBox:disabled::title {{ color: {DISABLED}; }}
QPlainTextEdit {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px; }}

/* ---------- 捲軸 ---------- */
QScrollArea, #ScrollContent {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* ---------- 影像、標籤、狀態 ---------- */
#VideoView {{ background: #07090c; border-radius: 10px; color: {MUTED}; }}
#Chip {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 11px;
    padding: 3px 10px; color: {MUTED}; font-size: 9pt;
}}
#Pill {{ border-radius: 13px; padding: 4px 14px; font-weight: 700; }}
#Pill[state="idle"] {{ background: {SURFACE_2}; color: {MUTED}; border: 1px solid {BORDER}; }}
#Pill[state="recording"] {{ background: rgba(248, 113, 113, 38); color: {DANGER}; border: 1px solid {DANGER}; }}
#Pill[state="saved"] {{ background: rgba(74, 222, 128, 30); color: {SUCCESS}; border: 1px solid {SUCCESS}; }}
#Pill[state="warning"] {{ background: rgba(251, 191, 36, 30); color: {WARNING}; border: 1px solid {WARNING}; }}
#BaselineBox {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
#BaselineWarn {{ color: {WARNING}; }}
#BaselineTip {{ color: {ACCENT}; font-size: 11pt; font-weight: 700; }}
#Placeholder {{ border: 2px dashed {BORDER}; border-radius: 12px; background: transparent; }}
#PlaceholderIcon {{ color: #3b4756; font-size: 64pt; }}
QTableWidget {{
    background: {SURFACE}; alternate-background-color: {SURFACE_2}; border: 1px solid {BORDER};
    border-radius: 8px; gridline-color: {BORDER}; selection-background-color: rgba(45, 212, 191, 50);
    selection-color: {TEXT};
}}
QHeaderView::section {{
    background: {SURFACE_2}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 6px 4px; font-weight: 700;
}}
QLineEdit {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px;
    selection-background-color: {ACCENT}; selection-color: {ON_ACCENT};
}}
QLineEdit:hover, QLineEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled {{ color: {DISABLED}; background: #151a21; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {BORDER}; border-radius: 4px; background: {SURFACE_2};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox:disabled {{ color: {DISABLED}; }}
QListWidget {{
    background: transparent; border: none; outline: none;
}}
QListWidget::item {{ padding: 10px 10px; border-radius: 8px; margin: 2px 0; }}
QListWidget::item:hover {{ background: {SURFACE_2}; }}
QListWidget::item:selected {{ background: rgba(45, 212, 191, 40); color: {TEXT}; }}
QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER}; padding: 6px; }}
QMenu::item {{ padding: 8px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: rgba(45, 212, 191, 50); }}
#ConditionBox {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 10px; }}
#ConditionTitle {{ color: {ACCENT}; font-weight: 700; }}
#LiveValue {{ color: {TEXT}; font-size: 11pt; font-weight: 700; }}
#LiveValue[state="met"] {{ color: {WARNING}; }}
#LiveValue[state="clear"] {{ color: {SUCCESS}; }}
#LiveValue[state="unknown"] {{ color: {MUTED}; font-weight: 400; }}
#SectionTitle {{ font-size: 11pt; font-weight: 700; }}
#ResultRow {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px; }}
#ResultRow[state="recording"] {{ background: rgba(248, 113, 113, 30); border-color: {DANGER}; }}
#ResultRow[state="warning"] {{ border-color: {WARNING}; }}
#ResultRow[state="saved"] {{ border-color: #2f5b45; }}
#ResultRow[state="recording"] #ResultName {{ color: {DANGER}; }}
#ResultName {{ font-weight: 700; }}
#ErrorTitle {{ color: {WARNING}; font-size: 13pt; font-weight: 700; }}
"""


def apply_theme(app: QApplication) -> None:
    """Fusion 風格 + 暗色調色盤 + 樣式表。調色盤讓下拉箭頭、數字框箭頭等原生繪製的部分也是亮色。"""
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in [
        (QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, SURFACE_2),
        (QPalette.AlternateBase, SURFACE), (QPalette.Text, TEXT), (QPalette.Button, SURFACE_2),
        (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, ON_ACCENT),
        (QPalette.ToolTipBase, SURFACE_2), (QPalette.ToolTipText, TEXT),
    ]:
        palette.setColor(role, QColor(color))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        palette.setColor(QPalette.Disabled, role, QColor(DISABLED))
    app.setPalette(palette)
    app.setFont(QFont(FONT_FAMILY, 10))
    app.setStyleSheet(STYLE)
