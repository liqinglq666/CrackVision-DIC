from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.config import load_config
from src.core.input import FILE_DIALOG_FILTER
from src.gui.worker import AnalysisWorker


class MainWindow(QMainWindow):
    """Focused scientific workflow for peak-stress DIC crack-width analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CrackVision-DIC")
        self.resize(1180, 760)
        self.setMinimumSize(980, 680)
        self.setAcceptDrops(True)

        self.data_file: Path | None = None
        self.mts_file: Path | None = None
        self.output_path: Path | None = None
        self.worker: AnalysisWorker | None = None
        self.config = load_config()

        self._build_ui()
        self._refresh_ready_state()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(34, 28, 34, 28)
        layout.setSpacing(20)

        # Header
        header = QHBoxLayout()
        header.setSpacing(18)

        brand = QVBoxLayout()
        brand.setSpacing(5)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title = QLabel("CrackVision-DIC")
        title.setObjectName("appTitle")
        title_row.addWidget(title)

        scope_badge = QLabel("ECC · SHCC")
        scope_badge.setObjectName("scopeBadge")
        scope_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row.addWidget(scope_badge)
        title_row.addStretch(1)

        subtitle = QLabel("峰值拉应力状态下的 DIC 裂缝定位、COD 计算与科研结果导出")
        subtitle.setObjectName("subtitle")

        brand.addLayout(title_row)
        brand.addWidget(subtitle)
        header.addLayout(brand, 1)

        mode = QFrame()
        mode.setObjectName("modeCard")
        mode_layout = QVBoxLayout(mode)
        mode_layout.setContentsMargins(14, 9, 14, 9)
        mode_layout.setSpacing(2)
        mode_label = QLabel("分析模式")
        mode_label.setObjectName("microLabel")
        mode_value = QLabel("Peak-frame")
        mode_value.setObjectName("modeValue")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(mode_value)
        header.addWidget(mode)

        layout.addLayout(header)

        # Workflow strip
        workflow = QFrame()
        workflow.setObjectName("workflowStrip")
        workflow_layout = QHBoxLayout(workflow)
        workflow_layout.setContentsMargins(12, 10, 12, 10)
        workflow_layout.setSpacing(8)
        workflow_layout.addWidget(
            self._make_step("01", "导入 DIC", "H5 / MAT 位移与应变场")
        )
        workflow_layout.addWidget(self._workflow_arrow())
        workflow_layout.addWidget(
            self._make_step("02", "匹配 MTS", "CSV 峰值拉力与时间")
        )
        workflow_layout.addWidget(self._workflow_arrow())
        workflow_layout.addWidget(
            self._make_step("03", "裂缝 COD", "峰值帧定位、统计与 Excel")
        )
        layout.addWidget(workflow)

        # Main content
        content = QHBoxLayout()
        content.setSpacing(18)

        input_card = QFrame()
        input_card.setObjectName("panelCard")
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(22, 20, 22, 20)
        input_layout.setSpacing(16)

        input_header = QHBoxLayout()
        input_title_block = QVBoxLayout()
        input_title_block.setSpacing(3)
        input_title = QLabel("数据输入")
        input_title.setObjectName("panelTitle")
        input_help = QLabel("选择同一试件的 DIC 与 MTS 数据；也可以直接拖入窗口。")
        input_help.setObjectName("panelHelp")
        input_title_block.addWidget(input_title)
        input_title_block.addWidget(input_help)
        input_header.addLayout(input_title_block, 1)

        input_tip = QLabel("2 files required")
        input_tip.setObjectName("miniBadge")
        input_header.addWidget(input_tip, 0, Qt.AlignmentFlag.AlignTop)
        input_layout.addLayout(input_header)

        (
            data_picker,
            self.data_edit,
            self.data_status,
            self.data_button,
        ) = self._make_file_picker(
            number="01",
            title="Ncorr / CrackVision-DIC 数据",
            description="推荐 CrackVision-Ncorr H5，也支持原始 MAT。",
            placeholder="尚未选择 DIC 数据",
            button_text="选择 DIC",
            handler=self._choose_data,
        )
        input_layout.addWidget(data_picker)

        (
            mts_picker,
            self.mts_edit,
            self.mts_status,
            self.mts_button,
        ) = self._make_file_picker(
            number="02",
            title="MTS / DAQ 原始数据",
            description="选择与 DIC 同步开始的 CSV；时间零点固定为 0 s。",
            placeholder="尚未选择 MTS / DAQ CSV",
            button_text="选择 CSV",
            handler=self._choose_mts,
        )
        input_layout.addWidget(mts_picker)

        assumption = QFrame()
        assumption.setObjectName("assumptionBox")
        assumption_layout = QHBoxLayout(assumption)
        assumption_layout.setContentsMargins(13, 11, 13, 11)
        assumption_layout.setSpacing(10)

        assumption_icon = QLabel("i")
        assumption_icon.setObjectName("infoIcon")
        assumption_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        assumption_icon.setFixedSize(22, 22)

        assumption_text = QLabel(
            "输出自动写入 Ncorr 文件旁的 CrackVision_Output 文件夹；"
            "程序仅分析 MTS 峰值拉应力对应的最近 DIC 帧。"
        )
        assumption_text.setObjectName("assumptionText")
        assumption_text.setWordWrap(True)

        assumption_layout.addWidget(assumption_icon, 0, Qt.AlignmentFlag.AlignTop)
        assumption_layout.addWidget(assumption_text, 1)
        input_layout.addWidget(assumption)
        input_layout.addStretch(1)

        self.start_button = QPushButton("载入数据后开始分析")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setMinimumHeight(48)
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        input_layout.addWidget(self.start_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("analysisProgress")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.hide()
        input_layout.addWidget(self.progress_bar)

        content.addWidget(input_card, 3)

        # Result / status panel
        result_card = QFrame()
        result_card.setObjectName("panelCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(22, 20, 22, 20)
        result_layout.setSpacing(14)

        result_header = QHBoxLayout()
        result_heading_block = QVBoxLayout()
        result_heading_block.setSpacing(3)
        result_title = QLabel("分析状态")
        result_title.setObjectName("panelTitle")
        result_help = QLabel("完成后在这里查看关键统计量。")
        result_help.setObjectName("panelHelp")
        result_heading_block.addWidget(result_title)
        result_heading_block.addWidget(result_help)
        result_header.addLayout(result_heading_block, 1)

        self.status_pill = QLabel("等待输入")
        self.status_pill.setObjectName("statusPill")
        self.status_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_header.addWidget(self.status_pill, 0, Qt.AlignmentFlag.AlignTop)
        result_layout.addLayout(result_header)

        self.status_label = QLabel("请选择 Ncorr 数据和对应的 MTS CSV。")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        result_layout.addWidget(self.status_label)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)

        peak_card, self.peak_value = self._make_metric("峰值拉力", "—", "N")
        frame_card, self.frame_value = self._make_metric("DIC 帧", "—", "")
        crack_card, self.crack_value = self._make_metric("有效裂缝", "—", "条")
        mean_card, self.mean_value = self._make_metric("平均宽度", "—", "μm")
        p95_card, self.p95_value = self._make_metric("P95 宽度", "—", "μm")
        max_card, self.max_value = self._make_metric("最大宽度", "—", "μm")

        metrics.addWidget(peak_card, 0, 0)
        metrics.addWidget(frame_card, 0, 1)
        metrics.addWidget(crack_card, 1, 0)
        metrics.addWidget(mean_card, 1, 1)
        metrics.addWidget(p95_card, 2, 0)
        metrics.addWidget(max_card, 2, 1)
        result_layout.addLayout(metrics)

        self.result_label = QLabel("")
        self.result_label.setObjectName("resultText")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.result_label.hide()
        result_layout.addWidget(self.result_label)

        result_layout.addStretch(1)

        self.open_button = QPushButton("打开结果文件夹")
        self.open_button.setObjectName("secondaryButton")
        self.open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_button.clicked.connect(self._open_output_folder)
        self.open_button.hide()
        result_layout.addWidget(self.open_button)

        content.addWidget(result_card, 2)
        layout.addLayout(content, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer_note = QLabel(
            "Scientific output · Peak-force synchronization · Direction-aware COD"
        )
        footer_note.setObjectName("footerNote")
        footer.addWidget(footer_note)
        footer.addStretch(1)
        footer_state = QLabel("CrackVision-DIC")
        footer_state.setObjectName("footerNote")
        footer.addWidget(footer_state)
        layout.addLayout(footer)

        self.setStyleSheet(
            """
            QWidget#appRoot {
                background: #f4f7fb;
                color: #0f172a;
                font-family: "Segoe UI", "Noto Sans CJK SC", sans-serif;
                font-size: 13px;
            }
            QLabel#appTitle {
                color: #0f172a;
                font-size: 28px;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #64748b;
                font-size: 13px;
            }
            QLabel#scopeBadge, QLabel#miniBadge {
                color: #1d4ed8;
                background: #eff6ff;
                border: 1px solid #dbeafe;
                border-radius: 9px;
                padding: 4px 9px;
                font-size: 11px;
                font-weight: 600;
            }
            QFrame#modeCard {
                background: #0f172a;
                border-radius: 10px;
            }
            QLabel#microLabel {
                color: #94a3b8;
                font-size: 10px;
            }
            QLabel#modeValue {
                color: #ffffff;
                font-size: 13px;
                font-weight: 700;
            }
            QFrame#workflowStrip {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
            QFrame#stepCard {
                background: transparent;
                border: none;
            }
            QLabel#stepNumber {
                color: #2563eb;
                background: #eff6ff;
                border-radius: 12px;
                padding: 4px 8px;
                min-width: 22px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#stepTitle {
                color: #0f172a;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#stepCaption {
                color: #94a3b8;
                font-size: 10px;
            }
            QLabel#workflowArrow {
                color: #cbd5e1;
                font-size: 17px;
                padding: 0 4px;
            }
            QFrame#panelCard {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
            }
            QLabel#panelTitle {
                color: #0f172a;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#panelHelp {
                color: #64748b;
                font-size: 11px;
            }
            QFrame#filePicker {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 11px;
            }
            QLabel#pickerNumber {
                color: #2563eb;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#pickerTitle {
                color: #1e293b;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#pickerDescription {
                color: #64748b;
                font-size: 10px;
            }
            QLabel#fileState {
                color: #64748b;
                background: #eef2f7;
                border-radius: 8px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 600;
            }
            QLabel#fileState[ready="true"] {
                color: #047857;
                background: #ecfdf5;
            }
            QLineEdit {
                background: #ffffff;
                color: #334155;
                border: 1px solid #dbe3ee;
                border-radius: 8px;
                padding: 9px 10px;
                selection-background-color: #dbeafe;
            }
            QLineEdit:focus {
                border: 1px solid #93c5fd;
            }
            QPushButton {
                min-height: 34px;
                padding: 0 14px;
                color: #334155;
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f8fafc;
                border-color: #cbd5e1;
            }
            QPushButton:pressed {
                background: #f1f5f9;
            }
            QPushButton#primaryButton {
                min-height: 48px;
                color: #ffffff;
                background: #2563eb;
                border: 1px solid #2563eb;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background: #1d4ed8;
                border-color: #1d4ed8;
            }
            QPushButton#primaryButton:disabled {
                color: #e2e8f0;
                background: #94a3b8;
                border-color: #94a3b8;
            }
            QPushButton#secondaryButton {
                min-height: 40px;
                background: #0f172a;
                color: #ffffff;
                border: none;
            }
            QPushButton#secondaryButton:hover {
                background: #1e293b;
            }
            QFrame#assumptionBox {
                background: #f8fbff;
                border: 1px solid #dbeafe;
                border-radius: 9px;
            }
            QLabel#infoIcon {
                color: #1d4ed8;
                background: #dbeafe;
                border-radius: 11px;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#assumptionText {
                color: #475569;
                font-size: 10px;
            }
            QLabel#statusPill {
                color: #475569;
                background: #f1f5f9;
                border-radius: 9px;
                padding: 4px 9px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#status {
                color: #334155;
                font-size: 12px;
                font-weight: 600;
                padding: 2px 0 4px 0;
            }
            QFrame#metricCard {
                background: #f8fafc;
                border: 1px solid #edf1f6;
                border-radius: 10px;
            }
            QLabel#metricLabel {
                color: #64748b;
                font-size: 10px;
            }
            QLabel#metricValue {
                color: #0f172a;
                font-size: 19px;
                font-weight: 700;
            }
            QLabel#metricUnit {
                color: #94a3b8;
                font-size: 10px;
            }
            QLabel#resultText {
                color: #64748b;
                background: #f8fafc;
                border-radius: 8px;
                padding: 10px;
                font-size: 10px;
            }
            QProgressBar#analysisProgress {
                border: none;
                border-radius: 2px;
                background: #e2e8f0;
            }
            QProgressBar#analysisProgress::chunk {
                border-radius: 2px;
                background: #2563eb;
            }
            QLabel#footerNote {
                color: #94a3b8;
                font-size: 9px;
            }
            """
        )

    def _make_step(self, number: str, title: str, caption: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("stepCard")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        row = QHBoxLayout(frame)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(9)

        number_label = QLabel(number)
        number_label.setObjectName("stepNumber")
        number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text = QVBoxLayout()
        text.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("stepTitle")
        caption_label = QLabel(caption)
        caption_label.setObjectName("stepCaption")
        text.addWidget(title_label)
        text.addWidget(caption_label)

        row.addWidget(number_label)
        row.addLayout(text, 1)
        return frame

    @staticmethod
    def _workflow_arrow() -> QLabel:
        arrow = QLabel("→")
        arrow.setObjectName("workflowArrow")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return arrow

    def _make_file_picker(
        self,
        *,
        number: str,
        title: str,
        description: str,
        placeholder: str,
        button_text: str,
        handler,
    ) -> tuple[QFrame, QLineEdit, QLabel, QPushButton]:
        frame = QFrame()
        frame.setObjectName("filePicker")
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 12, 14, 13)
        box.setSpacing(9)

        top = QHBoxLayout()
        top.setSpacing(8)

        number_label = QLabel(number)
        number_label.setObjectName("pickerNumber")
        top.addWidget(number_label)

        title_block = QVBoxLayout()
        title_block.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("pickerTitle")
        description_label = QLabel(description)
        description_label.setObjectName("pickerDescription")
        description_label.setWordWrap(True)
        title_block.addWidget(title_label)
        title_block.addWidget(description_label)
        top.addLayout(title_block, 1)

        status = QLabel("未选择")
        status.setObjectName("fileState")
        status.setProperty("ready", False)
        top.addWidget(status, 0, Qt.AlignmentFlag.AlignTop)
        box.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(8)
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)

        button = QPushButton(button_text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(handler)

        row.addWidget(edit, 1)
        row.addWidget(button)
        box.addLayout(row)
        return frame, edit, status, button

    def _make_metric(
        self, label: str, value: str, unit: str
    ) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setObjectName("metricCard")
        box = QVBoxLayout(frame)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(3)

        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")

        value_row = QHBoxLayout()
        value_row.setSpacing(5)
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        unit_label = QLabel(unit)
        unit_label.setObjectName("metricUnit")
        value_row.addWidget(value_label)
        value_row.addWidget(unit_label, 0, Qt.AlignmentFlag.AlignBottom)
        value_row.addStretch(1)

        box.addWidget(label_widget)
        box.addLayout(value_row)
        return frame, value_label

    @staticmethod
    def _set_file_status(label: QLabel, ready: bool) -> None:
        label.setText("已载入" if ready else "未选择")
        label.setProperty("ready", ready)
        style = label.style()
        style.unpolish(label)
        style.polish(label)

    def _set_status(self, text: str, tone: str) -> None:
        palette = {
            "idle": ("等待输入", "#475569", "#f1f5f9"),
            "ready": ("可以分析", "#1d4ed8", "#eff6ff"),
            "running": ("分析中", "#b45309", "#fffbeb"),
            "success": ("已完成", "#047857", "#ecfdf5"),
            "error": ("失败", "#b91c1c", "#fef2f2"),
        }
        pill_text, fg, bg = palette.get(tone, palette["idle"])
        self.status_pill.setText(pill_text)
        self.status_pill.setStyleSheet(
            f"color:{fg};background:{bg};border-radius:9px;"
            "padding:4px 9px;font-size:10px;font-weight:700;"
        )
        self.status_label.setText(text)

    def _reset_metrics(self) -> None:
        for label in (
            self.peak_value,
            self.frame_value,
            self.crack_value,
            self.mean_value,
            self.p95_value,
            self.max_value,
        ):
            label.setText("—")

    def _apply_data_file(self, path: Path) -> None:
        self.data_file = path
        self.data_edit.setText(path.name)
        self.data_edit.setToolTip(str(path))
        self.output_path = None

    def _apply_mts_file(self, path: Path) -> None:
        self.mts_file = path
        self.mts_edit.setText(path.name)
        self.mts_edit.setToolTip(str(path))
        self.output_path = None

    def _choose_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Ncorr 数据",
            "",
            FILE_DIALOG_FILTER,
        )
        if not file_path:
            return
        self._apply_data_file(Path(file_path))
        self._refresh_ready_state()

    def _choose_mts(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 MTS / DAQ CSV",
            "",
            "CSV files (*.csv)",
        )
        if not file_path:
            return
        self._apply_mts_file(Path(file_path))
        self._refresh_ready_state()

    def _refresh_ready_state(self) -> None:
        data_ready = self.data_file is not None
        mts_ready = self.mts_file is not None
        ready = data_ready and mts_ready and self.worker is None

        self._set_file_status(self.data_status, data_ready)
        self._set_file_status(self.mts_status, mts_ready)

        self.start_button.setEnabled(ready)
        self.open_button.hide()
        self.result_label.hide()

        if self.worker is None:
            self._reset_metrics()

        if ready:
            self.start_button.setText("开始峰值帧裂缝分析")
            self._set_status(
                "输入已就绪。点击开始后将自动匹配峰值拉应力最近 DIC 帧并计算全部有效裂缝宽度。",
                "ready",
            )
        elif data_ready or mts_ready:
            self.start_button.setText("还需要 1 个输入文件")
            missing = "MTS / DAQ CSV" if data_ready else "Ncorr / DIC 数据"
            self._set_status(f"已载入 1 个文件，还需要选择 {missing}。", "idle")
        else:
            self.start_button.setText("载入数据后开始分析")
            self._set_status("请选择 Ncorr 数据和对应的 MTS CSV。", "idle")

    def _start(self) -> None:
        if self.data_file is None or self.mts_file is None:
            return

        out_dir = self.data_file.parent / "CrackVision_Output"
        self.worker = AnalysisWorker(
            self.data_file,
            self.mts_file,
            out_dir,
            self.config,
        )
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.start_button.setEnabled(False)
        self.start_button.setText("正在分析峰值帧…")
        self.data_button.setEnabled(False)
        self.mts_button.setEnabled(False)

        self.open_button.hide()
        self.result_label.hide()
        self._reset_metrics()

        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self._set_status(
            "正在读取 MTS 峰值、匹配 DIC 帧并执行裂缝定位与 COD 计算，请稍候。",
            "running",
        )
        self.worker.start()

    @staticmethod
    def _fmt(value: object, digits: int = 1) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def _on_completed(self, result: dict) -> None:
        self.output_path = Path(result["output_path"])
        status = result.get("cod_status", "unknown")
        crack_count = int(result.get("crack_count", 0))

        self.peak_value.setText(self._fmt(result.get("peak_force_N")))
        self.frame_value.setText(str(result.get("selected_frame", "—")))
        self.crack_value.setText(str(crack_count))
        self.mean_value.setText(self._fmt(result.get("mean_width_um")))
        self.p95_value.setText(self._fmt(result.get("p95_width_um")))
        self.max_value.setText(self._fmt(result.get("max_width_um")))

        if status == "ok":
            self._set_status(
                "分析完成。峰值拉应力对应帧的裂缝宽度统计已生成，并已导出 Excel。",
                "success",
            )
        else:
            self._set_status(f"分析完成，但 COD 状态为：{status}", "error")

        self.result_label.setText(
            f"峰值拉力 {self._fmt(result.get('peak_force_N'))} N @ "
            f"{self._fmt(result.get('peak_time_s'), 3)} s  ·  "
            f"DIC Frame {result.get('selected_frame', '—')}  ·  "
            f"Δt {self._fmt(result.get('match_error_s'), 3)} s\n"
            f"有效裂缝 {crack_count} 条  ·  "
            f"平均 {self._fmt(result.get('mean_width_um'))} μm  ·  "
            f"P95 {self._fmt(result.get('p95_width_um'))} μm  ·  "
            f"最大 {self._fmt(result.get('max_width_um'))} μm\n"
            f"Excel：{self.output_path}"
        )
        self.result_label.show()
        self.open_button.show()

    def _on_failed(self, message: str) -> None:
        self._set_status("分析失败，请检查输入数据格式或查看错误信息。", "error")
        QMessageBox.critical(self, "分析失败", message)

    def _on_finished(self) -> None:
        self.worker = None
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)
        self.data_button.setEnabled(True)
        self.mts_button.setEnabled(True)
        self.start_button.setText("重新分析")
        self.start_button.setEnabled(
            self.data_file is not None and self.mts_file is not None
        )

    def _open_output_folder(self) -> None:
        if self.output_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path.parent)))

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None:
            event.ignore()
            return
        urls = event.mimeData().urls()
        if any(
            Path(url.toLocalFile()).suffix.lower() in {".h5", ".hdf5", ".mat", ".csv"}
            for url in urls
            if url.isLocalFile()
        ):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None:
            event.ignore()
            return
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        data_candidates = [
            path for path in paths if path.suffix.lower() in {".h5", ".hdf5", ".mat"}
        ]
        mts_candidates = [path for path in paths if path.suffix.lower() == ".csv"]

        if data_candidates:
            self._apply_data_file(data_candidates[0])
        if mts_candidates:
            self._apply_mts_file(mts_candidates[0])

        if data_candidates or mts_candidates:
            self._refresh_ready_state()
            event.acceptProposedAction()
        else:
            event.ignore()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "正在分析",
                "当前分析尚未完成，请等待结果生成后再关闭。",
            )
            event.ignore()
            return
        event.accept()
