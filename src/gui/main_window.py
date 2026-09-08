from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.config import load_config
from src.core.input import FILE_DIALOG_FILTER
from src.gui.worker import AnalysisWorker


class MainWindow(QMainWindow):
    """Minimal peak-stress crack-width workflow for Ncorr ECC/SHCC data."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CrackVision-DIC")
        self.resize(820, 560)
        self.setMinimumSize(760, 520)

        self.data_file: Path | None = None
        self.mts_file: Path | None = None
        self.output_path: Path | None = None
        self.worker: AnalysisWorker | None = None
        self.config = load_config()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(36, 30, 36, 30)
        layout.setSpacing(18)

        title = QLabel("CrackVision-DIC")
        title.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        subtitle = QLabel("峰值拉应力状态 · ECC / SHCC 裂缝宽度")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        workflow = QLabel("Ncorr 位移/应变场  +  MTS 原始 CSV  →  自动定位峰值帧  →  裂缝 COD  →  Excel")
        workflow.setObjectName("workflow")
        workflow.setWordWrap(True)
        layout.addWidget(workflow)

        input_card = QFrame()
        input_card.setObjectName("card")
        card_layout = QVBoxLayout(input_card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(14)

        card_layout.addWidget(self._section_label("1  Ncorr 数据"))
        data_row = QHBoxLayout()
        self.data_edit = QLineEdit()
        self.data_edit.setReadOnly(True)
        self.data_edit.setPlaceholderText("选择 CrackVision-Ncorr H5（推荐）或原始 MAT")
        choose_data = QPushButton("选择文件")
        choose_data.clicked.connect(self._choose_data)
        data_row.addWidget(self.data_edit, 1)
        data_row.addWidget(choose_data)
        card_layout.addLayout(data_row)

        card_layout.addWidget(self._section_label("2  MTS 原始数据"))
        mts_row = QHBoxLayout()
        self.mts_edit = QLineEdit()
        self.mts_edit.setReadOnly(True)
        self.mts_edit.setPlaceholderText("选择同一试件的 MTS / DAQ CSV")
        choose_mts = QPushButton("选择文件")
        choose_mts.clicked.connect(self._choose_mts)
        mts_row.addWidget(self.mts_edit, 1)
        mts_row.addWidget(choose_mts)
        card_layout.addLayout(mts_row)

        hint = QLabel("MTS 与 DIC 同时开始，时间零点固定为 0 s；结果自动保存到 Ncorr 文件旁的 CrackVision_Output 文件夹。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        card_layout.addWidget(hint)
        layout.addWidget(input_card)

        self.start_button = QPushButton("分析峰值拉应力帧")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setMinimumHeight(46)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        layout.addWidget(self.start_button)

        self.result_card = QFrame()
        self.result_card.setObjectName("resultCard")
        result_layout = QVBoxLayout(self.result_card)
        result_layout.setContentsMargins(18, 16, 18, 16)
        result_layout.setSpacing(8)

        self.status_label = QLabel("请选择 Ncorr 数据和对应的 MTS CSV。")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        result_layout.addWidget(self.status_label)

        self.result_label = QLabel("")
        self.result_label.setObjectName("resultText")
        self.result_label.setWordWrap(True)
        self.result_label.hide()
        result_layout.addWidget(self.result_label)

        self.open_button = QPushButton("打开结果文件夹")
        self.open_button.setObjectName("secondaryButton")
        self.open_button.clicked.connect(self._open_output_folder)
        self.open_button.hide()
        result_layout.addWidget(self.open_button, 0)
        layout.addWidget(self.result_card)
        layout.addStretch(1)

        self.setStyleSheet(
            "QMainWindow{background:#f7f8fa;}"
            "QLabel#subtitle{color:#344054;font-size:14px;}"
            "QLabel#workflow{color:#667085;font-size:12px;padding:4px 0 2px 0;}"
            "QFrame#card,QFrame#resultCard{background:#ffffff;border:1px solid #e4e7ec;border-radius:10px;}"
            "QLabel#sectionLabel{font-size:12px;font-weight:600;color:#344054;}"
            "QLabel#hint{color:#667085;font-size:11px;}"
            "QLabel#status{font-size:13px;font-weight:600;color:#344054;}"
            "QLabel#resultText{font-size:12px;color:#475467;}"
            "QLineEdit{background:#ffffff;border:1px solid #d0d5dd;border-radius:7px;padding:9px 10px;}"
            "QPushButton{border:1px solid #d0d5dd;border-radius:7px;padding:8px 14px;background:#ffffff;}"
            "QPushButton:hover{background:#f2f4f7;}"
            "QPushButton#primaryButton{background:#1f4e78;color:#ffffff;border:none;font-weight:600;font-size:13px;}"
            "QPushButton#primaryButton:hover{background:#173b5b;}"
            "QPushButton#primaryButton:disabled{background:#98a2b3;color:#ffffff;}"
            "QPushButton#secondaryButton{max-width:140px;}"
        )

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _choose_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Ncorr 数据",
            "",
            FILE_DIALOG_FILTER,
        )
        if not file_path:
            return
        self.data_file = Path(file_path)
        self.data_edit.setText(self.data_file.name)
        self.data_edit.setToolTip(str(self.data_file))
        self.output_path = None
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
        self.mts_file = Path(file_path)
        self.mts_edit.setText(self.mts_file.name)
        self.mts_edit.setToolTip(str(self.mts_file))
        self.output_path = None
        self._refresh_ready_state()

    def _refresh_ready_state(self) -> None:
        ready = self.data_file is not None and self.mts_file is not None and self.worker is None
        self.start_button.setEnabled(ready)
        self.open_button.hide()
        self.result_label.hide()
        if ready:
            self.status_label.setText("输入已就绪。点击开始后只分析峰值拉应力对应的最近 DIC 帧。")
        else:
            self.status_label.setText("请选择 Ncorr 数据和对应的 MTS CSV。")

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
        self.start_button.setText("分析中…")
        self.open_button.hide()
        self.result_label.hide()
        self.status_label.setText("正在读取 MTS 峰值并计算对应 DIC 帧的全部裂缝宽度…")
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

        self.status_label.setText("✓ 分析完成" if status == "ok" else f"分析完成，但 COD 状态为：{status}")
        self.result_label.setText(
            f"峰值拉力 {self._fmt(result.get('peak_force_N'))} N @ {self._fmt(result.get('peak_time_s'), 3)} s  ·  "
            f"DIC Frame {result.get('selected_frame', '—')}  ·  Δt {self._fmt(result.get('match_error_s'), 3)} s\n"
            f"有效裂缝 {crack_count} 条  ·  平均 {self._fmt(result.get('mean_width_um'))} μm  ·  "
            f"P95 {self._fmt(result.get('p95_width_um'))} μm  ·  最大 {self._fmt(result.get('max_width_um'))} μm\n"
            f"Excel：{self.output_path}"
        )
        self.result_label.show()
        self.open_button.show()

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("分析失败")
        QMessageBox.critical(self, "分析失败", message)

    def _on_finished(self) -> None:
        self.worker = None
        self.start_button.setText("重新分析")
        self.start_button.setEnabled(self.data_file is not None and self.mts_file is not None)

    def _open_output_folder(self) -> None:
        if self.output_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path.parent)))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "正在分析", "当前分析尚未完成，请等待结果生成后再关闭。")
            event.ignore()
            return
        event.accept()
