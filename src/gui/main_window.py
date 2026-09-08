from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.config import load_config, with_mm_per_pixel
from src.core.input import FILE_DIALOG_FILTER
from src.gui.worker import AnalysisWorker


class MainWindow(QMainWindow):
    """Peak-tensile-stress frame analysis for Ncorr ECC/SHCC data."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CrackVision-DIC · Peak Stress COD")
        self.resize(920, 700)
        self.data_file: Path | None = None
        self.mts_file: Path | None = None
        self.out_dir: Path | None = None
        self.worker: AnalysisWorker | None = None
        self.config = load_config()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        title = QLabel("CrackVision-DIC")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        subtitle = QLabel(
            "MTS 峰值拉应力时刻 → 最近 DIC 帧 → 主拉应变裂缝识别 → COD"
        )
        subtitle.setStyleSheet("color:#5f6368;font-size:13px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setVerticalSpacing(14)

        data_row = QHBoxLayout()
        self.data_edit = QLineEdit()
        self.data_edit.setReadOnly(True)
        choose_data = QPushButton("选择 H5 / MAT")
        choose_data.clicked.connect(self._choose_data)
        data_row.addWidget(self.data_edit, 1)
        data_row.addWidget(choose_data)
        form.addRow("Ncorr 数据", data_row)

        mts_row = QHBoxLayout()
        self.mts_edit = QLineEdit()
        self.mts_edit.setReadOnly(True)
        choose_mts = QPushButton("选择 MTS CSV")
        choose_mts.clicked.connect(self._choose_mts)
        mts_row.addWidget(self.mts_edit, 1)
        mts_row.addWidget(choose_mts)
        form.addRow("MTS 原始数据", mts_row)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setReadOnly(True)
        choose_out = QPushButton("选择目录")
        choose_out.clicked.connect(self._choose_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(choose_out)
        form.addRow("输出目录", out_row)

        self.sync_spin = QDoubleSpinBox()
        self.sync_spin.setDecimals(3)
        self.sync_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.sync_spin.setSingleStep(0.5)
        self.sync_spin.setValue(0.0)
        self.sync_spin.setSuffix(" s")
        self.sync_spin.setToolTip(
            "填写 DIC 第0帧在 MTS 时间轴上的时刻。若相机与MTS同时开始，保持0。"
        )
        form.addRow("DIC 第0帧对应 MTS 时间", self.sync_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(6)
        self.scale_spin.setRange(0.000001, 1000.0)
        self.scale_spin.setSingleStep(0.001)
        self.scale_spin.setValue(
            float(self.config.get("experiment", {}).get("mm_per_pixel", 0.045))
        )
        self.scale_spin.setSuffix(" mm/px")
        self.scale_spin.setToolTip(
            "仅供旧 Ncorr MAT 缺少 pixtounits 时兜底；CrackVision H5 读取自身标定。"
        )
        form.addRow("尺度兜底", self.scale_spin)
        layout.addLayout(form)

        note = QLabel(
            "程序直接读取 MTS CSV 中的“力”和“时间”。对于恒定截面的拉伸试件，"
            "峰值拉应力与峰值拉力发生在同一时刻，因此无需输入截面积即可选择目标帧。"
            "软件只计算与该时刻最接近的一帧 DIC 数据，并在 Excel 中记录时间匹配误差。"
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background:#f5f7fa;border-radius:8px;padding:12px;color:#374151;"
        )
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.start_button = QPushButton("分析峰值拉应力帧")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("峰值时刻、匹配帧和 COD 状态会显示在这里。")
        layout.addWidget(self.log_box, 1)

        self.setStyleSheet(
            "QMainWindow{background:#ffffff;}"
            "QPushButton{padding:8px 14px;}"
            "QLineEdit,QDoubleSpinBox,QTextEdit{"
            "border:1px solid #d7dce2;border-radius:6px;padding:7px;}"
            "QProgressBar{height:12px;border:1px solid #d7dce2;"
            "border-radius:6px;text-align:center;}"
        )

    def _choose_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 CrackVision-Ncorr H5 或原始 Ncorr MAT",
            "",
            FILE_DIALOG_FILTER,
        )
        if not file_path:
            return
        self.data_file = Path(file_path)
        self.data_edit.setText(file_path)
        if self.out_dir is None:
            self.out_dir = self.data_file.parent / "CrackVision_Output"
            self.out_edit.setText(str(self.out_dir))

    def _choose_mts(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 MTS / DAQ CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if file_path:
            self.mts_file = Path(file_path)
            self.mts_edit.setText(file_path)

    def _choose_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.out_dir = Path(folder)
            self.out_edit.setText(folder)

    def _start(self) -> None:
        if self.data_file is None:
            QMessageBox.warning(self, "缺少输入", "请选择 Ncorr H5 / MAT 文件。")
            return
        if self.mts_file is None:
            QMessageBox.warning(self, "缺少 MTS", "请选择该试件对应的 MTS CSV 文件。")
            return
        if self.out_dir is None:
            QMessageBox.warning(self, "缺少输出目录", "请选择输出目录。")
            return

        run_config = with_mm_per_pixel(self.config, float(self.scale_spin.value()))
        self.worker = AnalysisWorker(
            self.data_file,
            self.mts_file,
            self.out_dir,
            run_config,
            float(self.sync_spin.value()),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log_box.clear()
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self._append_log("Cancelling...")

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)

    def _append_log(self, text: str) -> None:
        self.log_box.append(text)
        bar = self.log_box.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "分析失败", message)

    def _on_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        event.accept()
