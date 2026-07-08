import logging
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import yaml

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QRadioButton,
    QButtonGroup,
    QStackedWidget,
    QTextBrowser,
    QComboBox,
    QScrollArea,
)
from PySide6.QtGui import QDesktopServices, QAction
from PySide6.QtCore import Qt, QUrl

from src.gui.worker import AnalysisPipelineWorker

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: dict = {
    "experiment": {
        "mm_per_pixel": 0.045,
        "dic_subset_spacing_px": 1.0,
        "sampling_interval_s": 5.0,
        "cross_section_area_mm2": 100.0,
        "gauge_length_mm": 80.0,
        "virtual_extensometer": {
            "left_fraction": 0.10,
            "right_fraction": 0.90,
            "band_width_points": 3,
            "gauge_length_mm": None,
        },
    },
    "sync": {
        "min_overlap_fraction": 0.60,
        "max_missing_fraction": 0.05,
        "override_dic_strain_with_mts": False,
        "dic_time_offset_s": 0.0,
        "mts_time_offset_s": 0.0,
    },
    "quality": {
        "enabled": True,
        "metric_mode": "finite_only",
        "threshold": None,
        "min_valid_fraction": 0.20,
    },
    "crack_detection": {
        "fusion_mode": "strain_or_image",
        "image_dilation_radius_points": 1,
        "strain_dilation_radius_points": 0,
        "require_strain_support": False,
    },
    "image_crack_detection": {
        "enabled": False,
        "image_dir": None,
        "filename_pattern": None,
        "frame_index_offset": 0,
        "dark_cracks": True,
        "background_sigma_px": 12.0,
        "threshold_quantile": 0.92,
        "otsu_weight": 0.75,
        "min_object_area_px": 20,
        "min_object_area_points": 5,
        "closing_radius_px": 1,
        "auto_discover_dir_names": ["images", "imgs", "frames", "camera", "crack_images"],
    },
    "physics": {
        "strain_threshold_k": 1.5,
        "min_cracking_strain": 1.5e-4,
        "max_cracking_strain_threshold": 0.03,
        "morphology_closing_radius_points": 0,
        "min_crack_area_points": 10,
        "min_crack_length_mm": 0.2,
        "elastic_modulus_mpa": None,
        "require_v_map_for_cod": True,
        "cod_sampling": {
            "delta_points": 3,
            "max_search_points": 15,
            "delta_mm": None,
            "max_search_mm": None,
        },
        "cod_min_mm": 0.002,
        "cod_min_mean_mm": 0.002,
        "cod_max_mm": 5.0,
        "enforce_monotonic_strain": True,
    },
    "validation": {"annotation_path": None},
    "export": {"target_strains": [0.2, 2.0, 4.0, 6.0]},
}


def deep_merge(base: dict, override: dict | None) -> dict:
    """Return base recursively merged with override without mutating either input."""
    merged = deepcopy(base)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class UserManualDialog(QDialog):
    """内置的科研用户说明书与排错指南。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CrackVision-DIC 用户说明书与物理参数指南")
        self.resize(820, 720)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(
            """
        <h2 style='color: #2F3640;'>CrackVision-DIC 物理分析引擎说明书</h2>

        <h3 style='color: #e1b12c;'>一、核心物理参数</h3>
        <ul>
            <li><b>DIC 帧间隔</b>: .mat 内没有真实时间轴时，系统用
                <code>Frame × Sampling Interval</code> 构造 DIC 时间轴。填错，MTS 同步直接偏航。</li>
            <li><b>缺省兜底比例尺</b>: .mat 读不到标定信息时才用。最终来源看 QA_Metadata。</li>
            <li><b>COD 底噪拦截</b>: 过滤亚像素噪声。默认 0.002 mm；噪声大就上调。</li>
            <li><b>MAD k</b>: DIC exx 裂缝候选区阈值。裂缝过多上调，断裂不连续下调。</li>
            <li><b>COD 采样距离</b>: delta_mm / max_search_mm 优先于 points。跨不同 subset spacing 时，物理长度更稳。</li>
            <li><b>弹性模量 E</b>: 用于 <code>W_global_est = max(0, strain - stress/E) × spacing</code> 的 sanity check。</li>
        </ul>

        <h3 style='color: #44BD32;'>二、裂缝识别方法</h3>
        <ul>
            <li><b>strain_or_image</b>: DIC 高 exx 区 + 相机图像 mask 取并集。默认，适合 ECC 细裂缝。</li>
            <li><b>strain_and_image</b>: 交集。更严格，但可能漏裂缝。</li>
            <li><b>image_near_strain</b>: 图像裂缝必须靠近 DIC 高应变支撑。</li>
            <li><b>image_only</b>: 只用图像找裂缝。能跑，但别拿它当主物理方法。</li>
        </ul>
        <p>相机图像 mask 只辅助找裂缝位置。主宽度仍来自 DIC 法向位移跳量。照片黑线宽不是神谕。</p>

        <h3 style='color: #e84118;'>三、经典翻车点</h3>
        <ul>
            <li><b>MTS 同步失败：</b>检查 DIC 帧间隔、MTS 起始时间、触发延迟和 CSV 时间列单位。</li>
            <li><b>COD 全部为 0：</b>若配置要求 v 位移场，请确认 .mat 中有 v_map。缺 v 会给出
                <code>missing_v_map_required</code>。</li>
            <li><b>图像 mask 没参与：</b>检查 image_crack_detection.enabled、image_dir、filename_pattern，导出 QA 看
                <code>image_mask_present</code>。</li>
            <li><b>裂缝数量暴涨：</b>调高 MAD k、min_crack_area_points、COD 底噪，或把融合模式改成 image_near_strain。</li>
        </ul>
        <hr>
        <p style='color: #718093; font-size: 12px;'>Compute first, audit second, publish last.</p>
        """
        )
        layout.addWidget(browser)

        btn_close = QPushButton("我已了解")
        btn_close.setMinimumHeight(40)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class DataPairingDialog(QDialog):
    def __init__(self, dic_dir: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("批处理: 智能数据对齐控制台")
        self.resize(860, 520)
        self.dic_dir = Path(dic_dir)
        self.paired_data: Dict[str, str] = {}
        self.mts_count = 0
        self._init_ui()
        self._load_dic_files()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        grp_auto = QGroupBox("智能配对区 (可选)")
        layout_auto = QHBoxLayout(grp_auto)

        self.edit_mts_dir = QLineEdit()
        self.edit_mts_dir.setPlaceholderText("选择包含 MTS .csv 文件的目录；严格匹配，避免 E1 误配 E10...")
        btn_browse_mts = QPushButton("选择 MTS 目录")
        btn_browse_mts.clicked.connect(self._select_mts_dir)

        btn_auto_match = QPushButton("一键严格配对")
        btn_auto_match.setStyleSheet("background-color: #2F3640; color: #FFFFFF; font-weight: bold;")
        btn_auto_match.clicked.connect(self._auto_match)

        layout_auto.addWidget(self.edit_mts_dir)
        layout_auto.addWidget(btn_browse_mts)
        layout_auto.addWidget(btn_auto_match)
        layout.addWidget(grp_auto)

        layout_tools = QHBoxLayout()
        layout_tools.addWidget(QLabel("勾选需要进入计算队列的试件：", styleSheet="font-weight: bold;"))
        layout_tools.addStretch()
        btn_select_all = QPushButton("全部勾选")
        btn_select_all.clicked.connect(lambda: self._toggle_all(Qt.CheckState.Checked))
        btn_deselect_all = QPushButton("全部取消")
        btn_deselect_all.clicked.connect(lambda: self._toggle_all(Qt.CheckState.Unchecked))
        layout_tools.addWidget(btn_select_all)
        layout_tools.addWidget(btn_deselect_all)
        layout.addLayout(layout_tools)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["DIC 数据矩阵 (勾选以计算)", "MTS 时域曲线 (为空即退回纯DIC)", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        btn_confirm = QPushButton("确认勾选并锁定队列")
        btn_confirm.setMinimumHeight(40)
        btn_confirm.clicked.connect(self._on_confirm)
        layout.addWidget(btn_confirm)

    def _toggle_all(self, state: Qt.CheckState) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(state)

    def _select_mts_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "选择 MTS 力学数据根目录")
        if dir_path:
            self.edit_mts_dir.setText(dir_path)
            self._auto_match()

    @staticmethod
    def _specimen_tokens(path_or_stem: str) -> tuple[str, ...]:
        stem = Path(path_or_stem).stem.lower()
        tokens = re.findall(r"[a-zA-Z]+|\d+", stem)
        stop_words = {
            "dic",
            "mts",
            "ncorr",
            "data",
            "result",
            "results",
            "strain",
            "stress",
            "force",
            "load",
            "disp",
            "displacement",
            "extension",
            "csv",
            "mat",
            "plot",
        }
        return tuple(token for token in tokens if token not in stop_words)

    @staticmethod
    def _contains_subsequence(container: tuple[str, ...], candidate: tuple[str, ...]) -> bool:
        if not candidate or len(candidate) > len(container):
            return False
        return any(container[i : i + len(candidate)] == candidate for i in range(len(container) - len(candidate) + 1))

    @classmethod
    def _match_score(cls, mat_stem: str, csv_stem: str) -> int:
        mat_tokens = cls._specimen_tokens(mat_stem)
        csv_tokens = cls._specimen_tokens(csv_stem)
        if not mat_tokens or not csv_tokens:
            return 0
        if mat_tokens == csv_tokens:
            return 100
        if cls._contains_subsequence(csv_tokens, mat_tokens) or cls._contains_subsequence(mat_tokens, csv_tokens):
            return 80
        common = set(mat_tokens) & set(csv_tokens)
        if len(common) >= max(2, math.ceil(len(set(mat_tokens)) * 0.8)):
            return 60
        return 0

    def _auto_match(self) -> None:
        mts_dir = self.edit_mts_dir.text().strip()
        if not mts_dir or not Path(mts_dir).exists():
            QMessageBox.information(self, "提示", "请先选择有效的 MTS 目录。")
            return

        csv_files = sorted(set(Path(mts_dir).rglob("*.csv")) | set(Path(mts_dir).rglob("*.CSV")))
        used_csv: set[Path] = set()
        match_count = 0
        ambiguous_count = 0

        for row in range(self.table.rowCount()):
            mat_item = self.table.item(row, 0)
            if mat_item is None:
                continue
            mat_path = Path(mat_item.data(Qt.ItemDataRole.UserRole))
            candidates: list[tuple[int, Path]] = []
            for csv_path in csv_files:
                if csv_path in used_csv:
                    continue
                score = self._match_score(mat_path.stem, csv_path.stem)
                if score > 0:
                    candidates.append((score, csv_path))
            candidates.sort(key=lambda item: (-item[0], len(item[1].stem), item[1].name.lower()))

            matched_csv = ""
            if candidates:
                best_score = candidates[0][0]
                best_candidates = [path for score, path in candidates if score == best_score]
                if len(best_candidates) == 1:
                    matched = best_candidates[0]
                    matched_csv = str(matched)
                    used_csv.add(matched)
                    match_count += 1
                else:
                    ambiguous_count += 1
            self.table.item(row, 1).setText(matched_csv)

        QMessageBox.information(
            self,
            "配对完成",
            f"严格匹配结束。\n尝试匹配: {self.table.rowCount()} 组\n成功匹配: {match_count} 组\n歧义跳过: {ambiguous_count} 组",
        )

    def _load_dic_files(self) -> None:
        if not self.dic_dir.exists():
            return
        mat_files = sorted(set(self.dic_dir.rglob("*.mat")) | set(self.dic_dir.rglob("*.MAT")))
        self.table.setRowCount(len(mat_files))

        for row, mat_file in enumerate(mat_files):
            item_mat = QTableWidgetItem(mat_file.name)
            item_mat.setData(Qt.ItemDataRole.UserRole, str(mat_file))
            item_mat.setFlags(item_mat.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item_mat.setCheckState(Qt.CheckState.Checked)

            self.table.setItem(row, 0, item_mat)
            self.table.setItem(row, 1, QTableWidgetItem(""))

            btn_browse = QPushButton("手动指定...")
            btn_browse.clicked.connect(lambda checked=False, r=row: self._browse_csv(r))
            self.table.setCellWidget(row, 2, btn_browse)

    def _browse_csv(self, row: int) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择力学文件", "", "CSV Files (*.csv *.CSV)")
        if file_path:
            self.table.item(row, 1).setText(file_path)

    def _on_confirm(self) -> None:
        self.paired_data.clear()
        self.mts_count = 0
        for row in range(self.table.rowCount()):
            item_mat = self.table.item(row, 0)
            if item_mat and item_mat.checkState() == Qt.CheckState.Checked:
                mat_path = item_mat.data(Qt.ItemDataRole.UserRole)
                csv_path = self.table.item(row, 1).text().strip()
                self.paired_data[mat_path] = csv_path
                if csv_path:
                    self.mts_count += 1
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CrackVision-DIC Core Engine")
        self.resize(780, 980)

        self.worker: Optional[AnalysisPipelineWorker] = None
        self.paired_dict: Dict[str, str] = {}
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = self.project_root / "config" / "default.yaml"
        self.config: dict = {}

        self._load_config()
        self._apply_minimalist_style()
        self._init_ui()
        self._init_menu_bar()

    def _apply_minimalist_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #F5F6FA; }
            QGroupBox { font-weight: bold; border: 1px solid #DCDDE1; border-radius: 4px; margin-top: 2ex; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #2F3640; }
            QPushButton { background-color: #ECDFE6; border: 1px solid #DCDDE1; padding: 6px; border-radius: 3px; color: #2F3640; }
            QPushButton:hover { background-color: #DCDDE1; }
            QPushButton#BtnStart { background-color: #2F3640; color: #F5F6FA; font-weight: bold; font-size: 11pt; border: none; }
            QPushButton#BtnStart:hover { background-color: #353B48; }
            QPushButton#BtnStart:disabled { background-color: #718093; }
            QLineEdit, QDoubleSpinBox, QComboBox { border: 1px solid #DCDDE1; padding: 5px; border-radius: 3px; background-color: #FFFFFF; }
            QTextEdit { border: 1px solid #DCDDE1; background-color: #2F3640; color: #F5F6FA; font-family: Consolas, monospace; font-size: 9pt; }
            QProgressBar { border: 1px solid #DCDDE1; border-radius: 3px; text-align: center; background-color: #FFFFFF; }
            QProgressBar::chunk { background-color: #44BD32; }
        """
        )

    def _load_config(self) -> None:
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                self.config = deep_merge(DEFAULT_CONFIG, loaded)
            else:
                self.config = deepcopy(DEFAULT_CONFIG)
                logger.warning("Config file not found, using built-in defaults: %s", self.config_path)
        except Exception:
            logger.exception("Failed to load config, using built-in defaults.")
            self.config = deepcopy(DEFAULT_CONFIG)

    def _init_menu_bar(self) -> None:
        menubar = self.menuBar()
        help_menu = menubar.addMenu("系统帮助 (&Help)")

        action_manual = QAction("打开用户说明书与排错指南", self)
        action_manual.setShortcut("F1")
        action_manual.triggered.connect(self._show_manual)
        help_menu.addAction(action_manual)

    def _show_manual(self) -> None:
        dialog = UserManualDialog(self)
        dialog.exec()

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        self._init_data_group(main_layout)
        self._init_param_group(main_layout)

        self.btn_start = QPushButton("启动物理分析引擎")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.clicked.connect(self._start_pipeline)
        main_layout.addWidget(self.btn_start)

        self.progress = QProgressBar()
        self.progress.setMinimumHeight(15)
        main_layout.addWidget(self.progress)

        main_layout.addWidget(QLabel("引擎运行日志 (Engine Logs):", styleSheet="font-weight: bold; color: #2F3640;"))
        self.logger_console = QTextEdit()
        self.logger_console.setReadOnly(True)
        self.logger_console.setMinimumHeight(180)
        main_layout.addWidget(self.logger_console)

        scroll.setWidget(content)
        root_layout.addWidget(scroll)

    def _init_data_group(self, layout: QVBoxLayout) -> None:
        grp = QGroupBox("I/O 挂载配置")
        main_vbox = QVBoxLayout()

        mode_layout = QHBoxLayout()
        self.radio_single = QRadioButton("单点计算 (Single)")
        self.radio_batch = QRadioButton("批处理队列 (Batch)")
        self.radio_single.setChecked(True)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_single)
        self.mode_group.addButton(self.radio_batch)
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_batch)
        main_vbox.addLayout(mode_layout)

        self.stack_data = QStackedWidget()

        page_single = QWidget()
        form_single = QFormLayout(page_single)
        form_single.setContentsMargins(0, 5, 0, 0)
        self.edit_s_mat = QLineEdit()
        btn_s_mat = QPushButton("浏览")
        btn_s_mat.clicked.connect(lambda: self._select_file(self.edit_s_mat, "MAT Files (*.mat *.MAT)"))
        h_s_mat = QHBoxLayout()
        h_s_mat.addWidget(self.edit_s_mat)
        h_s_mat.addWidget(btn_s_mat)
        form_single.addRow("DIC 矩阵文件:", h_s_mat)

        self.edit_s_mts = QLineEdit()
        self.edit_s_mts.setPlaceholderText("无力学数据可留空")
        btn_s_mts = QPushButton("浏览")
        btn_s_mts.clicked.connect(lambda: self._select_file(self.edit_s_mts, "CSV Files (*.csv *.CSV)"))
        h_s_mts = QHBoxLayout()
        h_s_mts.addWidget(self.edit_s_mts)
        h_s_mts.addWidget(btn_s_mts)
        form_single.addRow("MTS 时域文件:", h_s_mts)
        self.stack_data.addWidget(page_single)

        page_batch = QWidget()
        form_batch = QFormLayout(page_batch)
        form_batch.setContentsMargins(0, 5, 0, 0)
        self.edit_dic_dir = QLineEdit()
        btn_dic_dir = QPushButton("浏览")
        btn_dic_dir.clicked.connect(lambda: self._select_dir(self.edit_dic_dir))
        h_dic_dir = QHBoxLayout()
        h_dic_dir.addWidget(self.edit_dic_dir)
        h_dic_dir.addWidget(btn_dic_dir)
        form_batch.addRow("DIC 工作目录:", h_dic_dir)

        self.btn_pair = QPushButton("打开智能挂载台")
        self.btn_pair.clicked.connect(self._open_pairing_dialog)
        self.lbl_pair_status = QLabel("状态: 未挂载")
        h_pair = QHBoxLayout()
        h_pair.addWidget(self.btn_pair)
        h_pair.addWidget(self.lbl_pair_status)
        form_batch.addRow("批处理队列:", h_pair)
        self.stack_data.addWidget(page_batch)

        main_vbox.addWidget(self.stack_data)
        self.radio_single.toggled.connect(lambda: self.stack_data.setCurrentIndex(0))
        self.radio_batch.toggled.connect(lambda: self.stack_data.setCurrentIndex(1))

        form_out = QFormLayout()
        form_out.setContentsMargins(0, 10, 0, 0)
        self.edit_out = QLineEdit()
        btn_out = QPushButton("浏览")
        btn_out.clicked.connect(lambda: self._select_dir(self.edit_out))
        h_out = QHBoxLayout()
        h_out.addWidget(self.edit_out)
        h_out.addWidget(btn_out)
        form_out.addRow("输出落地目录:", h_out)

        main_vbox.addLayout(form_out)
        grp.setLayout(main_vbox)
        layout.addWidget(grp)

    def _init_param_group(self, layout: QVBoxLayout) -> None:
        grp = QGroupBox("物理算子、裂缝识别与宽度方法")
        form = QFormLayout()

        experiment = self.config.get("experiment", {})
        physics = self.config.get("physics", {})
        sampling = physics.get("cod_sampling", {}) or {}
        crack_detection = self.config.get("crack_detection", {}) or {}
        image_cfg = self.config.get("image_crack_detection", {}) or {}

        form.addRow(self._section_label("基础标定"))

        self.spin_gauge_len = QDoubleSpinBox()
        self.spin_gauge_len.setRange(1.0, 1000.0)
        self.spin_gauge_len.setDecimals(1)
        self.spin_gauge_len.setSingleStep(5.0)
        self.spin_gauge_len.setValue(float(experiment.get("gauge_length_mm", 80.0)))
        form.addRow("宏观拉伸标距 (mm):", self.spin_gauge_len)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(1e-5, 10.0)
        self.spin_scale.setDecimals(5)
        self.spin_scale.setSingleStep(0.0001)
        self.spin_scale.setValue(float(experiment.get("mm_per_pixel", 0.045)))
        form.addRow("兜底比例尺 (mm/px):", self.spin_scale)

        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(1e-6, 3600.0)
        self.spin_interval.setDecimals(4)
        self.spin_interval.setSingleStep(0.1)
        self.spin_interval.setValue(float(experiment.get("sampling_interval_s", 5.0)))
        form.addRow("DIC 帧间隔兜底 (s/frame):", self.spin_interval)

        form.addRow(self._section_label("DIC 裂缝与 COD"))

        self.spin_cod_min = QDoubleSpinBox()
        self.spin_cod_min.setRange(0.0, 10.0)
        self.spin_cod_min.setDecimals(4)
        self.spin_cod_min.setSingleStep(0.001)
        self.spin_cod_min.setValue(float(physics.get("cod_min_mm", 0.002)))
        form.addRow("COD 底噪拦截 (mm):", self.spin_cod_min)

        self.spin_k = QDoubleSpinBox()
        self.spin_k.setRange(0.1, 20.0)
        self.spin_k.setDecimals(2)
        self.spin_k.setSingleStep(0.1)
        self.spin_k.setValue(float(physics.get("strain_threshold_k", 1.5)))
        form.addRow("MAD 稳健阈值 (k):", self.spin_k)

        self.chk_require_v = QCheckBox("要求 v 位移场；斜裂缝 COD 更靠谱")
        self.chk_require_v.setChecked(bool(physics.get("require_v_map_for_cod", True)))
        form.addRow("COD 向量模式:", self.chk_require_v)

        self.spin_delta_mm = QDoubleSpinBox()
        self.spin_delta_mm.setRange(0.0, 50.0)
        self.spin_delta_mm.setDecimals(3)
        self.spin_delta_mm.setSingleStep(0.01)
        self.spin_delta_mm.setValue(self._optional_float(sampling.get("delta_mm")))
        form.addRow("COD 起采距离 delta_mm (0=按points):", self.spin_delta_mm)

        self.spin_max_search_mm = QDoubleSpinBox()
        self.spin_max_search_mm.setRange(0.0, 100.0)
        self.spin_max_search_mm.setDecimals(3)
        self.spin_max_search_mm.setSingleStep(0.01)
        self.spin_max_search_mm.setValue(self._optional_float(sampling.get("max_search_mm")))
        form.addRow("COD 搜索窗口 max_search_mm (0=按points):", self.spin_max_search_mm)

        self.spin_elastic_modulus = QDoubleSpinBox()
        self.spin_elastic_modulus.setRange(0.0, 200000.0)
        self.spin_elastic_modulus.setDecimals(1)
        self.spin_elastic_modulus.setSingleStep(500.0)
        self.spin_elastic_modulus.setValue(self._optional_float(physics.get("elastic_modulus_mpa")))
        form.addRow("弹性模量 E (MPa, 0=不用扣除 σ/E):", self.spin_elastic_modulus)

        self.chk_monotonic = QCheckBox("强制全局 DIC 应变单调不下降")
        self.chk_monotonic.setChecked(bool(physics.get("enforce_monotonic_strain", True)))
        form.addRow("应变后处理:", self.chk_monotonic)

        form.addRow(self._section_label("裂缝识别融合"))

        self.combo_fusion = QComboBox()
        self.combo_fusion.addItems(["strain_or_image", "strain_and_image", "image_near_strain", "image_only", "strain_only"])
        self._set_combo_current(self.combo_fusion, str(crack_detection.get("fusion_mode", "strain_or_image")))
        form.addRow("融合模式:", self.combo_fusion)

        self.spin_image_dilate = QDoubleSpinBox()
        self.spin_image_dilate.setRange(0, 20)
        self.spin_image_dilate.setDecimals(0)
        self.spin_image_dilate.setSingleStep(1)
        self.spin_image_dilate.setValue(float(crack_detection.get("image_dilation_radius_points", 1)))
        form.addRow("图像 mask 膨胀半径 (DIC points):", self.spin_image_dilate)

        self.chk_strain_support = QCheckBox("图像裂缝必须靠近 DIC 高应变支撑")
        self.chk_strain_support.setChecked(bool(crack_detection.get("require_strain_support", False)))
        form.addRow("保守过滤:", self.chk_strain_support)

        form.addRow(self._section_label("相机图像辅助"))

        self.chk_image_enabled = QCheckBox("启用相机图像 crack mask")
        self.chk_image_enabled.setChecked(bool(image_cfg.get("enabled", False)))
        self.chk_image_enabled.toggled.connect(self._toggle_image_controls)
        form.addRow("图像辅助:", self.chk_image_enabled)

        self.edit_image_dir = QLineEdit(str(image_cfg.get("image_dir") or ""))
        self.edit_image_dir.setPlaceholderText("留空自动找 images/imgs/frames/camera/crack_images；相对路径基于 .mat 所在目录")
        btn_img_dir = QPushButton("浏览")
        btn_img_dir.clicked.connect(lambda: self._select_dir(self.edit_image_dir))
        h_img = QHBoxLayout()
        h_img.addWidget(self.edit_image_dir)
        h_img.addWidget(btn_img_dir)
        form.addRow("图片目录:", h_img)
        self.btn_image_dir = btn_img_dir

        self.edit_image_pattern = QLineEdit(str(image_cfg.get("filename_pattern") or ""))
        self.edit_image_pattern.setPlaceholderText("可空。例：frame_{frame:04d}.png")
        form.addRow("图片命名模板:", self.edit_image_pattern)

        self.spin_frame_offset = QDoubleSpinBox()
        self.spin_frame_offset.setRange(-100000, 100000)
        self.spin_frame_offset.setDecimals(0)
        self.spin_frame_offset.setSingleStep(1)
        self.spin_frame_offset.setValue(float(image_cfg.get("frame_index_offset", 0)))
        form.addRow("图像帧偏移:", self.spin_frame_offset)

        self.chk_dark_cracks = QCheckBox("暗裂缝/黑线模式")
        self.chk_dark_cracks.setChecked(bool(image_cfg.get("dark_cracks", True)))
        form.addRow("图像阈值方向:", self.chk_dark_cracks)

        form.addRow(self._section_label("输出切片"))
        target_strains = self.config.get("export", {}).get("target_strains", [0.2, 2.0, 4.0, 6.0])
        self.edit_target_strains = QLineEdit(", ".join(str(v) for v in target_strains))
        form.addRow("多梯度切片目标 (%):", self.edit_target_strains)

        grp.setLayout(form)
        layout.addWidget(grp)
        self._toggle_image_controls(self.chk_image_enabled.isChecked())

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; color: #2F3640; padding-top: 8px;")
        return label

    @staticmethod
    def _optional_float(value: object) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _optional_positive(value: float) -> float | None:
        return float(value) if value > 0 else None

    @staticmethod
    def _set_combo_current(combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _toggle_image_controls(self, enabled: bool) -> None:
        for widget in [
            getattr(self, "edit_image_dir", None),
            getattr(self, "btn_image_dir", None),
            getattr(self, "edit_image_pattern", None),
            getattr(self, "spin_frame_offset", None),
            getattr(self, "chk_dark_cracks", None),
            getattr(self, "spin_image_dilate", None),
            getattr(self, "chk_strain_support", None),
        ]:
            if widget is not None:
                widget.setEnabled(enabled)

    def _select_dir(self, line_edit: QLineEdit) -> None:
        dialog = QFileDialog(self, "选择目录")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        if dialog.exec():
            paths = dialog.selectedFiles()
            if paths:
                line_edit.setText(str(Path(paths[0])))

    def _select_file(self, line_edit: QLineEdit, file_filter: str) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def _open_pairing_dialog(self) -> None:
        dic_dir = self.edit_dic_dir.text()
        if not dic_dir or not Path(dic_dir).exists():
            QMessageBox.warning(self, "中断", "请先挂载有效的 DIC 工作目录。")
            return

        dialog = DataPairingDialog(dic_dir, self)
        if dialog.exec():
            self.paired_dict = dialog.paired_data
            self.lbl_pair_status.setText(f"已勾选: {len(self.paired_dict)} 组 (其中含 MTS: {dialog.mts_count} 组)")

    def _update_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress.setValue(int(current / total * 100))

    def _start_pipeline(self) -> None:
        out_dir_str = self.edit_out.text().strip()
        if not out_dir_str:
            QMessageBox.warning(self, "中断", "输出目录不能为空。")
            return

        gauge_len = self.spin_gauge_len.value()
        scale_val = self.spin_scale.value()
        sampling_interval = self.spin_interval.value()
        cod_min = self.spin_cod_min.value()

        if gauge_len <= 0.1:
            QMessageBox.critical(self, "物理参数违规", "宏观标距 (Gauge Length) 必须大于 0.1 mm。")
            return
        if scale_val <= 0.00001:
            QMessageBox.critical(self, "物理参数违规", "比例尺 (Scale) 设置过低，请检查小数点精度。")
            return
        if sampling_interval <= 0:
            QMessageBox.critical(self, "物理参数违规", "DIC 帧间隔必须大于 0。")
            return
        if cod_min == 0.0:
            reply = QMessageBox.question(
                self,
                "高危操作确认",
                "你把 COD 底噪拦截设成了 0 mm。\nDIC 插值噪声会大摇大摆进表。\n\n坚持使用 0 mm？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                self.spin_cod_min.setValue(0.002)
                return

        try:
            target_strains = [float(s.strip()) for s in self.edit_target_strains.text().split(",") if s.strip()]
            if not target_strains:
                raise ValueError("empty target strains")

            self.config.setdefault("export", {})["target_strains"] = target_strains

            experiment = self.config.setdefault("experiment", {})
            experiment["gauge_length_mm"] = gauge_len
            experiment["mm_per_pixel"] = scale_val
            experiment["sampling_interval_s"] = sampling_interval

            physics = self.config.setdefault("physics", {})
            physics["strain_threshold_k"] = self.spin_k.value()
            physics["cod_min_mm"] = cod_min
            physics["enforce_monotonic_strain"] = self.chk_monotonic.isChecked()
            physics["require_v_map_for_cod"] = self.chk_require_v.isChecked()
            physics["elastic_modulus_mpa"] = self._optional_positive(self.spin_elastic_modulus.value())

            sampling = physics.setdefault("cod_sampling", {})
            sampling["delta_mm"] = self._optional_positive(self.spin_delta_mm.value())
            sampling["max_search_mm"] = self._optional_positive(self.spin_max_search_mm.value())

            crack_detection = self.config.setdefault("crack_detection", {})
            crack_detection["fusion_mode"] = self.combo_fusion.currentText()
            crack_detection["image_dilation_radius_points"] = int(self.spin_image_dilate.value())
            crack_detection["require_strain_support"] = self.chk_strain_support.isChecked()

            image_cfg = self.config.setdefault("image_crack_detection", {})
            image_cfg["enabled"] = self.chk_image_enabled.isChecked()
            image_cfg["image_dir"] = self.edit_image_dir.text().strip() or None
            image_cfg["filename_pattern"] = self.edit_image_pattern.text().strip() or None
            image_cfg["frame_index_offset"] = int(self.spin_frame_offset.value())
            image_cfg["dark_cracks"] = self.chk_dark_cracks.isChecked()
        except ValueError:
            QMessageBox.warning(self, "格式错误", "参数解析失败。目标应变请写成 0.2, 2.0, 4.0 这种格式。")
            return

        process_dict = {}
        if self.radio_single.isChecked():
            mat_f = self.edit_s_mat.text().strip()
            if not mat_f:
                QMessageBox.warning(self, "中断", "单次模式下必须指定 MAT 文件。")
                return
            if not Path(mat_f).exists():
                QMessageBox.warning(self, "中断", "MAT 文件不存在。")
                return
            process_dict[mat_f] = self.edit_s_mts.text().strip()
        else:
            if not self.paired_dict:
                QMessageBox.warning(self, "中断", "批处理队列为空。请先进入【智能挂载台】勾选试件。")
                return
            process_dict = self.paired_dict

        Path(out_dir_str).mkdir(parents=True, exist_ok=True)
        self.btn_start.setEnabled(False)
        self.btn_start.setText("引擎全速运转中...")
        self.progress.setValue(0)
        self.logger_console.clear()

        self.logger_console.append(f"[Pre-flight] 队列: {len(process_dict)} 组")
        self.logger_console.append(
            f"[Core] 标距={gauge_len:.1f} mm | 帧间隔={sampling_interval:.4f} s | COD底噪={cod_min:.4f} mm | require_v={self.chk_require_v.isChecked()}"
        )
        self.logger_console.append(
            f"[Crack] fusion={self.combo_fusion.currentText()} | image_mask={self.chk_image_enabled.isChecked()} | E={physics.get('elastic_modulus_mpa') or 'none'} MPa"
        )
        if self.chk_image_enabled.isChecked():
            self.logger_console.append(
                f"[Image] dir={self.edit_image_dir.text().strip() or 'auto-discover'} | pattern={self.edit_image_pattern.text().strip() or 'sorted/auto'}"
            )
        self.logger_console.append("-" * 50)

        self.worker = AnalysisPipelineWorker(process_dict, Path(out_dir_str), self.config)
        self.worker.error_occurred.connect(lambda err: self.logger_console.append(f"\n[FATAL] {err}"))
        self.worker.log_emitted.connect(self.logger_console.append)
        self.worker.progress_updated.connect(self._update_progress)
        self.worker.specimen_processed.connect(
            lambda p1, p2: self.logger_console.append(f"[SUCCESS] 数据已落盘: {Path(p1).stem}")
        )
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.start()

    def _on_pipeline_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_start.setText("启动物理分析引擎")
        out_dir = self.edit_out.text().strip()
        if out_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))
