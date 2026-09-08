from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.table import Table, TableStyleInfo


FRAME_WIDTH_COLUMNS_MM = (
    "W_median_mm",
    "W_avg_mm",
    "W_95_mm",
    "W_max_mm",
    "Crack_width_mean_mm",
    "Crack_width_median_mm",
    "Crack_width_95_mm",
    "Crack_width_max_mm",
)

CRACK_DETAIL_COLUMNS_MM = (
    "W_median_mm",
    "W_avg_mm",
    "W_95_mm",
    "W_max_mm",
    "Slip_median_mm",
)

NAVY = "1F4E78"
BLUE = "D9EAF7"
LIGHT = "F6F8FB"
GREEN = "E2F0D9"
RED = "FCE8E6"
GRAY = "667085"
WHITE = "FFFFFF"
BORDER = Side(style="thin", color="D9E1F2")


def prepare_frame_summary(rows: Iterable[dict]) -> pd.DataFrame:
    """Build the selected-frame table while preserving failed measurements as NaN."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    for col in FRAME_WIDTH_COLUMNS_MM:
        if col not in df.columns:
            df[col] = np.nan
        df[col.replace("_mm", "_um")] = (
            pd.to_numeric(df[col], errors="coerce") * 1000.0
        )
    return df.sort_values("Frame").reset_index(drop=True)


def prepare_crack_details(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [table for table in tables if table is not None and not table.empty]
    if not valid:
        return pd.DataFrame()
    df = pd.concat(valid, ignore_index=True)
    for col in CRACK_DETAIL_COLUMNS_MM:
        if col in df.columns:
            df[col.replace("_mm", "_um")] = (
                pd.to_numeric(df[col], errors="coerce") * 1000.0
            )
    return df


def _safe(row: pd.Series, key: str, default: object = "—") -> object:
    value = row.get(key, default)
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return default
    return value


def _metric_card(ws, label_cell: str, value_range: str, label: str, value: object, number_format: str | None = None) -> None:
    start, end = value_range.split(":")
    ws[label_cell] = label
    ws[label_cell].font = Font(size=10, color=GRAY, bold=True)
    ws[label_cell].alignment = Alignment(vertical="center")

    ws.merge_cells(value_range)
    cell = ws[start]
    cell.value = value
    cell.font = Font(size=16, bold=True, color=NAVY)
    cell.alignment = Alignment(vertical="center")
    cell.fill = PatternFill("solid", fgColor=LIGHT)
    cell.border = Border(left=BORDER, right=BORDER, top=BORDER, bottom=BORDER)
    if number_format is not None and isinstance(value, (int, float, np.integer, np.floating)):
        cell.number_format = number_format


def _write_summary_sheet(writer: pd.ExcelWriter, frame_df: pd.DataFrame, crack_df: pd.DataFrame) -> None:
    wb = writer.book
    ws = wb.create_sheet("01_结果汇总")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:H1")
    ws["A1"] = "CrackVision-DIC｜峰值拉应力状态裂缝宽度"
    ws["A1"].font = Font(size=18, bold=True, color=WHITE)
    ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 30

    ws.merge_cells("A2:H2")
    ws["A2"] = "统计口径：每条有效裂缝以沿线局部 COD 的中位数作为代表宽度；试件级统计按裂缝等权。"
    ws["A2"].font = Font(size=10, color=GRAY)
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[2].height = 28

    if frame_df.empty:
        ws["A4"] = "无可导出的结果"
        ws["A4"].font = Font(size=13, bold=True, color="B42318")
        return

    row = frame_df.iloc[0]

    _metric_card(ws, "A4", "A5:B5", "峰值拉力 (N)", _safe(row, "MTS_peak_force_N"), "0.0")
    _metric_card(ws, "C4", "C5:D5", "峰值时刻 (s)", _safe(row, "MTS_peak_time_s"), "0.000")
    _metric_card(ws, "E4", "E5:F5", "DIC 帧", int(_safe(row, "Frame", 0)) if _safe(row, "Frame", "—") != "—" else "—", "0")
    _metric_card(ws, "G4", "G5:H5", "时间匹配误差 (s)", _safe(row, "frame_match_error_s"), "+0.000;-0.000;0.000")

    _metric_card(ws, "A7", "A8:B8", "有效裂缝数", int(_safe(row, "crack_count", 0)) if _safe(row, "crack_count", "—") != "—" else "—", "0")
    _metric_card(ws, "C7", "C8:D8", "平均裂缝宽度 (μm)", _safe(row, "Crack_width_mean_um"), "0.0")
    _metric_card(ws, "E7", "E8:F8", "中位裂缝宽度 (μm)", _safe(row, "Crack_width_median_um"), "0.0")
    _metric_card(ws, "G7", "G8:H8", "P95 裂缝宽度 (μm)", _safe(row, "Crack_width_95_um"), "0.0")

    _metric_card(ws, "A10", "A11:B11", "最大裂缝宽度 (μm)", _safe(row, "Crack_width_max_um"), "0.0")
    status = str(_safe(row, "cod_status", "unknown"))
    _metric_card(ws, "C10", "C11:D11", "COD 状态", status)
    ws["C11"].fill = PatternFill("solid", fgColor=GREEN if status == "ok" else RED)

    for col in "ABCDEFGH":
        ws.column_dimensions[col].width = 17

    if crack_df is not None and not crack_df.empty and "W_median_um" in crack_df.columns:
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "峰值拉应力帧各裂缝代表宽度"
        chart.y_axis.title = "裂缝宽度 (μm)"
        chart.x_axis.title = "裂缝编号"
        chart.legend = None
        chart.height = 7.0
        chart.width = 15.5
        details_ws = wb["02_裂缝明细"]
        max_row = details_ws.max_row
        if max_row >= 2:
            data = Reference(details_ws, min_col=3, min_row=1, max_row=max_row)
            cats = Reference(details_ws, min_col=1, min_row=2, max_row=max_row)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            ws.add_chart(chart, "A14")

    ws.freeze_panes = "A4"


def _write_crack_detail_sheet(writer: pd.ExcelWriter, crack_df: pd.DataFrame) -> None:
    wb = writer.book
    ws = wb.create_sheet("02_裂缝明细")
    ws.sheet_view.showGridLines = False

    headers = ["裂缝编号", "裂缝长度 (mm)", "代表宽度 (μm)", "COD 有效点数", "拟合 R² 中位数"]
    ws.append(headers)

    if crack_df is not None and not crack_df.empty:
        table_df = crack_df.copy().sort_values("Crack_ID")
        for _, row in table_df.iterrows():
            ws.append(
                [
                    int(row.get("Crack_ID", 0)),
                    float(row.get("Length_mm", np.nan)),
                    float(row.get("W_median_um", np.nan)),
                    int(row.get("COD_samples", 0)),
                    float(row.get("Fit_R2_median", np.nan)),
                ]
            )

    for cell in ws[1]:
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 24

    widths = [12, 18, 20, 16, 18]
    for idx, width in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + idx)].width = width

    for row in ws.iter_rows(min_row=2):
        row[0].number_format = "0"
        row[1].number_format = "0.000"
        row[2].number_format = "0.0"
        row[3].number_format = "0"
        row[4].number_format = "0.000"
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")

    if ws.max_row >= 2:
        table = Table(displayName="CrackDetailsTable", ref=f"A1:E{ws.max_row}")
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        ws.add_table(table)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:E{max(ws.max_row, 1)}"


def _write_qa_sheet(writer: pd.ExcelWriter, frame_df: pd.DataFrame) -> None:
    wb = writer.book
    ws = wb.create_sheet("03_质量检查")
    ws.sheet_view.showGridLines = False
    ws.append(["检查项", "数值"])

    if frame_df.empty:
        qa_rows = [("结果", "无数据")]
    else:
        row = frame_df.iloc[0]
        qa_rows = [
            ("COD 状态", _safe(row, "cod_status")),
            ("峰值拉力 (N)", _safe(row, "MTS_peak_force_N")),
            ("峰值时刻 (s)", _safe(row, "MTS_peak_time_s")),
            ("选中 DIC 帧", _safe(row, "Frame")),
            ("DIC 时刻 (s)", _safe(row, "DIC_selected_time_s")),
            ("时间匹配误差 (s)", _safe(row, "frame_match_error_s")),
            ("像素尺度 (mm/px)", _safe(row, "pixel_size_mm")),
            ("DIC 网格步长 (px)", _safe(row, "dic_step_px")),
            ("DIC 网格间距 (mm/point)", _safe(row, "dic_point_spacing_mm")),
            ("有效区比例", _safe(row, "valid_fraction")),
            ("主拉应变阈值", _safe(row, "principal_strain_threshold")),
            ("候选裂缝点数", _safe(row, "candidate_points")),
            ("骨架点数", _safe(row, "skeleton_points")),
            ("有效裂缝数", _safe(row, "crack_count")),
            ("裂缝宽度统计口径", "每条裂缝 W_median 等权"),
            ("元数据来源", _safe(row, "metadata_source")),
        ]

    for item, value in qa_rows:
        ws.append([item, value])

    for cell in ws[1]:
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center")
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 44
    for row in ws.iter_rows(min_row=2):
        row[0].font = Font(bold=True, color=GRAY)
        row[0].fill = PatternFill("solid", fgColor=LIGHT)
        row[0].alignment = Alignment(vertical="center")
        row[1].alignment = Alignment(vertical="center", wrap_text=True)
        row[0].border = Border(bottom=BORDER)
        row[1].border = Border(bottom=BORDER)
    ws.freeze_panes = "A2"


def export_workbook(path: Path, frame_df: pd.DataFrame, crack_df: pd.DataFrame) -> None:
    """Export a compact, paper-facing workbook with only results, crack rows and QA."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Remove pandas' default worksheet and create sheets in the intended order.
        default_ws = writer.book.active
        writer.book.remove(default_ws)
        _write_crack_detail_sheet(writer, crack_df)
        _write_summary_sheet(writer, frame_df, crack_df)
        _write_qa_sheet(writer, frame_df)

        # Reorder after chart references are established.
        wb = writer.book
        wb._sheets = [wb["01_结果汇总"], wb["02_裂缝明细"], wb["03_质量检查"]]
