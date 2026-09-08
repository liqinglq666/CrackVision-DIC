from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter


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


def build_qa(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame([{"Metric": "frames", "Value": 0}])
    rows: list[dict[str, object]] = [
        {"Metric": "frames", "Value": int(len(frame_df))},
        {
            "Metric": "frames_cod_ok",
            "Value": int((frame_df["cod_status"] == "ok").sum()),
        },
        {
            "Metric": "frames_cod_failed",
            "Value": int((frame_df["cod_status"] != "ok").sum()),
        },
        {
            "Metric": "median_valid_fraction",
            "Value": float(
                pd.to_numeric(frame_df["valid_fraction"], errors="coerce").median()
            ),
        },
    ]
    for status, count in frame_df["cod_status"].value_counts(dropna=False).items():
        rows.append({"Metric": f"status::{status}", "Value": int(count)})
    for col in (
        "pixel_size_mm",
        "dic_step_px",
        "dic_point_spacing_mm",
        "metadata_source",
        "crack_width_basis",
        "crack_representative_count",
    ):
        if col in frame_df.columns:
            first = frame_df[col].dropna()
            if not first.empty:
                rows.append({"Metric": col, "Value": first.iloc[0]})
    return pd.DataFrame(rows)


def export_workbook(
    path: Path,
    frame_df: pd.DataFrame,
    crack_df: pd.DataFrame,
    qa_df: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    readme = pd.DataFrame(
        {
            "Item": [
                "Selected state",
                "Frame matching",
                "Per-crack representative width",
                "Specimen average crack width",
                "Frame-summary compatibility fields",
                "Failure semantics",
                "Crack detection",
                "COD method",
                "Units",
            ],
            "Meaning": [
                "Only the DIC frame nearest to the MTS peak tensile-force/stress time is analysed",
                "frame_match_error_s = selected-frame MTS-equivalent time minus true MTS peak time",
                "Each row in 02_Crack_Details is one accepted crack; W_median_um is that crack's representative width",
                "Crack_width_mean_um = arithmetic mean of all accepted cracks' W_median_um values; every crack has equal weight",
                "Frame-level W_avg/W_median/W_95/W_max are aliases of the same equal-weight per-crack representative-width distribution",
                "NaN means not measurable; it is never silently converted to zero",
                "Maximum principal tensile strain from Exx/Eyy/Exy",
                "Multi-point linear fits on both crack faces extrapolated to the crack plane",
                "Displacement: Ncorr px -> mm via pixel_size_mm; geometry: DIC grid -> mm via dic_point_spacing_mm",
            ],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="00_READ_ME", index=False)
        frame_df.to_excel(writer, sheet_name="01_Frame_Summary", index=False)
        crack_df.to_excel(writer, sheet_name="02_Crack_Details", index=False)
        qa_df.to_excel(writer, sheet_name="03_QA", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for idx, column in enumerate(ws.columns, start=1):
                max_len = min(
                    60,
                    max(
                        (
                            len(str(cell.value)) if cell.value is not None else 0
                            for cell in column
                        ),
                        default=0,
                    )
                    + 2,
                )
                ws.column_dimensions[get_column_letter(idx)].width = max(10, max_len)
