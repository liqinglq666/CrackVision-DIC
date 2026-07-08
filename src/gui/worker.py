import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from src.core.evolution_analyzer import EvolutionAnalyzer
from src.core.io_sync import PipelineIO
from src.core.physics import CrackPhysicsEngine

import logging

logger = logging.getLogger(__name__)

_worker_engine = None


@dataclass(frozen=True)
class FrameTaskPayload:
    config: dict
    u_path: str
    exx_path: str
    mask_path: str
    v_path: Optional[str]
    quality_path: Optional[str]
    ratio: float
    dic_point_spacing_mm: float
    subset_spacing_px: float
    metadata_source: str
    frame_id: int
    time_s: float


def _median_band(arr: np.ndarray, valid: np.ndarray, x_center: int, half_width: int) -> float:
    x0 = max(0, x_center - half_width)
    x1 = min(arr.shape[1], x_center + half_width + 1)
    roi = arr[:, x0:x1]
    roi_valid = valid[:, x0:x1] & np.isfinite(roi)
    if np.count_nonzero(roi_valid) == 0:
        return np.nan
    return float(np.nanmedian(roi[roi_valid]))


def _virtual_extensometer_strain(
    u: np.ndarray,
    valid_mask: np.ndarray,
    displacement_scale_mm: float,
    dic_point_spacing_mm: float,
    config: dict,
) -> tuple[float, float, int, int, str]:
    experiment = config.get("experiment", {})
    ve = experiment.get("virtual_extensometer", {})
    active_cols = np.where(np.any(valid_mask & np.isfinite(u), axis=0))[0]
    if active_cols.size < 5:
        return 0.0, float(experiment.get("gauge_length_mm", 80.0)), -1, -1, "no_valid_columns"

    left_fraction = float(ve.get("left_fraction", 0.10))
    right_fraction = float(ve.get("right_fraction", 0.90))
    band_width_points = max(1, int(ve.get("band_width_points", 3)))
    x_min, x_max = int(active_cols[0]), int(active_cols[-1])
    span = max(1, x_max - x_min)
    left_x = int(round(x_min + span * left_fraction))
    right_x = int(round(x_min + span * right_fraction))
    if right_x <= left_x:
        return 0.0, float(experiment.get("gauge_length_mm", 80.0)), left_x, right_x, "invalid_extensometer"

    left_u = _median_band(u, valid_mask, left_x, band_width_points)
    right_u = _median_band(u, valid_mask, right_x, band_width_points)
    if np.isnan(left_u) or np.isnan(right_u):
        return 0.0, float(experiment.get("gauge_length_mm", 80.0)), left_x, right_x, "missing_band_displacement"

    configured_l0 = ve.get("gauge_length_mm")
    if configured_l0 is not None and float(configured_l0) > 0:
        gauge_len = float(configured_l0)
        source = "dic_virtual_extensometer_configured_L0"
    else:
        gauge_len = abs(right_x - left_x) * float(dic_point_spacing_mm)
        source = "dic_virtual_extensometer_grid_L0"

    if gauge_len <= 0:
        return 0.0, float(experiment.get("gauge_length_mm", 80.0)), left_x, right_x, "invalid_gauge_length"
    du_mm = abs(right_u - left_u) * float(displacement_scale_mm)
    return max(0.0, float(du_mm / gauge_len)), gauge_len, left_x, right_x, source


def analyze_single_frame_task(payload: FrameTaskPayload) -> Optional[Dict[str, Any]]:
    global _worker_engine
    try:
        if _worker_engine is None:
            _worker_engine = CrackPhysicsEngine(payload.config)

        u = np.load(payload.u_path)
        exx = np.load(payload.exx_path)
        mask = np.load(payload.mask_path).astype(bool)
        v = np.load(payload.v_path) if payload.v_path else None
        quality = np.load(payload.quality_path) if payload.quality_path else None

        quality_mask, quality_fraction, quality_reason = _worker_engine.build_quality_mask(mask, u, v, quality)
        strain, virtual_l0, left_x, right_x, strain_source = _virtual_extensometer_strain(
            u, quality_mask, payload.ratio, payload.dic_point_spacing_mm, payload.config
        )

        if quality_fraction < _worker_engine.min_valid_fraction:
            res = _worker_engine._empty("quality_rejected")
            threshold = 0.0
        else:
            skeleton, threshold = _worker_engine.extract_skeleton(exx, quality_mask)
            res = _worker_engine.compute_cod(
                u,
                skeleton,
                payload.ratio,
                payload.dic_point_spacing_mm,
                v_map=v,
            )

        res.update(
            {
                "Frame": int(payload.frame_id),
                "Time_s": float(payload.time_s),
                "global_strain": max(0.0, float(strain)),
                "virtual_gauge_length_mm": float(virtual_l0),
                "virtual_left_col": int(left_x),
                "virtual_right_col": int(right_x),
                "strain_source": strain_source,
                "quality_valid_fraction": float(quality_fraction),
                "quality_filter": quality_reason,
                "quality_map_present": bool(quality is not None),
                "strain_threshold_used": float(threshold),
                "pixel_size_mm": float(payload.ratio),
                "subset_spacing_px": float(payload.subset_spacing_px),
                "dic_point_spacing_mm": float(payload.dic_point_spacing_mm),
                "metadata_source": payload.metadata_source,
                "v_map_present": bool(v is not None),
            }
        )

        crack_count = int(res.get("crack_count", 0))
        res["crack_spacing_mm"] = float(virtual_l0 / crack_count) if crack_count > 0 and virtual_l0 > 0 else 0.0

        return res
    except Exception as e:
        logger.error("Frame task crashed (Frame %s): %s", payload.frame_id, e, exc_info=True)
        return None


class AnalysisPipelineWorker(QThread):
    progress_updated = Signal(int, int)
    log_emitted = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()
    specimen_processed = Signal(str, str)

    def __init__(self, paired_data: dict, out_dir: Path, config: dict) -> None:
        super().__init__()
        self.paired_data = paired_data
        self.out_dir = out_dir
        self.config = config
        self._is_running = True

    def run(self) -> None:
        try:
            fallback_ratio = float(self.config["experiment"]["mm_per_pixel"])
            interval = float(self.config["experiment"]["sampling_interval_s"])
            total = len(self.paired_data)

            for i, (mat_f, mts_f) in enumerate(self.paired_data.items()):
                if not self._is_running:
                    break
                try:
                    self._process_specimen(Path(mat_f), Path(mts_f) if mts_f else None, fallback_ratio, interval)
                except Exception as e:
                    self.log_emitted.emit(f"Failed specimen {Path(mat_f).name}; skipped: {e}")
                    logger.error("Specimen process failed: %s", mat_f, exc_info=True)
                self.progress_updated.emit(i + 1, total)

        except Exception as e:
            self.error_occurred.emit(f"Pipeline scheduler crashed: {e}")
        finally:
            self.finished.emit()

    def _process_specimen(self, mat_path: Path, mts_path: Optional[Path], ratio: float, interval: float) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="cv_"))
        tasks = []
        first_meta = None

        try:
            for frame in PipelineIO.stream_dic_frames(mat_path, ratio, self.config):
                if first_meta is None:
                    first_meta = frame
                    self.log_emitted.emit(
                        f"Parsing {mat_path.name} | pixel={frame.ratio:.6f} mm/px | "
                        f"DIC step={frame.subset_spacing_px:.3f} px | "
                        f"grid={frame.dic_point_spacing_mm:.6f} mm/point | source={frame.metadata_source}"
                    )

                u_p = temp_dir / f"u_{frame.frame_id}.npy"
                exx_p = temp_dir / f"exx_{frame.frame_id}.npy"
                mask_p = temp_dir / f"mask_{frame.frame_id}.npy"
                v_p = temp_dir / f"v_{frame.frame_id}.npy" if frame.v_map is not None else None
                q_p = temp_dir / f"q_{frame.frame_id}.npy" if frame.quality_map is not None else None
                np.save(u_p, frame.u_map)
                np.save(exx_p, frame.exx_map)
                np.save(mask_p, frame.mask)
                if v_p is not None:
                    np.save(v_p, frame.v_map)
                if q_p is not None:
                    np.save(q_p, frame.quality_map)

                tasks.append(
                    FrameTaskPayload(
                        self.config,
                        str(u_p),
                        str(exx_p),
                        str(mask_p),
                        str(v_p) if v_p is not None else None,
                        str(q_p) if q_p is not None else None,
                        frame.ratio,
                        float(frame.dic_point_spacing_mm),
                        frame.subset_spacing_px,
                        frame.metadata_source,
                        frame.frame_id,
                        frame.frame_id * interval,
                    )
                )

            if not tasks:
                self.log_emitted.emit(f"{mat_path.name} contains no DIC frames.")
                return

            results = []
            cur_max_strain = 0.0
            max_workers = min(10, max(1, multiprocessing.cpu_count() - 2))

            with ProcessPoolExecutor(max_workers=max_workers) as exec:
                for res in exec.map(analyze_single_frame_task, tasks):
                    if res:
                        cur_max_strain = max(cur_max_strain, res.get("global_strain", 0.0))
                        res["global_strain"] = cur_max_strain
                        results.append(res)

            if not results:
                self.log_emitted.emit(f"{mat_path.name} extraction failed: no valid frames.")
                return

            results.sort(key=lambda x: x["Frame"])
            df = pd.DataFrame(results)

            if mts_path and mts_path.exists():
                try:
                    df = EvolutionAnalyzer(self.config, mts_path).synchronize(df)
                    self.log_emitted.emit(f"{mat_path.name} MTS sync passed strict overlap checks.")
                except Exception as e:
                    df["sync_status"] = f"failed: {e}"
                    self.log_emitted.emit(f"MTS sync failed; kept DIC virtual strain: {e}")
            else:
                df["sync_status"] = "no_mts"

            self._export_results(mat_path, df, results)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _export_results(self, mat_path: Path, df: pd.DataFrame, results: list[Dict[str, Any]]) -> None:
        df["Strain_pct"] = df.get("global_strain", 0.0) * 100.0
        df["W_avg_um"] = df.get("w_avg", 0.0) * 1000.0
        df["W_max_um"] = df.get("w_max", 0.0) * 1000.0
        df["W_99_um"] = df.get("w_99", 0.0) * 1000.0
        df["crack_spacing_mm"] = df.get("crack_spacing_mm", 0.0)

        for col in [
            "Strain_pct",
            "crack_count",
            "crack_spacing_mm",
            "W_avg_um",
            "W_99_um",
            "W_max_um",
            "quality_valid_fraction",
            "strain_threshold_used",
        ]:
            if col not in df.columns:
                df[col] = 0.0

        sat_idx = df["crack_count"].idxmax()
        if "Stress_MPa" in df.columns and not df["Stress_MPa"].isna().all():
            ult_idx = df["Stress_MPa"].idxmax()
        else:
            ult_idx = df["Strain_pct"].idxmax()

        ult_s_raw = float(df.loc[ult_idx, "global_strain"])
        df["Normalized_Strain"] = df["global_strain"] / ult_s_raw if ult_s_raw > 0 else 0.0
        sat_row = df.loc[sat_idx]
        ult_row = df.loc[ult_idx]

        out_f = self.out_dir / f"{mat_path.stem}_Origin_Plot_Data.xlsx"
        with pd.ExcelWriter(out_f, engine="openpyxl") as writer:
            fig1_cols = [
                "Strain_pct",
                "crack_count",
                "crack_spacing_mm",
                "W_avg_um",
                "W_99_um",
                "W_max_um",
                "quality_valid_fraction",
            ]
            df[fig1_cols].to_excel(writer, sheet_name="Fig1_Dynamics", index=False)
            df[["Normalized_Strain", "crack_count", "W_avg_um", "W_max_um"]].to_excel(
                writer, sheet_name="Fig2_Normalized", index=False
            )

            p_sat = self._details_for_frame(results, sat_row["Frame"])
            p_ult = self._details_for_frame(results, ult_row["Frame"])
            dist_dict = {
                "Saturated_um": pd.Series(p_sat["W_avg_mm"].values * 1000.0) if not p_sat.empty else pd.Series(dtype=float),
                "Ultimate_um": pd.Series(p_ult["W_avg_mm"].values * 1000.0) if not p_ult.empty else pd.Series(dtype=float),
            }
            pd.DataFrame(dist_dict).to_excel(writer, sheet_name="Fig3_Distribution", index=False)

            grad_data = {}
            for ts in self.config.get("export", {}).get("target_strains", [0.2, 2.0, 4.0, 6.0]):
                if ts <= df["Strain_pct"].max():
                    idx = (df["Strain_pct"] - ts).abs().idxmin()
                    p_d = self._details_for_frame(results, df.loc[idx, "Frame"])
                    if not p_d.empty:
                        grad_data[f"Strain_{df.loc[idx, 'Strain_pct']:.2f}%_um"] = p_d["W_avg_mm"] * 1000.0
            if not grad_data:
                grad_data = {"Notice": ["No cracks reached target strains"]}
            pd.DataFrame({k: pd.Series(v) for k, v in grad_data.items()}).to_excel(
                writer, sheet_name="Fig4_Gradient", index=False
            )

        stat_f = self.out_dir / f"{mat_path.stem}_Statistics_Report.xlsx"
        with pd.ExcelWriter(stat_f, engine="openpyxl") as writer:
            has_stress = "Stress_MPa" in df.columns and not df["Stress_MPa"].isna().all()
            summary_dict = {
                "Specimen": [mat_path.stem],
                "UTS_Stress_MPa": [float(ult_row.get("Stress_MPa", np.nan)) if has_stress else np.nan],
                "Ultimate_Strain_pct": [float(ult_row["Strain_pct"])],
                "Saturated_Crack_Count": [int(sat_row["crack_count"])],
                "Saturated_Spacing_mm": [float(sat_row["crack_spacing_mm"])],
                "Saturated_W_avg_um": [float(sat_row["W_avg_um"])],
                "Ultimate_W_99_um": [float(ult_row["W_99_um"])],
                "Ultimate_W_max_um": [float(ult_row["W_max_um"])],
                "Pixel_Size_mm_per_px": [float(df["pixel_size_mm"].dropna().iloc[0]) if "pixel_size_mm" in df else np.nan],
                "Subset_Spacing_px": [float(df["subset_spacing_px"].dropna().iloc[0]) if "subset_spacing_px" in df else np.nan],
                "DIC_Point_Spacing_mm": [
                    float(df["dic_point_spacing_mm"].dropna().iloc[0]) if "dic_point_spacing_mm" in df else np.nan
                ],
                "Strain_Source": [str(ult_row.get("strain_source", "unknown"))],
                "Sync_Status": [str(ult_row.get("sync_status", "unknown"))],
            }
            pd.DataFrame(summary_dict).to_excel(writer, sheet_name="01_Macro_Summary", index=False)

            grad_rows = []
            for ts in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                if ts <= df["Strain_pct"].max():
                    idx = (df["Strain_pct"] - ts).abs().idxmin()
                    grad_rows.append(
                        {
                            "Target_Strain_pct": ts,
                            "Real_Strain_pct": float(df.loc[idx, "Strain_pct"]),
                            "Crack_Count": int(df.loc[idx, "crack_count"]),
                            "Spacing_mm": float(df.loc[idx, "crack_spacing_mm"]),
                            "W_avg_um": float(df.loc[idx, "W_avg_um"]),
                            "W_99_um": float(df.loc[idx, "W_99_um"]),
                            "Quality_Valid_Fraction": float(df.loc[idx, "quality_valid_fraction"]),
                        }
                    )
            if not grad_rows:
                grad_rows = [{"Notice": "No data reached target strains"}]
            pd.DataFrame(grad_rows).to_excel(writer, sheet_name="02_Gradient_States", index=False)

            self._export_crack_details(writer, p_sat, "03_Saturated_Cracks")
            self._export_crack_details(writer, p_ult, "04_Ultimate_Cracks")
            self._qa_frame(df).to_excel(writer, sheet_name="05_QA_Metadata", index=False)
            self._validation_frame(mat_path, df).to_excel(writer, sheet_name="06_Validation", index=False)

        for r in results:
            r.pop("per_crack_details", None)
            r.pop("raw_widths", None)

        self.log_emitted.emit(f"Statistics report generated: {stat_f.name}")
        self.specimen_processed.emit(str(out_f), str(stat_f))

    @staticmethod
    def _details_for_frame(results: list[Dict[str, Any]], frame: Any) -> pd.DataFrame:
        for r in results:
            if int(r["Frame"]) == int(frame):
                return r.get("per_crack_details", pd.DataFrame())
        return pd.DataFrame()

    @staticmethod
    def _export_crack_details(writer: pd.ExcelWriter, p_df: pd.DataFrame, sheet_name: str) -> None:
        if not p_df.empty:
            out_df = p_df.copy()
            out_df["W_avg_um"] = out_df["W_avg_mm"] * 1000.0
            out_df["W_max_um"] = out_df["W_max_mm"] * 1000.0
            keep = ["Crack_ID", "Length_mm", "W_avg_um", "W_max_um", "count"]
            out_df[[c for c in keep if c in out_df.columns]].sort_values(
                "W_avg_um", ascending=False
            ).to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            pd.DataFrame({"Notice": ["No Cracks Detected"]}).to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def _qa_frame(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for key in [
            "pixel_size_mm",
            "subset_spacing_px",
            "dic_point_spacing_mm",
            "metadata_source",
            "v_map_present",
            "quality_map_present",
            "quality_filter",
            "quality_valid_fraction",
            "cod_vector_mode",
            "cod_status",
            "sync_status",
            "strain_source",
        ]:
            if key in df.columns:
                vals = df[key].dropna()
                rows.append(
                    {
                        "Item": key,
                        "First": vals.iloc[0] if not vals.empty else np.nan,
                        "Min": vals.min() if not vals.empty and pd.api.types.is_numeric_dtype(vals) else np.nan,
                        "Max": vals.max() if not vals.empty and pd.api.types.is_numeric_dtype(vals) else np.nan,
                        "Unique_Count": int(vals.astype(str).nunique()) if not vals.empty else 0,
                    }
                )
        return pd.DataFrame(rows)

    def _validation_frame(self, mat_path: Path, df: pd.DataFrame) -> pd.DataFrame:
        validation = self.config.get("validation", {})
        candidates = []
        if validation.get("annotation_path"):
            candidates.append(Path(validation["annotation_path"]))
        candidates.extend(
            [
                mat_path.with_name(f"{mat_path.stem}_annotations.csv"),
                mat_path.with_name(f"{mat_path.stem}_annotation.csv"),
                mat_path.with_name(f"{mat_path.stem}_annotations.xlsx"),
            ]
        )
        ann_path = next((p for p in candidates if p.exists()), None)
        if ann_path is None:
            return pd.DataFrame(
                {
                    "Status": ["No annotation file found"],
                    "Expected": ["<specimen>_annotations.csv/xlsx with Frame and optional crack_count/W_avg_um/W_max_um"],
                }
            )

        ann = pd.read_excel(ann_path) if ann_path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(ann_path)
        if "Frame" not in ann.columns:
            return pd.DataFrame({"Status": [f"Annotation file has no Frame column: {ann_path.name}"]})
        merged = pd.merge(df, ann, on="Frame", suffixes=("_calc", "_manual"))
        if merged.empty:
            return pd.DataFrame({"Status": [f"No matching frames with annotation file: {ann_path.name}"]})
        rows = [{"Status": "OK", "Annotation_File": str(ann_path), "Matched_Frames": int(len(merged))}]
        for metric in ("crack_count", "W_avg_um", "W_max_um"):
            calc_col = f"{metric}_calc"
            manual_col = f"{metric}_manual"
            if calc_col in merged.columns and manual_col in merged.columns:
                err = pd.to_numeric(merged[calc_col], errors="coerce") - pd.to_numeric(
                    merged[manual_col], errors="coerce"
                )
                rows.append(
                    {
                        "Metric": metric,
                        "MAE": float(np.nanmean(np.abs(err))),
                        "Bias": float(np.nanmean(err)),
                        "MaxAbsError": float(np.nanmax(np.abs(err))),
                    }
                )
        return pd.DataFrame(rows)

    def stop(self) -> None:
        self._is_running = False
