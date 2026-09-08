from __future__ import annotations

import numpy as np

from src.core.export import prepare_frame_summary
from src.core.io_ncorr import NcorrLoader
from src.core.models import FrameData
from src.core.physics import CrackPhysicsEngine


def make_frame(*, horizontal: bool, jump_px: float, slope_px_per_index: float = 0.0) -> FrameData:
    h = w = 61
    u = np.zeros((h, w), dtype=float)
    v = np.zeros((h, w), dtype=float)
    exx = np.zeros((h, w), dtype=float)
    eyy = np.zeros((h, w), dtype=float)
    exy = np.zeros((h, w), dtype=float)
    center = 30

    if horizontal:
        yy = np.arange(h)[:, None]
        v[:] = slope_px_per_index * (yy - center)
        v[center + 1 :, :] += jump_px
        eyy[center - 1 : center + 2, 8:-8] = 0.02
    else:
        xx = np.arange(w)[None, :]
        u[:] = slope_px_per_index * (xx - center)
        u[:, center + 1 :] += jump_px
        exx[8:-8, center - 1 : center + 2] = 0.02

    mask = np.ones((h, w), dtype=bool)
    return FrameData(
        frame_id=0,
        u_map=u,
        v_map=v,
        exx_map=exx,
        eyy_map=eyy,
        exy_map=exy,
        mask=mask,
        pixel_size_mm=0.05,
        dic_point_spacing_mm=0.10,
        dic_step_px=2.0,
        metadata_source="synthetic",
    )


def engine() -> CrackPhysicsEngine:
    return CrackPhysicsEngine(
        {
            "physics": {
                "detection": {
                    "threshold_k": 1.0,
                    "min_tensile_strain": 1e-5,
                    "min_crack_area_points": 3,
                    "min_crack_length_mm": 0.5,
                    "normal_radius_points": 4,
                },
                "cod": {
                    "near_mm": 0.2,
                    "far_mm": 0.8,
                    "samples_per_side": 7,
                    "min_valid_per_side": 3,
                    "min_samples_per_crack": 3,
                    "min_width_mm": 0.0,
                    "max_width_mm": 1.0,
                },
            }
        }
    )


def test_principal_strain_is_coordinate_invariant_for_simple_cases():
    eng = engine()
    exx = np.array([[0.01, 0.0]])
    eyy = np.array([[0.0, 0.02]])
    exy = np.zeros_like(exx)
    eps1 = eng.maximum_principal_tensile_strain(exx, eyy, exy)
    assert np.allclose(eps1, [[0.01, 0.02]])


def test_horizontal_crack_uses_vertical_displacement_jump():
    summary, details = engine().analyze_frame(make_frame(horizontal=True, jump_px=2.0))
    assert summary["cod_status"] == "ok"
    assert summary["crack_count"] == 1
    assert np.isclose(summary["W_median_mm"], 0.10, atol=2e-3)
    assert np.isclose(details["W_median_mm"].iloc[0], 0.10, atol=2e-3)


def test_vertical_crack_uses_horizontal_displacement_jump():
    summary, _ = engine().analyze_frame(make_frame(horizontal=False, jump_px=1.5))
    assert summary["cod_status"] == "ok"
    assert np.isclose(summary["W_median_mm"], 0.075, atol=2e-3)


def test_regression_removes_continuous_background_gradient():
    # 0.04 px/index continuous deformation is present on both sides; the true discontinuity remains 2 px.
    summary, _ = engine().analyze_frame(make_frame(horizontal=True, jump_px=2.0, slope_px_per_index=0.04))
    assert summary["cod_status"] == "ok"
    assert np.isclose(summary["W_median_mm"], 0.10, atol=3e-3)


def test_native_ncorr_spacing_uses_plus_one_step():
    assert NcorrLoader.spacing_to_step_px(2.0, ncorr_spacing_is_gap_count=True) == 3.0
    assert NcorrLoader.spacing_to_step_px(3.0, ncorr_spacing_is_gap_count=False) == 3.0


def test_failed_cod_remains_nan_in_export_table():
    df = prepare_frame_summary(
        [
            {
                "Frame": 0,
                "cod_status": "insufficient_cod_samples",
                "W_median_mm": np.nan,
                "W_avg_mm": np.nan,
                "W_95_mm": np.nan,
                "W_max_mm": np.nan,
            }
        ]
    )
    assert np.isnan(df.loc[0, "W_median_um"])
    assert np.isnan(df.loc[0, "W_max_um"])


def test_classic_ncorr_mat_loader_reads_full_tensor_and_scale(tmp_path):
    from scipy.io import savemat

    shape = (11, 13)
    strains = np.empty(2, dtype=object)
    displacements = np.empty(2, dtype=object)
    for i in range(2):
        strains[i] = {
            "plot_exx_ref_formatted": np.ones(shape) * 0.001 * i,
            "plot_eyy_ref_formatted": np.ones(shape) * 0.002 * i,
            "plot_exy_ref_formatted": np.zeros(shape),
        }
        displacements[i] = {
            "plot_u_dic": np.ones(shape) * i,
            "plot_v_dic": np.ones(shape) * 2 * i,
        }

    mat_path = tmp_path / "synthetic_ncorr.mat"
    savemat(
        mat_path,
        {
            "data_dic_save": {
                "strains": strains,
                "displacements": displacements,
                "dispinfo": {"pixtounits": 0.05, "spacing": 2.0},
            }
        },
    )

    frames = list(
        NcorrLoader.stream_frames(
            mat_path,
            fallback_ratio=0.1,
            config={"experiment": {"ncorr_spacing_is_gap_count": True}},
        )
    )
    assert len(frames) == 2
    assert frames[0].u_map.shape == shape
    assert np.isclose(frames[0].pixel_size_mm, 0.05)
    assert np.isclose(frames[0].dic_step_px, 3.0)
    assert np.isclose(frames[0].dic_point_spacing_mm, 0.15)
    assert frames[0].metadata_source == "mat_pixtounits;mat_ncorr_spacing_plus_one"
    assert np.isclose(frames[1].u_map[0, 0], 1.0)
    assert np.isclose(frames[1].v_map[0, 0], 2.0)
