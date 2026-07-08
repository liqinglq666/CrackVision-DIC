from pathlib import Path

import numpy as np
import pandas as pd

from src.core.evolution_analyzer import EvolutionAnalyzer
from src.core.io_sync import PipelineIO
from src.core.physics import CrackPhysicsEngine


def test_cod_uses_vector_projection_with_vertical_displacement():
    config = {
        "physics": {
            "cod_min_mm": 0.005,
            "cod_min_mean_mm": 0.005,
            "cod_max_mm": 1.0,
            "min_crack_length_mm": 0.1,
            "cod_sampling": {"delta_px": 3, "max_search_px": 4},
        }
    }
    engine = CrackPhysicsEngine(config)

    u = np.zeros((30, 30), dtype=float)
    v = np.zeros((30, 30), dtype=float)
    v[13:, :] = 1.0

    skeleton = np.zeros((30, 30), dtype=bool)
    skeleton[10, 5:25] = True

    result = engine.compute_cod(
        u,
        skeleton,
        displacement_scale_mm=0.01,
        dic_point_spacing_mm=0.01,
        v_map=v,
    )

    assert result["crack_count"] == 1
    assert np.isclose(result["w_max"], 0.01)
    assert np.isclose(result["w_avg"], 0.01)


def test_mts_sync_computes_stress_and_global_strain(tmp_path: Path):
    mts_path = tmp_path / "mts.csv"
    mts_path.write_text(
        "Time,Load,Displacement\n"
        "sec,N,mm\n"
        "0,0,0\n"
        "5,1000,0.4\n"
        "10,2000,0.8\n",
        encoding="utf-8",
    )
    df_dic = pd.DataFrame({"Time_s": [0.0, 2.5, 10.0]})
    config = {"experiment": {"cross_section_area_mm2": 100.0, "gauge_length_mm": 80.0}}

    synced = EvolutionAnalyzer(config, mts_path).synchronize(df_dic)

    assert np.allclose(synced["Stress_MPa"], [0.0, 5.0, 20.0])
    assert np.allclose(synced["MTS_Strain"], [0.0, 0.0025, 0.01])
    assert "global_strain" not in synced.columns
    assert (synced["strain_source"] == "dic_virtual_extensometer").all()


def test_mts_sync_can_override_dic_global_strain(tmp_path: Path):
    mts_path = tmp_path / "mts.csv"
    mts_path.write_text(
        "Time,Load,Displacement\n"
        "sec,N,mm\n"
        "0,0,0\n"
        "5,1000,0.4\n"
        "10,2000,0.8\n",
        encoding="utf-8",
    )
    df_dic = pd.DataFrame({"Time_s": [0.0, 2.5, 10.0], "global_strain": [0.0, 0.0, 0.0]})
    config = {
        "experiment": {"cross_section_area_mm2": 100.0, "gauge_length_mm": 80.0},
        "sync": {"override_dic_strain_with_mts": True},
    }

    synced = EvolutionAnalyzer(config, mts_path).synchronize(df_dic)

    assert np.allclose(synced["global_strain"], [0.0, 0.0025, 0.01])
    assert (synced["strain_source"] == "mts_displacement").all()


def test_ncorr_formatted_hdf5_field_names_are_recognized():
    strain_keys = [
        "plot_exx_cur_formatted",
        "plot_exx_ref_formatted",
        "plot_exy_cur_formatted",
        "plot_eyy_ref_formatted",
    ]
    displacement_keys = [
        "plot_corrcoef_dic",
        "plot_u_cur_formatted",
        "plot_u_dic",
        "plot_v_cur_formatted",
        "plot_v_dic",
    ]

    assert PipelineIO._pick_field(strain_keys, PipelineIO.EXX_KEYS, ("plot_exx", "exx")) == "plot_exx_ref_formatted"
    assert PipelineIO._pick_field(displacement_keys, PipelineIO.U_KEYS, ("plot_u", "disp_u")) == "plot_u_dic"
    assert PipelineIO._pick_field(displacement_keys, PipelineIO.V_KEYS, ("plot_v", "disp_v")) == "plot_v_dic"
    assert (
        PipelineIO._pick_field(displacement_keys + strain_keys, PipelineIO.QUALITY_KEYS, PipelineIO.QUALITY_KEYS)
        == "plot_corrcoef_dic"
    )
