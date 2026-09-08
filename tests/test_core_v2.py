from __future__ import annotations

import numpy as np

from src.core.export import prepare_frame_summary
from src.core.io_ncorr import NcorrLoader
from src.core.models import FrameData
from src.core.physics import CrackPhysicsEngine


def make_frame(*, horizontal: bool, jump_px: float, slope_px_per_index: float = 0.0) -> FrameData:
    h = w = 61
    u=np.zeros((h,w)); v=np.zeros((h,w)); exx=np.zeros((h,w)); eyy=np.zeros((h,w)); exy=np.zeros((h,w)); center=30
    if horizontal:
        yy=np.arange(h)[:,None]; v[:]=slope_px_per_index*(yy-center); v[center+1:,:]+=jump_px; eyy[center-1:center+2,8:-8]=0.02
    else:
        xx=np.arange(w)[None,:]; u[:]=slope_px_per_index*(xx-center); u[:,center+1:]+=jump_px; exx[8:-8,center-1:center+2]=0.02
    mask=np.ones((h,w),dtype=bool)
    return FrameData(0,u,v,exx,eyy,exy,mask,0.05,0.10,float("nan"),"synthetic",None,2.0)


def engine()->CrackPhysicsEngine:
    return CrackPhysicsEngine({"physics":{"detection":{"threshold_k":1.0,"min_tensile_strain":1e-5,"min_crack_area_points":3,"min_crack_length_mm":0.5,"normal_radius_points":4},"cod":{"near_mm":0.2,"far_mm":0.8,"samples_per_side":7,"min_valid_per_side":3,"min_samples_per_crack":3,"min_width_mm":0.0,"max_width_mm":1.0}}})


def test_principal_strain_is_coordinate_invariant_for_simple_cases():
    eng=engine(); exx=np.array([[0.01,0.0]]); eyy=np.array([[0.0,0.02]]); exy=np.zeros_like(exx); eps1=eng.maximum_principal_tensile_strain(exx,eyy,exy); assert np.allclose(eps1,[[0.01,0.02]])

def test_horizontal_crack_uses_vertical_displacement_jump():
    summary,details=engine().analyze_frame(make_frame(horizontal=True,jump_px=2.0)); assert summary["cod_status"]=="ok"; assert summary["crack_count"]==1; assert np.isclose(summary["W_median_mm"],0.10,atol=2e-3); assert np.isclose(details["W_median_mm"].iloc[0],0.10,atol=2e-3)

def test_vertical_crack_uses_horizontal_displacement_jump():
    summary,_=engine().analyze_frame(make_frame(horizontal=False,jump_px=1.5)); assert summary["cod_status"]=="ok"; assert np.isclose(summary["W_median_mm"],0.075,atol=2e-3)

def test_regression_removes_continuous_background_gradient():
    summary,_=engine().analyze_frame(make_frame(horizontal=True,jump_px=2.0,slope_px_per_index=0.04)); assert summary["cod_status"]=="ok"; assert np.isclose(summary["W_median_mm"],0.10,atol=3e-3)

def test_native_ncorr_spacing_uses_plus_one_step():
    assert NcorrLoader.spacing_to_step_px(2.0,ncorr_spacing_is_gap_count=True)==3.0; assert NcorrLoader.spacing_to_step_px(3.0,ncorr_spacing_is_gap_count=False)==3.0

def test_failed_cod_remains_nan_in_export_table():
    df=prepare_frame_summary([{"Frame":0,"cod_status":"insufficient_cod_samples","W_median_mm":np.nan,"W_avg_mm":np.nan,"W_95_mm":np.nan,"W_max_mm":np.nan}]); assert np.isnan(df.loc[0,"W_median_um"]); assert np.isnan(df.loc[0,"W_max_um"])
