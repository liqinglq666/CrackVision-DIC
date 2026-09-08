# CrackVision-DIC 3.0

CrackVision-DIC is a deliberately small post-processing tool for **ECC / SHCC tensile-test DIC data exported by Ncorr**. Version 3 removes the feature pile-up and keeps only the scientific chain needed to obtain auditable crack widths.

## What remains

```text
Ncorr .mat
  ├─ U, V displacement
  ├─ Exx, Eyy, Exy strain
  ├─ pixtounits
  └─ spacing
       ↓
maximum principal tensile strain ε1
       ↓
crack candidate + skeleton
       ↓
local crack normal
       ↓
multiple displacement samples on both crack faces
       ↓
robust linear fit on each face
       ↓
extrapolate both fits to the crack plane
       ↓
COD = normal displacement discontinuity
       ↓
one Excel workbook + QA status
```

Removed on purpose: MTS synchronisation, camera crack masks, image width estimates, multi-report exports, batch summary dashboards, multiprocessing/temp-file plumbing, compatibility wrapper modules and secondary statistics code. They made failure modes harder to audit without improving the primary COD measurement.

## Why the crack-width method changed

The old implementation detected cracks from `Exx` only. That is orientation dependent. Version 3 reads `Exx`, `Eyy` and `Exy` and calculates the maximum principal tensile strain

```text
ε1 = (Exx + Eyy)/2 + sqrt(((Exx - Eyy)/2)^2 + Exy^2)
```

for Ncorr tensor shear strain. If your imported file stores engineering shear, set `shear_component_is_engineering: true` in `config/default.yaml`.

Crack width is not estimated from the visible strain-band thickness. For each skeleton point the program determines a local normal and samples U/V on both sides. Each side is fitted as a local linear displacement field and extrapolated to the crack plane. The normal intercept discontinuity is COD. This suppresses the continuous elastic displacement gradient that contaminates a simple two-point subtraction.

## Important scale fix

Ncorr native `spacing` is treated as a skipped-pixel count, so the subset-centre step is

```text
DIC step px = spacing + 1
DIC grid spacing mm = (spacing + 1) × pixtounits
```

Set `ncorr_spacing_is_gap_count: false` only for non-native files whose spacing value already is a centre-to-centre step.

Two scales remain separate:

```text
U/V displacement px × pixel_size_mm       -> displacement mm
DIC grid index × dic_point_spacing_mm      -> geometry/search distance mm
```

## Failure semantics

A failed measurement is **NaN**, never zero. The `cod_status` column explains why, for example:

```text
ok
no_crack_candidate
insufficient_cod_samples
crack_filter_removed_all
```

Zero therefore means an actual zero only if a valid measurement produces it. A missing/invalid COD is not silently turned into `0 μm`.

## Input contract

The Ncorr MAT file must contain, frame by frame:

```text
plot_u_*      or u
plot_v_*      or v
plot_exx_*    or exx
plot_eyy_*    or eyy
plot_exy_*    or exy
```

Both classic MATLAB MAT files and common v7.3/HDF5 Ncorr layouts are supported. If `Eyy`, `Exy` or `V` is missing, the program stops with a clear field error instead of falling back to a physically incomplete width.

## Output

Each specimen creates one workbook:

```text
<specimen>_CrackVision.xlsx
├─ 00_READ_ME
├─ 01_Frame_Summary
├─ 02_Crack_Details
└─ 03_QA
```

Recommended paper-facing metrics:

```text
W_median_um   primary robust crack width
W_avg_um      mean width
W_95_um       upper distribution metric
W_max_um      diagnostic/extreme value
```

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

Run tests:

```bash
python -m pytest -q
```

## Configuration

`config/default.yaml` contains only parameters that materially affect the core measurement: fallback scale, Ncorr spacing semantics, principal-strain threshold, crack morphology and physical COD sampling distances.

For formal analysis, prefer physical sampling distances in millimetres (`near_mm`, `far_mm`) rather than arbitrary DIC-point counts.
