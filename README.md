# CrackVision-DIC

CrackVision-DIC is a focused post-processing tool for **ECC / SHCC tensile-test DIC data calculated by Ncorr**.

The project intentionally does one job: convert Ncorr displacement/strain fields into auditable crack-width statistics.

```text
Ncorr
  ↓
U / V / Exx / Eyy / Exy
  ↓
maximum principal tensile strain
  ↓
crack candidate + skeleton
  ↓
local crack normal
  ↓
multiple U/V samples on both crack faces
  ↓
robust linear fits extrapolated to the crack plane
  ↓
COD
  ↓
Excel + QA
```

## Recommended input

For new tests, do not save a huge Ncorr MAT only for CrackVision. After Ncorr has completed **Format Displacements**, physical unit conversion and **Calculate Strains**, export the compact bridge directly:

```matlab
handles_ncorr = ncorr;
% ... finish the normal Ncorr workflow ...

addpath('path/to/CrackVision-DIC/matlab')
export_ncorr_to_crackvision( ...
    handles_ncorr, ...
    'Specimen01_CrackVision.h5', ...
    5);   % 5 s/image
```

The bridge stores only the fields required by CrackVision:

```text
U, V
Exx, Eyy, Exy
finite mask
time
pixel_size_mm
Ncorr spacing
DIC step
DIC point spacing in mm
```

It uses one reference configuration throughout:

```text
plot_u_ref_formatted
plot_v_ref_formatted
plot_exx_ref_formatted
plot_eyy_ref_formatted
plot_exy_ref_formatted
```

Existing saved Ncorr `.mat` files remain supported as a compatibility input.

## Crack-width method

Crack location is detected from the maximum principal tensile strain:

```text
ε1 = (Exx + Eyy)/2 + sqrt(((Exx - Eyy)/2)^2 + Exy^2)
```

For each crack-skeleton point, CrackVision estimates the local crack normal, samples U/V on both sides, fits the local displacement field on each crack face and extrapolates both fits to the crack plane.

The crack opening displacement is the normal displacement discontinuity. Continuous elastic deformation is therefore separated from the discontinuous crack opening better than with a simple two-point subtraction.

Failed measurements remain `NaN`; they are never converted to fake `0 μm` values.

Common `cod_status` values:

```text
ok
no_crack_candidate
insufficient_cod_samples
crack_filter_removed_all
```

## Scale convention

Ncorr displacement and DIC-grid geometry use different scales:

```text
U/V displacement px × pixel_size_mm
    -> displacement mm

DIC grid index × dic_point_spacing_mm
    -> geometry/search distance mm
```

For native Ncorr spacing, CrackVision uses:

```text
DIC step px = spacing + 1
DIC point spacing mm = DIC step px × pixel_size_mm
```

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
W_median_um   primary robust width
W_avg_um      mean width
W_95_um       upper distribution metric
W_max_um      extreme/diagnostic value
```

## Run

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

The GUI accepts:

```text
*.h5 / *.hdf5    preferred CrackVision-Ncorr bridge
*.mat            original Ncorr data compatibility
```

## Tests

Development dependencies are separated from runtime dependencies:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project structure

```text
CrackVision-DIC/
├─ main.py
├─ config/
│  └─ default.yaml
├─ matlab/
│  └─ export_ncorr_to_crackvision.m
├─ src/
│  ├─ core/
│  │  ├─ config.py        # configuration boundary
│  │  ├─ input.py         # H5/MAT dispatch
│  │  ├─ io_bridge.py     # compact H5 reader
│  │  ├─ io_ncorr.py      # original Ncorr MAT compatibility reader
│  │  ├─ models.py        # strict frame data contract
│  │  ├─ physics.py       # principal strain + crack detection + fitted COD
│  │  ├─ pipeline.py      # complete file-analysis orchestration
│  │  └─ export.py        # Excel + QA
│  └─ gui/
│     ├─ main_window.py   # UI only
│     └─ worker.py        # thin QThread adapter
└─ tests/
```

The architecture rule is simple:

```text
GUI must not know Ncorr file internals.
Worker must not implement scientific calculations.
Readers must not calculate COD.
Physics must not know about Qt or Excel.
```

## Intentionally not included

The project no longer carries unrelated feature layers such as MTS synchronisation, camera crack segmentation, image-width estimates, multiple report systems, batch dashboards, multiprocessing/temp-file plumbing or compatibility wrapper classes.

Those features do not improve the primary Ncorr-COD measurement and make scientific failure modes harder to audit.
