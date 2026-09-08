# CrackVision-DIC

CrackVision-DIC is a focused post-processing tool for **ECC / SHCC tensile-test DIC data calculated by Ncorr**. The current workflow is intentionally narrow: use the **MTS peak tensile-stress time** to select the nearest DIC frame, then calculate crack width only for that state.

```text
MTS CSV (force + time)
        ↓
peak tensile force time
        ↓
nearest Ncorr DIC frame
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
COD crack width
        ↓
Excel + QA
```

For a constant specimen cross-section, peak tensile stress and peak tensile force occur at the same instant. Therefore **cross-sectional area is not required to select the peak-stress DIC frame**.

## MTS CSV

The parser is designed for the MTS/DAQ export used in this project, including files with metadata rows before the actual table. It searches the file for the force/load and time columns, for example:

```text
横梁 | 力 | 时间 | 引伸计 | ...
mm   | N  | sec  | mm     | ...
```

Both positive-tension and negative-tension sign conventions are supported.

## DIC/MTS time synchronization

The GUI asks for:

```text
DIC 第0帧对应 MTS 时间 (s)
```

If the camera/DIC sequence and MTS recording start together, keep this at `0 s`.

If DIC frame 0 was captured 3.2 s after MTS started, enter:

```text
3.2 s
```

Then the target DIC-relative time is:

```text
MTS peak time - DIC frame0 MTS time
```

The software selects the nearest available DIC frame and exports the time mismatch as `frame_match_error_s`.

## Recommended Ncorr input

For new tests, after Ncorr has completed **Format Displacements**, physical unit conversion and **Calculate Strains**, export the compact HDF5 bridge:

```matlab
handles_ncorr = ncorr;
% ... finish the normal Ncorr workflow ...

addpath('path/to/CrackVision-DIC/matlab')
export_ncorr_to_crackvision( ...
    handles_ncorr, ...
    'Specimen01_CrackVision.h5', ...
    5);   % 5 s/image
```

The bridge stores only:

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

Existing saved Ncorr `.mat` files remain supported as a compatibility input. H5 is preferred because the program can **random-read only the selected peak-stress frame** instead of traversing the whole result set.

## Crack-width method

Crack location is detected from maximum principal tensile strain:

```text
ε1 = (Exx + Eyy)/2 + sqrt(((Exx - Eyy)/2)^2 + Exy^2)
```

For each crack-skeleton point, CrackVision estimates the local crack normal, samples U/V on both sides, fits the local displacement field on each crack face and extrapolates both fits to the crack plane. COD is the normal displacement discontinuity.

Failed measurements remain `NaN`; they are never converted to fake `0 μm` values.

## Output

Each specimen creates one workbook:

```text
<specimen>_CrackVision.xlsx
├─ 00_READ_ME
├─ 01_Frame_Summary     # exactly one selected peak-stress frame
├─ 02_Crack_Details     # all accepted cracks in that frame
└─ 03_QA                # MTS/DIC matching + measurement QA
```

Important columns include:

```text
MTS_peak_force_N
MTS_peak_time_s
DIC_selected_time_s
frame_match_error_s
crack_count
W_median_um
W_avg_um
W_95_um
W_max_um
cod_status
```

## Run

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

The GUI requires one specimen pair per run:

```text
1 × CrackVision-Ncorr H5 / original Ncorr MAT
1 × matching MTS CSV
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project structure

```text
CrackVision-DIC/
├─ main.py
├─ config/default.yaml
├─ matlab/export_ncorr_to_crackvision.m
├─ src/
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ mts.py           # MTS CSV parser + peak tensile time
│  │  ├─ input.py         # target-frame selection
│  │  ├─ io_bridge.py     # compact H5 random reader
│  │  ├─ io_ncorr.py      # legacy Ncorr MAT compatibility
│  │  ├─ models.py
│  │  ├─ physics.py
│  │  ├─ pipeline.py      # MTS peak → selected frame → COD
│  │  └─ export.py
│  └─ gui/
│     ├─ main_window.py
│     └─ worker.py
└─ tests/
```

The project intentionally does **not** include camera crack segmentation, image-width estimates, batch dashboards, multiprocessing/temp-file plumbing or multiple competing report systems. MTS is used only for the scientifically necessary task of selecting the peak tensile-stress state; it does not participate in the COD calculation itself.
