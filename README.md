# meepIntro

Structured introduction to electromagnetic metasurface simulation and optimisation.

**Stack:** MEEP (FDTD) → RCWA solvers → GPU-accelerated optimisation (ROCm/PyTorch)
**Target hardware:** Any CPU (laptop/WSL) + AMD Ryzen AI Max 395 (ROCm, Phase 3)
**Wavelength regime:** UV-Vis (400–700 nm), 2D first → 3D

---

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 1 | MEEP fundamentals — 4 metasurface examples | 🔄 in progress |
| 2 | Speed up base sims (resolution, symmetry, MPI) | planned |
| 3 | Solver comparison: FDTD vs RCWA vs torcwa | planned |
| 4 | Surrogate-model optimisation (GPU inference) | planned |

---

## Phase 1 Examples

| Folder | Physics | Key MEEP concepts |
|--------|---------|-------------------|
| `01_beam_steering/` | Phase-gradient metasurface | Bloch BCs, phase library, angular spectrum |
| `02_metalens/` | Focusing / metalens | Near-to-far field, focal spot |
| `03_holography/` | Farfield hologram | Complex amplitude encoding |
| `04_absorption/` | Resonant absorber / filter | Transmission spectra, harminv |

---

## Quick Start

### 1 — Install environment

```bash
conda env create -f envs/meep_env.yml
conda activate meep
```

### 2 — Run a simulation

All simulations go through `run.py`, which auto-detects CPU cores and
launches under MPI if available:

```bash
# Build phase library first (run once, ~20–60 min depending on resolution)
python run.py 01_beam_steering/unit_cell_sweep.py

# Then run the full metasurface simulation
python run.py 01_beam_steering/full_array_sim.py

# Override parameters via CLI
python run.py 01_beam_steering/full_array_sim.py --angle 45 --wavelength 0.633
python run.py 01_beam_steering/full_array_sim.py --help

# Force fewer cores (e.g. for testing)
MEEP_NPROCS=4 python run.py 01_beam_steering/unit_cell_sweep.py --n-widths 5 --resolution 32
```

### 3 — Edit simulation parameters

Open `01_beam_steering/full_array_sim.py` and edit the block at the top:

```python
WAVELENGTH  = 0.532   # μm  — free-space wavelength
ANGLE       = 30.0    # °   — target steering angle
MS_WIDTH    = 5.0     # μm  — metasurface lateral width
PILLAR_H    = 0.60    # μm  — pillar height
RESOLUTION  = 32      #     — pixels/μm (32=fast, 64=accurate)
```

---

## Materials

Bundled n,k data files live in `materials/`.
Pass `--material TiO2` (default) or `--material path/to/myfile.txt`.

| Name | File | Source |
|------|------|--------|
| `TiO2` | `TiO2_rutile_Siefke2016.txt` | Siefke et al., Adv. Opt. Mat. 2016 |
| `SiO2` / `glass` | `SiO2_Malitson1965.txt` | Malitson, JOSA 1965 |

**Adding your own material:**
Download from [refractiveindex.info](https://refractiveindex.info) (export as
space-separated), then:

```bash
python run.py 01_beam_steering/unit_cell_sweep.py --material path/to/my_material.txt
```

File format (any number of `#` comment lines, then data):
```
# wavelength_um   n      k
0.400             2.78   0.001
0.500             2.58   0.0
...
```

---

## Repository Structure

```
meepIntro/
├── run.py                          ← MPI launcher (start here)
├── envs/
│   └── meep_env.yml                ← conda environment
├── materials/
│   ├── TiO2_rutile_Siefke2016.txt
│   └── SiO2_Malitson1965.txt
├── utils/
│   ├── device.py                   ← CPU/GPU detection
│   ├── materials.py                ← n,k loader → MEEP Medium
│   └── viz.py                      ← shared plotting helpers
├── 01_beam_steering/               ← see 01_beam_steering/README.md
│   ├── unit_cell_sweep.py          ← step 1: build phase library
│   ├── full_array_sim.py           ← step 2: full metasurface FDTD
│   └── results/                    ← auto-created: plots + data
├── 02_metalens/                    ← (coming next)
├── 03_holography/
└── 04_absorption/
```

---

## Utils API

### `utils/materials.py` — Material loader

```python
from utils.materials import load_material

mat = load_material("TiO2")               # bundled short name
mat = load_material("path/to/my_nk.txt")  # custom file

n   = mat.n(0.532)           # interpolated refractive index
k   = mat.k(0.532)           # interpolated extinction coefficient
med = mat(0.532)             # → mp.Medium at λ=532 nm (shorthand)
```

Reads space-separated n,k files (compatible with refractiveindex.info exports).
Interpolates linearly; warns if the requested wavelength is outside the
tabulated range. For lossy materials (k > 0) it converts to MEEP's
`D_conductivity` representation.

Bundled names: `"TiO2"`, `"SiO2"`, `"glass"`.

---

### `utils/device.py` — Compute resource detection

```python
from utils.device import get_cpu_count, get_torch_device, print_platform_info

ncpu = get_cpu_count()          # logical CPU count
dev  = get_torch_device()       # torch.device or None
print_platform_info()           # prints CPU / MPI / GPU summary
```

`get_torch_device()` returns `cuda` for both CUDA and ROCm builds (PyTorch
exposes ROCm as `cuda`). Returns `None` if PyTorch is not installed — safe
to call in Phase 1/2 where PyTorch is not required.

---

### `utils/viz.py` — Plotting helpers

All functions save PNG files to disk (Matplotlib `Agg` backend — no display
needed, works in WSL/headless).

| Function | Output |
|----------|--------|
| `plot_phase_library(widths, phases, amplitudes, ...)` | 4-panel phase/amplitude sweep plot |
| `plot_angular_spectrum(theta, intensity, ...)` | Linear + log farfield plot |
| `plot_fields(sim, component, filename)` | 2D MEEP near-field snapshot |
| `plot_epsilon(sim, filename)` | Permittivity cross-section |
| `plot_pillar_layout(x_positions, widths, ...)` | Pillar geometry + phase profile |

Output directories are created automatically.

---

## ROCm / GPU Notes

MEEP itself is CPU-only (MPI parallelism). GPU acceleration applies to:
- Phase 3: `torcwa` RCWA solver (PyTorch backend)
- Phase 4: surrogate NN training and inference

To enable ROCm PyTorch after installing the conda env:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

The code is device-agnostic — `utils/device.py` auto-detects and falls back
to CPU if ROCm is unavailable (e.g. iGPU in WSL).
