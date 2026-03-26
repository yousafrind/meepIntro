# meepIntro

Structured introduction to electromagnetic metasurface simulation and optimisation.

**Stack:** MEEP (FDTD) → RCWA solvers → GPU-accelerated optimisation (ROCm/PyTorch)
**Target hardware:** Any CPU (laptop/WSL) + AMD Ryzen AI Max 395 (ROCm, Phase 3)
**Wavelength regime:** UV-Vis (400–700 nm), 2D first → 3D

---

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 1 | MEEP fundamentals — 4 metasurface examples | ✅ complete |
| 2 | Speed up base sims (resolution, symmetry, MPI) | ✅ complete |
| 3 | Solver comparison: FDTD vs RCWA vs torcwa | ✅ complete |
| 4 | Surrogate-model optimisation (GPU inference) | ✅ complete |
| 5 | Broadband / achromatic design (multi-λ surrogate) | ✅ complete |

---

## Phase 1 Examples

| Folder | Physics | Key MEEP concepts | Status |
|--------|---------|-------------------|--------|
| `01_beam_steering/` | Phase-gradient metasurface | Bloch BCs, phase library, angular spectrum | ✅ done |
| `02_metalens/` | Focusing / flat metalens | Near-to-far (ASM), focal spot | ✅ done |
| `03_holography/` | Farfield hologram (GS phase retrieval) | Iterative design, farfield FFT | ✅ done |
| `04_absorption/` | Resonant absorber / filter | Broadband DFT flux, harminv Q | ✅ done |

---

## Phase 2 — Benchmarks

```bash
# Full benchmark suite
python run.py benchmarks/benchmark.py

# Quick smoke-test (< 1 min)
python run.py benchmarks/benchmark.py --quick

# Individual benchmarks
python run.py benchmarks/benchmark.py --mode resolution --resolutions 16 32 64
python run.py benchmarks/benchmark.py --mode symmetry
python run.py benchmarks/benchmark.py --mode mpi --max-procs 8
```

| Benchmark | Output |
|-----------|--------|
| Resolution | Wall time vs px/μm + O(r³) reference |
| Symmetry | Mirror(X) vs no-symmetry speedup bar chart |
| MPI | Speedup + parallel efficiency vs nprocs |

Results → `benchmarks/results/` (PNG plot + text report + JSON).

**Mirror(X) symmetry** is now a flag on all unit-cell sweeps:

```bash
python run.py utils/sweep.py --symmetry --resolution 64 \
    --outdir 01_beam_steering/results
```

---

## Phase 3 — Solver Comparison (FDTD vs RCWA)

`05_solver_comparison/` benchmarks MEEP FDTD against torcwa RCWA on the same
unit-cell geometry.  RCWA solves Maxwell's equations analytically for periodic
structures — orders of magnitude faster than FDTD with equivalent accuracy.

```bash
# Step 1: run RCWA sweep (seconds vs hours for MEEP)
MEEP_NPROCS=1 python run.py 05_solver_comparison/rcwa_sim.py

# Step 2: compare results (requires a MEEP library from 01_beam_steering)
python run.py 05_solver_comparison/compare_solvers.py

# Optional: pass MEEP sweep time for speed table
python run.py 05_solver_comparison/compare_solvers.py --meep-time 1820

# Shift RCWA phase curve to match MEEP reference convention
python run.py 05_solver_comparison/compare_solvers.py --align-phase
```

| Script | What it does |
|--------|-------------|
| `rcwa_sim.py` | torcwa RCWA sweep → `rcwa_phase_library.npz` + diagnostic plot |
| `compare_solvers.py` | Overlay |T| and ∠T; print speed table |

Output → `05_solver_comparison/results/`:
- `rcwa_phase_library.npz` — PhaseLibrary-compatible, usable in any Phase 1 sim
- `rcwa_phase_library.png` — 4-panel diagnostic (amplitude + phase)
- `solver_comparison.png`  — MEEP vs RCWA overlay

**Installing torcwa** (not in `meep_env.yml` — CPU-only):
```bash
pip install torcwa
```
GPU (ROCm/CUDA) acceleration is automatic when PyTorch detects a GPU.

---

## Phase 4 — Surrogate-Model Optimisation

`06_surrogate/` trains a differentiable MLP surrogate on phase library data,
then uses gradient descent through the surrogate for fast inverse design of
pillar width profiles — no FDTD or RCWA calls during optimisation.

```bash
# Step 1: train surrogate (uses phase library from 01_beam_steering or RCWA)
python run.py 06_surrogate/train.py \
    --libs 01_beam_steering/results/phase_library.npz

# Or train on RCWA data (faster to generate)
python run.py 06_surrogate/train.py \
    --libs 05_solver_comparison/results/rcwa_phase_library.npz

# Step 2: gradient-based inverse design
python run.py 06_surrogate/optimise.py --target beam --angle 30
python run.py 06_surrogate/optimise.py --target metalens --focal-len 10
```

| Script | What it does |
|--------|-------------|
| `train.py` | MLP: `w/period → (|T|, sin∠T, cos∠T)`. Adam + cosine LR. Saves `surrogate_model.pt` |
| `optimise.py` | Gradient descent on pillar widths; compares to NN-baseline from PhaseLibrary |

Output → `06_surrogate/results/`:
- `surrogate_model.pt` — model weights + normalisation stats
- `training_curves.png` — loss curves + |T| and ∠T parity plots
- `optimised_design.png` — phase profile / widths / amplitudes / convergence

**ROCm/CUDA**: `get_torch_device()` auto-detects GPU. Falls back to CPU silently.

---

## Phase 5 — Broadband / Achromatic Design

`07_broadband/` extends the Phase 4 surrogate to support wavelength as an input,
enabling a single set of pillar widths to satisfy the design target at multiple
wavelengths simultaneously — the defining challenge of achromatic metalens design.

```bash
# Step 1: auto-generate RCWA data for each wavelength + train
MEEP_NPROCS=1 python run.py 07_broadband/multiwl_train.py \
    --wavelengths 0.45 0.532 0.633

# Step 2: achromatic metalens (joint optimisation over all λ)
python run.py 07_broadband/achromatic_design.py --target metalens --focal-len 10

# Or achromatic beam steering
python run.py 07_broadband/achromatic_design.py --target beam --angle 30
```

| Script | What it does |
|--------|-------------|
| `multiwl_train.py` | Generates per-λ RCWA libraries, trains 2-input MLP `[w/period, λ] → (|T|, ∠T)` |
| `achromatic_design.py` | Joint gradient descent over all λ; compares vs per-λ NN baselines |

Output → `07_broadband/results/`:
- `multiwl_model.pt` — multi-wavelength surrogate weights
- `multiwl_training.png` — loss curves + parity plots
- `wl_<NNN>nm_library.npz` — per-wavelength RCWA PhaseLibrary files
- `achromatic_design.png` — per-λ phase error comparison + convergence

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

# -- or use the generic sweep engine with adaptive phase-step sampling --
python run.py utils/sweep.py --phase-step 5 --outdir 01_beam_steering/results

# Then run any metasurface simulation (all reuse the same library)
python run.py 01_beam_steering/full_array_sim.py
python run.py 02_metalens/metalens_sim.py
python run.py 03_holography/hologram_sim.py --target-angles -30 30

# Override parameters via CLI
python run.py 01_beam_steering/full_array_sim.py --angle 45 --wavelength 0.633
python run.py 02_metalens/metalens_sim.py --focal-len 15 --lens-width 8
python run.py 03_holography/hologram_sim.py --target-angles -45 0 45

# 04_absorption — standalone (no phase library needed)
python run.py 04_absorption/absorption_sim.py
python run.py 04_absorption/absorption_sim.py --resolution 64 --period 0.40

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
│   ├── sweep.py                    ← unit-cell sweep engine + PhaseLibrary
│   ├── device.py                   ← CPU/GPU detection
│   ├── materials.py                ← n,k loader → MEEP Medium
│   └── viz.py                      ← shared plotting helpers
├── 01_beam_steering/               ← see 01_beam_steering/README.md
│   ├── unit_cell_sweep.py          ← step 1: build phase library (wraps utils/sweep.py)
│   ├── full_array_sim.py           ← step 2: full metasurface FDTD
│   └── results/                    ← auto-created: plots + data
├── 02_metalens/                    ← see 02_metalens/README.md
│   ├── metalens_sim.py             ← quadratic phase + MEEP + ASM focal analysis
│   └── results/
├── 03_holography/                  ← see 03_holography/README.md
│   ├── hologram_sim.py             ← GS phase retrieval + MEEP farfield validation
│   └── results/
├── 04_absorption/                  ← see 04_absorption/README.md
│   ├── absorption_sim.py           ← broadband T/R/A spectrum + harminv Q
│   └── results/
├── benchmarks/                     ← Phase 2: see benchmarks/README.md
│   ├── benchmark.py                ← resolution / symmetry / MPI benchmarks
│   └── results/
├── 05_solver_comparison/           ← Phase 3: FDTD vs RCWA
│   ├── rcwa_sim.py                 ← torcwa RCWA sweep → PhaseLibrary
│   ├── compare_solvers.py          ← overlay plot + speed table
│   └── results/
├── 06_surrogate/                   ← Phase 4: surrogate + inverse design
│   ├── train.py                    ← train MLP on PhaseLibrary data
│   ├── optimise.py                 ← gradient-based pillar width optimisation
│   └── results/
├── 07_broadband/                   ← Phase 5: achromatic / broadband design
│   ├── multiwl_train.py            ← multi-λ MLP: [w/period, λ] → (|T|, ∠T)
│   ├── achromatic_design.py        ← joint optimisation across wavelengths
│   └── results/
└── envs/
    └── meep_env.yml
```

---

## Utils API

### `utils/sweep.py` — Unit-cell sweep engine

The central sweep engine used by all Phase 1 simulations.  Runs a 2D FDTD
parameter sweep over pillar widths, extracts complex transmission coefficients,
and returns a **`PhaseLibrary`** object.

```python
from utils.sweep import sweep, PhaseLibrary, run_unit_cell

# Build a library (runs MEEP)
lib = sweep(
    material="TiO2", wavelength=0.532, period=0.25, height=0.60,
    n_glass=1.5, resolution=64,
    n_widths=50,         # fixed number of samples, OR:
    phase_step=5.0,      # adaptive — auto-sets n_widths for ≤ 5° spacing
)
lib.save("results/phase_library.npz")

# Load an existing library
lib = PhaseLibrary.load("results/phase_library.npz")

# Nearest-neighbour phase → width lookup
widths, errors = lib.assign_widths(target_phases_rad)

# Metadata
lib.phase_coverage()              # total phase range in degrees
lib.period, lib.wavelength        # geometry parameters

# Dict-style access (backward compatible with raw np.load output)
lib["widths"], lib["phases"], lib["period"], lib["material"]
```

Can also be run directly as a script:
```bash
# Generic sweep (outputs to any directory)
python run.py utils/sweep.py --outdir 01_beam_steering/results

# Adaptive mode: guarantee ≤ 5° phase steps
python run.py utils/sweep.py --phase-step 5 --outdir 01_beam_steering/results

# Different material / geometry
python run.py utils/sweep.py --material SiO2 --height 0.80 --period 0.30
```

**`--phase-step DEG`** mode: runs a fast 15-point coarse sweep first to
estimate the geometry's phase coverage, then computes the number of samples
needed before running the full sweep — no manual tuning of `--n-widths`
required.

---

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
| `plot_focal_spot(x, intensity, ...)` | |Ez|² at focal plane with FWHM annotation |
| `plot_field_propagation(x_um, y_um, intensity_2d, ...)` | 2D |Ez|² propagation map |
| `plot_hologram_comparison(theta, intensity, targets, ...)` | Farfield vs target markers + phase profile |
| `plot_absorption_spectrum(wavelengths_nm, T, R, A, ...)` | 3-panel T / R / A spectrum with harminv resonance markers |
| `plot_solver_comparison(widths_meep, ..., widths_rcwa, ...)` | MEEP vs RCWA overlay: |T| and ∠T side by side |
| `plot_surrogate_training(train_losses, val_losses, pred_amp, ...)` | 3-panel: loss curves + |T| parity + ∠T parity |
| `plot_optimised_design(x_um, target_phases, phases_opt, ...)` | 4-panel: phase profile, widths, amplitudes, convergence |
| `plot_achromatic_design(x_um, wavelengths, target_phases, ...)` | Per-λ phase panels + widths + convergence |

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
