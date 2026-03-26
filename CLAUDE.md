# CLAUDE.md — Developer Context for meepIntro

This file gives Claude Code instant context when starting a new session on this repo.

---

## What this repo is

A structured, self-paced introduction to electromagnetic metasurface simulation
and optimisation. The progression is:

```
Phase 1: MEEP FDTD fundamentals (4 metasurface examples)
Phase 2: Performance — resolution tuning, symmetry, MPI scaling
Phase 3: Solver comparison — FDTD (MEEP) vs RCWA vs torcwa (GPU)
Phase 4: Surrogate-model optimisation (PyTorch / ROCm)
```

**Current status:** Phase 1 complete. Phase 2 complete. Phase 3 complete.
- `01_beam_steering/` — complete (unit cell sweep + full array sim)
- `02_metalens/` — complete (metalens phase profile + MEEP FDTD + ASM focal spot analysis)
- `03_holography/` — complete (Gerchberg-Saxton phase retrieval + MEEP FDTD validation)
- `04_absorption/` — complete (broadband T/R/A spectra + harminv Q-factor extraction)
- `benchmarks/` — Phase 2: resolution / symmetry / MPI scaling benchmarks
- `05_solver_comparison/` — Phase 3: torcwa RCWA sweep + MEEP vs RCWA overlay

---

## Hardware target

- Primary dev machine: any CPU (laptop / WSL)
- Phase 3+ target: AMD Ryzen AI Max 395 with ROCm
- PyTorch ROCm exposes itself as `cuda` — code uses `torch.cuda.is_available()`
  and works unchanged on both CUDA and ROCm

---

## How to run anything

All simulations go through `run.py` — never call scripts directly.

```bash
# Always activate the env first
conda activate meep

# Run a script (auto-detects cores, uses MPI if available)
python run.py 01_beam_steering/unit_cell_sweep.py
python run.py 01_beam_steering/full_array_sim.py --angle 45

# Override core count for quick tests
MEEP_NPROCS=4 python run.py 01_beam_steering/unit_cell_sweep.py --n-widths 5 --resolution 32

# Single process (no mpirun)
MEEP_NPROCS=1 python run.py ...
```

`run.py` logic: reads `MEEP_NPROCS` env var → falls back to `multiprocessing.cpu_count()`.
If `mpirun` is on PATH and nprocs > 1, launches with MPI. Otherwise single-process Python.

---

## 01_beam_steering

**Two-step workflow:**

```
unit_cell_sweep.py  →  results/phase_library.npz  →  full_array_sim.py
```

1. **unit_cell_sweep.py** — sweeps TiO2 pillar width 5–90% of period.
   Per width: 2D FDTD with Bloch BCs, extracts T = E_struct/E_ref, records |T| and ∠T.
   Output: `results/phase_library.npz`, `results/phase_library.png`

2. **full_array_sim.py** — reads library, builds supercell, runs MEEP, computes
   farfield via FFT of DFT near-field.
   Output: `results/epsilon_map.png`, `results/pillar_layout.png`,
           `results/farfield_<angle>deg.{png,npz}`

**Key parameters** (top of `full_array_sim.py` or CLI flags):

| Parameter | Default | Note |
|-----------|---------|------|
| `WAVELENGTH` | 0.532 μm | Free-space wavelength |
| `ANGLE` | 30° | Target steering angle |
| `MS_WIDTH` | 5.0 μm | Metasurface lateral width |
| `PILLAR_H` | 0.60 μm | Must match unit_cell_sweep |
| `RESOLUTION` | 32 px/μm | 32=fast preview, 64=accurate |

**Polarisation:** TM (Ez). Switching to TE → change source to `mp.Hz`.
Phase libraries for TM and TE differ; must sweep separately.

**Expected performance** (λ=532 nm, TiO2, period=250 nm, h=600 nm):
- Phase coverage ~250–300° (reasonable; need h > ~700 nm for full 2π)
- Steering efficiency 40–70%

---

## 02_metalens

**Workflow** (reuses phase library from 01_beam_steering — no re-sweep needed):

```
01_beam_steering/results/phase_library.npz  →  metalens_sim.py
```

**metalens_sim.py** — applies quadratic phase φ(x) = -k₀(√(x²+f²) - f), assigns
pillar widths, runs 2D MEEP, then propagates the DFT near-field to the focal
plane via the **Angular Spectrum Method (ASM)**.
Output: `results/epsilon_map.png`, `results/metalens_layout.png`,
        `results/focal_spot.png`, `results/field_propagation.png`,
        `results/focal_spot.npz`

**Key parameters:**

| Parameter | Default | Note |
|-----------|---------|------|
| `WAVELENGTH` | 0.532 μm | Free-space wavelength |
| `FOCAL_LEN` | 10.0 μm | Focal length |
| `LENS_WIDTH` | 5.0 μm | Lens aperture |
| `PILLAR_H` | 0.60 μm | Must match unit_cell_sweep |
| `RESOLUTION` | 32 px/μm | 32=fast preview, 64=accurate |
| `N_ASM` | 200 | Y-planes in the 2D field propagation map |

**Expected results** (defaults): NA ≈ 0.24, FWHM ≈ 1–2× diffraction limit (1.1 μm).
`field_propagation.png` shows |Ez|² concentrating at y ≈ f above the lens.

---

## 03_holography

**Workflow** (reuses phase library from 01_beam_steering):

```
01_beam_steering/results/phase_library.npz  →  hologram_sim.py
```

**hologram_sim.py** — Gerchberg-Saxton (GS) iterative phase retrieval computes
the hologram phase profile that reconstructs target farfield spots. Assigns
pillar widths, runs MEEP, compares simulated farfield to GS target.
Output: `results/epsilon_map.png`, `results/hologram_layout.png`,
        `results/hologram_comparison.png`, `results/hologram.npz`

**Key parameters:**

| Parameter | Default | Note |
|-----------|---------|------|
| `WAVELENGTH` | 0.532 μm | Free-space wavelength |
| `HOLO_WIDTH` | 10.0 μm | Hologram aperture (40 pillars at period=250 nm) |
| `TARGET_ANGLES` | -30°, +30° | Farfield target spot angles (space-separated) |
| `N_GS` | 100 | GS iterations |
| `PILLAR_H` | 0.60 μm | Must match unit_cell_sweep |
| `RESOLUTION` | 32 px/μm | 32=fast preview, 64=accurate |

```bash
# Custom target angles
python run.py 03_holography/hologram_sim.py --target-angles -45 0 45
```

**Expected results**: GS efficiency ~40–50%; MEEP combined spot efficiency ~25–45%.

---

## 04_absorption

**Workflow** (standalone — no phase library needed):

```
absorption_sim.py  (Stage 1: broadband)  →  T(ω), R(ω), A(ω) spectrum
                   (Stage 2: harminv)    →  resonance freq, Q-factor, decay rate
```

**absorption_sim.py** — two-stage MEEP simulation:
1. Broadband Gaussian source + DFT flux monitors → T/R/A spectrum
2. Narrow Gaussian centred on peak absorption → harminv extracts modes

Output: `results/epsilon_map.png`, `results/transmission_spectrum.png`,
        `results/absorption.npz`

**Key parameters:**

| Parameter | Default | Note |
|-----------|---------|------|
| `MATERIAL` | TiO2 | Pillar material |
| `PERIOD` | 0.35 μm | Unit cell period |
| `HEIGHT` | 0.40 μm | Pillar height |
| `WIDTH` | 0.21 μm | Pillar width (60% fill factor) |
| `WAVELENGTH` | 0.55 μm | harminv search centre (≈ resonance) |
| `BW` | 0.30 μm | Half-bandwidth of broadband source |
| `NFREQ` | 150 | DFT frequency samples |
| `RESOLUTION` | 32 px/μm | 32=fast, 64=accurate |

```bash
python run.py 04_absorption/absorption_sim.py
python run.py 04_absorption/absorption_sim.py --resolution 64
python run.py 04_absorption/absorption_sim.py --period 0.40 --height 0.50

# Skip harminv (spectrum only)
python run.py 04_absorption/absorption_sim.py --no-harminv
```

**Expected results**: Mie resonance dip in T near 500–600 nm; possible sharp
Fano feature at Wood's anomaly λ_W = n_glass × period ≈ 525 nm; harminv
Q = 10–100 for Mie, 100–1000 for GMR.

---

## Phase 2 — benchmarks/

### `benchmarks/benchmark.py`
Drives three performance benchmarks and reports wall-time tables + plots.

```bash
python run.py benchmarks/benchmark.py                # all three benchmarks
python run.py benchmarks/benchmark.py --mode resolution
python run.py benchmarks/benchmark.py --mode symmetry
python run.py benchmarks/benchmark.py --mode mpi --max-procs 8
python run.py benchmarks/benchmark.py --quick        # smoke-test in < 1 min
```

| Benchmark | What it measures |
|-----------|-----------------|
| `resolution` | Wall time at res=16/32/64/128; compares to O(r³) theory |
| `symmetry` | Mirror(X) speedup (~2×) vs no-symmetry at fixed resolution |
| `mpi` | Strong scaling speedup and parallel efficiency vs nprocs |

Output: `benchmarks/results/benchmark.png`, `benchmark_report.txt`, `benchmark.json`

**Mirror(X) symmetry** is now exposed in `utils/sweep.py`:

```bash
# Use symmetry in any sweep (halves x grid, ~2× speedup, valid for k=0)
python run.py utils/sweep.py --symmetry --resolution 64
```

Or in Python:
```python
from utils.sweep import sweep
lib = sweep(..., use_symmetry=True)
```

---

## Utils modules

### `utils/sweep.py`
Central unit-cell sweep engine used by all Phase 1 sims.

```python
from utils.sweep import sweep, PhaseLibrary, run_unit_cell

# Build a new library (runs MEEP)
lib = sweep(
    material="TiO2", wavelength=0.532, period=0.25, height=0.60,
    n_glass=1.5, resolution=64,
    n_widths=50,        # OR: phase_step=5.0 (degrees) for adaptive sampling
)
lib.save("results/phase_library.npz")

# Load an existing library
lib = PhaseLibrary.load("results/phase_library.npz")
print(lib.phase_coverage())           # total phase range in degrees
widths, errors = lib.assign_widths(target_phases_rad)  # nearest-neighbour lookup

# Dict-style access (backward compatible with raw np.load)
lib["widths"], lib["phases"], lib["period"], lib["material"]
```

Can also be run standalone as a script:
```bash
# Same as running 01_beam_steering/unit_cell_sweep.py but more flexible
python run.py utils/sweep.py --outdir 01_beam_steering/results

# Adaptive mode: sample every 5° of phase (auto-computes n_widths)
python run.py utils/sweep.py --phase-step 5 --outdir my_sim/results

# Different geometry
python run.py utils/sweep.py --material SiO2 --height 0.80 --period 0.30
```

`--phase-step DEG` runs a 15-point coarse sweep first to estimate phase coverage,
then computes the n_widths needed to achieve ≤ DEG spacing before the full sweep.

### `utils/materials.py`
```python
from utils.materials import load_material
mat = load_material("TiO2")      # or "SiO2", "glass", or a file path
med = mat(0.532)                 # → mp.Medium at λ=532 nm
n   = mat.n(0.532)               # refractive index only
```
Reads space-separated n,k files (refractiveindex.info format). Linear interpolation.
For lossy materials uses `D_conductivity` to represent Im(ε) in MEEP.

### `utils/device.py`
```python
from utils.device import print_platform_info, get_torch_device
print_platform_info()   # prints CPU/MPI/GPU summary — call at script start
dev = get_torch_device()  # torch.device or None if PyTorch not installed
```
MPI-rank-aware: only rank 0 prints. Safe to call in Phase 1/2 (no PyTorch needed).

### `utils/viz.py`
Matplotlib `Agg` backend — no display needed, works headless/WSL.
All functions auto-create output directories.

Key functions:
- `plot_phase_library(...)` — 4-panel amplitude/phase sweep
- `plot_angular_spectrum(...)` — linear + log farfield
- `plot_epsilon(sim, filename)` — permittivity map
- `plot_fields(sim, component, filename)` — near-field snapshot
- `plot_pillar_layout(...)` — geometry + phase profile
- `plot_focal_spot(...)` — |Ez|² at focal plane with FWHM annotation
- `plot_field_propagation(...)` — 2D |Ez|² field map showing beam focusing
- `plot_hologram_comparison(...)` — simulated farfield vs target markers + phase profile
- `plot_absorption_spectrum(wavelengths_nm, T, R, A, resonances, filename)` — 3-panel T/R/A spectrum with harminv resonance markers

---

## Materials

Bundled in `materials/`:
- `TiO2` → `TiO2_rutile_Siefke2016.txt` (Siefke et al., Adv. Opt. Mat. 2016)
- `SiO2` / `glass` → `SiO2_Malitson1965.txt` (Malitson, JOSA 1965)

To add a material: download from refractiveindex.info (space-separated export),
pass `--material path/to/file.txt`. Format: `wavelength_um  n  [k]`.

---

## Conda environment

```bash
conda env create -f envs/meep_env.yml
conda activate meep
```

ROCm PyTorch is **not** in the yml (Phase 3 only). Install separately:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

---

## Phase 3 — 05_solver_comparison/

### Workflow

```bash
# Step 1: RCWA sweep (single-process, seconds vs hours for MEEP)
MEEP_NPROCS=1 python run.py 05_solver_comparison/rcwa_sim.py

# Step 2: compare against MEEP library
python run.py 05_solver_comparison/compare_solvers.py \
    --meep-lib 01_beam_steering/results/phase_library.npz \
    --meep-time <seconds>    # optional: enables speed table
```

### `rcwa_sim.py`
RCWA unit-cell sweep using `torcwa`.  Geometry matches `01_beam_steering`
defaults (TiO2, period=250 nm, height=600 nm, λ=532 nm).

Key parameters:

| Parameter | Default | Note |
|-----------|---------|------|
| `FOURIER_ORDER` | 15 | N → (2N+1)² total plane waves |
| `GEO_RESOLUTION` | 128 | Spatial grid pts per axis for pillar mask |
| `N_WIDTHS` | 50 | Width samples |

Output: `results/rcwa_phase_library.npz` (PhaseLibrary-compatible + `sweep_time_s`),
`results/rcwa_phase_library.png`

### `compare_solvers.py`
Loads both libraries, prints metadata + optional speed table, saves
`results/solver_comparison.png`.

`--align-phase` shifts the RCWA curve so its first point matches the MEEP
reference (removes global phase offset from different interface conventions).

### Physics note
RCWA S-parameters are inherently normalised (no separate reference run needed).
The glass substrate → air output convention may introduce a fixed phase offset
vs MEEP's `E_struct / E_ref` normalisation.  The relative shape (phase coverage,
resonance features) is identical; only the absolute reference differs.

---

## Conventions

- All lengths in **micrometres (μm)** throughout — MEEP natural units with c=1
- MEEP frequency = 1/λ_μm
- `results/` dirs are gitignored (auto-created at runtime)
- Scripts are self-contained but share utils; always import from `utils.*`
- CLI flags override hardcoded constants at the top of each script
- MPI output: use `meep.am_master()` or rank-0 guard before printing/saving

---

## What's next

**Phase 1 — complete**
- [x] `01_beam_steering/` — done
- [x] `02_metalens/` — done
- [x] `03_holography/` — done
- [x] `04_absorption/` — done (broadband T/R/A + harminv Q-factor extraction)

**Phase 2 — complete**
- [x] `benchmarks/benchmark.py` — resolution / Mirror(X) symmetry / MPI scaling
- [x] Apply results: Mirror(X) on by default in `unit_cell_sweep.py` and `04_absorption/absorption_sim.py`; resolution guidance added

**Phase 3 — complete**
- [x] `05_solver_comparison/rcwa_sim.py` — torcwa RCWA unit-cell sweep → PhaseLibrary
- [x] `05_solver_comparison/compare_solvers.py` — |T|/∠T overlay + speed table
- [x] `utils/viz.py` — `plot_solver_comparison()` added

**Phase 4 — planned**
- [ ] Surrogate-model optimisation (PyTorch / ROCm)
