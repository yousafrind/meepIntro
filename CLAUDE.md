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

**Current status:** Phase 1 in progress.
- `01_beam_steering/` — complete (unit cell sweep + full array sim)
- `02_metalens/` — complete (metalens phase profile + MEEP FDTD + ASM focal spot analysis)
- `03_holography/` — complete (Gerchberg-Saxton phase retrieval + MEEP FDTD validation)
- `04_absorption/` — not yet started

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

## Utils modules

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

## Conventions

- All lengths in **micrometres (μm)** throughout — MEEP natural units with c=1
- MEEP frequency = 1/λ_μm
- `results/` dirs are gitignored (auto-created at runtime)
- Scripts are self-contained but share utils; always import from `utils.*`
- CLI flags override hardcoded constants at the top of each script
- MPI output: use `meep.am_master()` or rank-0 guard before printing/saving

---

## What's next (Phase 1 remaining work)

- [x] `01_beam_steering/` — done
- [x] `02_metalens/` — done
- [x] `03_holography/` — done
- [ ] `04_absorption/` — transmission spectra, harminv for resonance finding
- [ ] Phase 2: benchmark resolution/symmetry/MPI scaling on 01_beam_steering
