# 01 — Beam Steering Metasurface

A phase-gradient metasurface that deflects a normally-incident plane wave
to a user-specified angle, demonstrating the generalised Snell's law.

---

## Physics

A 1D array of high-index pillars (default: rutile TiO2) on a glass substrate.
Each pillar has a different width, imparting a different phase delay to the
transmitted field. Arranging widths so the phase increases linearly across one
**supercell** (0 → 2π) creates a phase gradient that steers the beam.

**Generalised Snell's law:**

```
sin(θ_t) = λ / Λ
```

where Λ = supercell period = N × (unit cell period).

Because Λ must contain an integer number of unit cells, the actual steering
angle is quantised. The scripts report both the requested and actual angle.

---

## Workflow

```
unit_cell_sweep.py  →  phase_library.npz  →  full_array_sim.py
```

### Step 1: Build phase library (run once per geometry)

```bash
python run.py 01_beam_steering/unit_cell_sweep.py
```

Sweeps TiO2 pillar width from 5% to 90% of the unit cell period.
At each width: runs a 2D FDTD sim with Bloch periodic BCs, extracts the
complex transmission coefficient T = E_struct / E_ref, records |T| and ∠T.

Output:
- `results/phase_library.npz` — sweep data
- `results/phase_library.png` — 4-panel plot

Key parameter to tune: `--height` (pillar height). Higher pillars accumulate
more phase → better 2π coverage. Default 0.60 μm at 532 nm is a good start.

### Step 2: Full metasurface simulation

```bash
python run.py 01_beam_steering/full_array_sim.py
```

Reads the library, builds the supercell geometry, runs MEEP, and computes
the farfield angular spectrum via FFT of the DFT near-field.

Output:
- `results/epsilon_map.png`           — permittivity cross-section
- `results/pillar_layout.png`         — assigned widths + phase profile
- `results/farfield_<angle>deg.png`   — linear + log farfield plots
- `results/farfield_<angle>deg.npz`   — raw angle/intensity arrays

---

## Parameters (edit top of full_array_sim.py or use CLI flags)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WAVELENGTH` | 0.532 μm | Free-space wavelength |
| `ANGLE` | 30° | Target steering angle from normal |
| `MS_WIDTH` | 5.0 μm | Metasurface lateral width (more = sharper farfield) |
| `PILLAR_H` | 0.60 μm | Pillar height (must match unit_cell_sweep) |
| `RESOLUTION` | 32 px/μm | MEEP resolution (32=fast preview, 64=accurate) |

---

## Expected results

For λ=532 nm, TiO2 pillars (n≈2.77), period=250 nm, height=600 nm:
- Phase coverage: ~250–300° (reasonable for visible TiO2)
- Steering efficiency: 40–70% (depends on amplitude uniformity)
- Main lobe at target angle, residual zeroth-order forward beam

To improve efficiency:
1. Increase pillar height until phase coverage ≥ 360°
2. Use a material with higher index (Si in NIR, GaN in UV)
3. Optimise unit cell geometry (Phase 4)

---

## Polarisation note

These simulations use **TM polarisation** (Ez, i.e. E-field parallel to
the pillar axis in 2D). Switching to TE requires changing the source
component to `mp.Hz`. Phase libraries for TM and TE are different and
must be swept separately.
