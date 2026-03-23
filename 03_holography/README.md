# 03 — Metasurface Hologram

A phase-only hologram that reconstructs a target multi-spot farfield pattern.
The **Gerchberg-Saxton (GS)** algorithm computes the required phase distribution;
MEEP FDTD validates the design.

---

## Physics

A phase-only metasurface hologram works by interference: each pillar imparts
a phase delay φ(x), and the superposition of all transmitted plane waves
forms the desired farfield pattern.

**Gerchberg-Saxton (GS) algorithm** — iterative phase retrieval:

```
Start: hologram h[n] = exp(i·φ_rand[n])  (random phases, unit amplitude)

Loop:
  1. Forward:   H[k] = FFT(h[n])          hologram → farfield
  2. Constrain: H'[k] = √T[k] · exp(i·∠H[k])   replace |H| with √target
  3. Backward:  h'[n] = IFFT(H'[k])       farfield → hologram
  4. Constrain: h[n] = exp(i·∠h'[n])      phase-only (discard amplitude)

Output: φ[n] = ∠h[n]
```

The target `T[k]` has unit peaks at the FFT bins corresponding to the desired
farfield angles.  Bins relate to angles via:

```
sin(θ) = k · λ / W       W = aperture width,  k = FFT bin index
```

**Limitation — phase-only constraint:** A phase-only hologram cannot achieve
100% efficiency for arbitrary multi-spot patterns.  The theoretical maximum
for N equal-intensity spots is `1/N` (power split equally), though GS finds
phase profiles that approach this in practice.

---

## Workflow

```
01_beam_steering/results/phase_library.npz  →  hologram_sim.py
```

Reuses the phase library from `01_beam_steering` — no re-sweep needed.

```bash
# Step 1: build phase library (once)
python run.py 01_beam_steering/unit_cell_sweep.py

# Step 2: design + simulate the hologram
python run.py 03_holography/hologram_sim.py
```

Output:

| File | Description |
|------|-------------|
| `results/epsilon_map.png` | Permittivity cross-section |
| `results/hologram_layout.png` | Pillar widths + GS phase profile |
| `results/hologram_comparison.png` | Simulated farfield + target markers + phase |
| `results/hologram.npz` | Raw arrays (theta, intensity, phases, eff_curve, …) |

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WAVELENGTH` | 0.532 μm | Free-space wavelength |
| `HOLO_WIDTH` | 10.0 μm | Hologram aperture (more pillars = better angular resolution) |
| `PILLAR_H` | 0.60 μm | Must match unit_cell_sweep |
| `TARGET_ANGLES` | -30°, +30° | Farfield target spot angles |
| `N_GS` | 100 | GS iterations (50–200 is typically sufficient) |
| `RESOLUTION` | 32 px/μm | MEEP resolution |

All overridable via CLI:

```bash
# Three-spot hologram at -45°, 0°, +45°
python run.py 03_holography/hologram_sim.py --target-angles -45 0 45

# Custom aperture and resolution
python run.py 03_holography/hologram_sim.py --holo-width 15 --resolution 64

python run.py 03_holography/hologram_sim.py --help
```

---

## Expected Results

For λ=532 nm, 10 μm aperture, targets at ±30°:

- **GS efficiency** (design stage): ~40–50% (power in target bins / total)
- **MEEP efficiency** (per spot): ~10–25% each, ~25–45% combined
- The two-spot pattern should be clearly visible in `hologram_comparison.png`
- Asymmetric residual background from amplitude non-uniformity (~10–30%)

**Angular resolution:** ≈ λ/W = 0.532/10 ≈ 3°.
Spots must be separated by more than this to be resolved.

**To improve efficiency:**
1. Increase pillar height → better 2π phase coverage → lower quantisation error
2. Increase aperture (`--holo-width`) → more pillars → finer GS resolution
3. Reduce number of target spots (2-spot is more efficient than 4-spot)
4. Phase 4 (inverse design) can achieve near-ideal efficiencies

---

## GS vs Inverse Design

GS is a classical iterative algorithm with no gradient information.
It converges quickly (~50 iterations) but cannot escape local optima.
For production-quality holograms, gradient-based optimisation (Phase 4,
PyTorch + autograd) achieves significantly higher efficiencies by directly
minimising the difference between simulated and target farfields.

---

## Polarisation Note

TM polarisation (Ez) throughout — same as `01_beam_steering` and `02_metalens`.
