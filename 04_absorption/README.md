# 04 — Resonant Metasurface Absorption

Broadband transmission/reflection/absorption spectra of a TiO2 pillar unit
cell, plus **harminv** resonance extraction to measure Q-factors.

---

## Physics

### Mie resonances

A sub-wavelength TiO2 cylinder supports **Mie resonances** — eigenmodes of
the pillar where the field is strongly confined inside the high-index material.
The dominant resonances in the visible are:

| Mode | Polarisation | Approximate condition |
|------|--------------|-----------------------|
| Magnetic dipole (MD) | TM (Ez) | 2r·n_eff ≈ λ |
| Electric dipole (ED) | TM (Ez) | appears above MD |

These produce dips in transmission and peaks in reflection/absorption.

### Guided-mode resonance (GMR) / Wood's anomaly

When the free-space wavelength equals the grating period times the substrate
index (λ = n_sub · period), the first diffraction order becomes grazing.
Near this **Wood's anomaly wavelength** a guided-mode resonance can appear —
a sharp Fano feature on the broad Mie background:

```
λ_Wood = n_glass × period   (normal incidence, first order)
```

For the default parameters (period = 350 nm, n_glass = 1.5):
**λ_Wood ≈ 525 nm**

### Q-factor from harminv

MEEP's harminv decomposes the time-domain field at a monitor point into a sum
of damped sinusoids:

```
E(t) = Σ  A_k · exp( i·ω_k·t − γ_k·t )
```

The quality factor is Q = ω_k / (2·γ_k).  A narrow harminv source centred
on a resonance found in the broadband spectrum isolates the mode cleanly.

---

## Two-stage workflow

```
Stage 1: broadband Gaussian source
         ↓
         DFT flux monitors → T(ω), R(ω), A = 1 − T − R spectrum
         ↓
         identify peak absorption wavelength λ_peak

Stage 2: narrow Gaussian source centred at λ_peak
         ↓
         harminv on Ez(t) at monitor
         ↓
         resonance frequencies, Q-factors, decay rates
```

No phase library is needed — this is a standalone unit-cell measurement.

---

## Usage

```bash
# Quick preview (resolution 32, ~5–15 min)
python run.py 04_absorption/absorption_sim.py

# Accurate run (resolution 64)
python run.py 04_absorption/absorption_sim.py --resolution 64

# Scan a different geometry
python run.py 04_absorption/absorption_sim.py \
    --period 0.40 --height 0.50 --width 0.30 --resolution 64

# Broadband spectrum only (skip harminv)
python run.py 04_absorption/absorption_sim.py --no-harminv

python run.py 04_absorption/absorption_sim.py --help
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MATERIAL` | TiO2 | Pillar material |
| `PERIOD` | 0.35 μm | Unit cell period |
| `HEIGHT` | 0.40 μm | Pillar height |
| `WIDTH` | 0.21 μm | Pillar width (60% fill) |
| `WAVELENGTH` | 0.55 μm | harminv search centre |
| `BW` | 0.30 μm | Half-bandwidth of broadband source |
| `NFREQ` | 150 | DFT frequency points |
| `RESOLUTION` | 32 px/μm | MEEP resolution |

---

## Output

| File | Description |
|------|-------------|
| `results/epsilon_map.png` | Permittivity cross-section |
| `results/transmission_spectrum.png` | T / R / A vs wavelength with resonance markers |
| `results/absorption.npz` | Raw spectrum + resonance data |

---

## Expected Results

For the defaults (TiO2, 350 nm period, 400 nm height, 210 nm width):

- Broad absorption feature centred near 500–600 nm (magnetic Mie dipole)
- Possible sharp Fano dip near λ_Wood ≈ 525 nm
- **harminv Q** of the dominant mode: typically 10–100 for a Mie resonance,
  100–1000 for a sharp GMR

**To improve peak absorption (towards unity):**
- Increase pillar height → stronger mode confinement
- Use a reflective back-plane (metal-insulator-metal stack) → Phase 4
- Optimise width for critical coupling (radiation loss = absorption loss)

**To increase Q:**
- Use a taller, narrower pillar to approach guided-mode resonance conditions
- Switch to a lower-index substrate to sharpen GMR features

---

## Polarisation Note

TM polarisation (Ez) throughout.  TE would require `mp.Hz` source and would
excite different Mie orders (electric dipole dominant for TE).

---

## Loading Results

```python
import numpy as np

data = np.load("04_absorption/results/absorption.npz", allow_pickle=True)
wl   = data["wavelengths_nm"]
T, R, A = data["T"], data["R"], data["A"]

import matplotlib.pyplot as plt
plt.plot(wl, A, label="Absorption")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorption")
plt.legend(); plt.show()
```
