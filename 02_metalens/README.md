# 02 — Flat Metalens (Focusing Metasurface)

A phase-only flat lens that focuses a normally-incident plane wave to a
diffraction-limited spot using the same TiO2 pillar geometry as
`01_beam_steering`.

---

## Physics

A metalens imposes a **quadratic phase profile** on the transmitted wavefront:

```
φ(x) = -k₀ · (√(x² + f²) − f)
```

where f is the focal length and k₀ = 2π/λ.
This ensures that the optical path length from every point on the lens to the
focal point is equal — constructive interference at (0, f).

The **numerical aperture** of a cylindrical lens with aperture D and focal
length f:

```
NA = sin(arctan(D / 2f)) = (D/2) / √((D/2)² + f²)
```

The **diffraction-limited FWHM** (cylindrical / line focus):

```
FWHM_ideal ≈ 0.5λ / NA
```

---

## Workflow

```
01_beam_steering/unit_cell_sweep.py → phase_library.npz → metalens_sim.py
```

### Step 1: Build phase library (shared with 01_beam_steering)

```bash
python run.py 01_beam_steering/unit_cell_sweep.py
```

This is the same sweep as in the beam steering example.
The metalens reuses the library; you do **not** need to re-sweep if you
already ran the beam steering example.

### Step 2: Simulate the metalens

```bash
python run.py 02_metalens/metalens_sim.py
```

Output:

| File | Description |
|------|-------------|
| `results/epsilon_map.png` | Permittivity cross-section |
| `results/metalens_layout.png` | Pillar widths + quadratic phase profile |
| `results/focal_spot.png` | |Ez|² at focal plane with FWHM annotation |
| `results/field_propagation.png` | 2D |Ez|² — beam focusing map |
| `results/focal_spot.npz` | Raw arrays (x, intensity, FWHM, field_2d, …) |

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WAVELENGTH` | 0.532 μm | Free-space wavelength |
| `FOCAL_LEN` | 10.0 μm | Focal length |
| `LENS_WIDTH` | 5.0 μm | Lens aperture (D) |
| `PILLAR_H` | 0.60 μm | Must match unit_cell_sweep |
| `RESOLUTION` | 32 px/μm | MEEP resolution |
| `N_ASM` | 200 | Y-planes in the field-propagation map |

All parameters are overridable via CLI:

```bash
python run.py 02_metalens/metalens_sim.py --focal-len 8 --lens-width 6 --resolution 64
python run.py 02_metalens/metalens_sim.py --help
```

---

## Near-to-Far Field: Angular Spectrum Method

After MEEP, a 1D DFT field Ez(x) is collected on a plane 0.4λ above the
pillar tops.  The **Angular Spectrum Method (ASM)** propagates this field to
any y-plane analytically:

```
Ê(kx, y) = Ê(kx, 0) · exp(i·kz·y)
kz = √(k₀² − kx²)        (propagating)
kz = i·√(kx² − k₀²)      (evanescent — decays automatically)
```

This is equivalent to solving the Helmholtz equation exactly for each plane
wave component.  No far-field approximation is made.

---

## Expected Results

For λ=532 nm, TiO2, D=5 μm, f=10 μm:

- NA ≈ 0.24
- Diffraction-limited FWHM ≈ 1.1 μm
- Measured FWHM: 1.1–2 × limit (depends on phase library coverage)
- `field_propagation.png` should show clear convergence to a bright spot at y ≈ f

**To improve focusing quality:**
1. Increase pillar height (better 2π phase coverage) in `unit_cell_sweep.py`
2. Increase `--lens-width` (higher NA → tighter spot)
3. Increase `--resolution` from 32 to 64 for a more accurate result

---

## Polarisation Note

TM polarisation (Ez) throughout — same as `01_beam_steering`.
Switching to TE requires changing the source to `mp.Hz` and re-sweeping the
phase library.
