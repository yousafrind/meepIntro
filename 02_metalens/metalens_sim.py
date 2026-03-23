"""
Flat metalens (focusing metasurface) — MEEP FDTD simulation.

A quadratic phase profile φ(x) = -k₀(√(x²+f²) - f) focuses a normally-incident
plane wave to a diffraction-limited spot at focal distance f.

Workflow
--------
1. Load the phase library produced by 01_beam_steering/unit_cell_sweep.py.
2. Compute the metalens phase profile across the lens aperture.
3. Assign pillar widths via nearest-neighbour lookup in the library.
4. Run 2D FDTD (MEEP, TM polarisation).
5. Extract DFT near-field → propagate via Angular Spectrum Method (ASM).
6. Characterise focal spot: FWHM vs diffraction limit.

Usage
-----
    # Step 1: build phase library (once)
    python run.py 01_beam_steering/unit_cell_sweep.py

    # Step 2: simulate the metalens
    python run.py 02_metalens/metalens_sim.py [options]

    --wavelength  0.532      free-space wavelength in μm
    --focal-len   10.0       focal length in μm
    --lens-width  5.0        lens aperture in μm
    --pillar-h    0.60       pillar height in μm (must match unit_cell_sweep)
    --n-glass     1.5        substrate refractive index
    --resolution  32         MEEP pixels per μm
    --lib-path    auto       path to phase_library.npz
                             ("auto" looks in ../01_beam_steering/results/)
    --n-asm       200        number of y-planes for field-propagation map
    --outdir      results    output directory (relative to this script)

Output
------
    results/epsilon_map.png          permittivity cross-section
    results/metalens_layout.png      pillar widths + quadratic phase profile
    results/focal_spot.png           |Ez|² at focal plane + FWHM annotation
    results/field_propagation.png    2D |Ez|² — beam focusing map
    results/focal_spot.npz           raw data arrays

Physics notes
-------------
* 2D FDTD, TM polarisation (Ez source).
* Periodic BC in x (k_point = 0); PML top and bottom.
  The lens is centred in the cell; for large enough apertures
  the periodic copies do not significantly overlap at the focal spot.
* Near-field DFT collected on a plane 0.4λ above the pillar tops.
* Angular Spectrum Method propagates this field to the focal plane
  and to N intermediate planes for the 2D field map.
* Diffraction-limited FWHM for a cylindrical lens: 0.5λ / NA,
  where NA = sin(arctan(D/2f)) and D = lens aperture.
"""

import argparse
import os
import sys

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.device    import print_platform_info
from utils.materials import load_material
from utils.sweep     import PhaseLibrary
from utils.viz       import (plot_epsilon, plot_pillar_layout,
                              plot_focal_spot, plot_field_propagation)


# ══════════════════════════════════════════════════════════════════════════════
#  USER PARAMETERS  ← edit these or override via CLI flags
# ══════════════════════════════════════════════════════════════════════════════

WAVELENGTH  = 0.532   # μm
FOCAL_LEN   = 10.0    # μm  — focal length
LENS_WIDTH  = 5.0     # μm  — lens aperture (lateral size)
PILLAR_H    = 0.60    # μm  — must match unit_cell_sweep
N_GLASS     = 1.5     #     — substrate refractive index
RESOLUTION  = 32      #     — MEEP pixels/μm
LIB_PATH    = "auto"  #     — "auto" → ../01_beam_steering/results/phase_library.npz
N_ASM       = 200     #     — y-planes in ASM field-propagation map
OUT_DIR     = "results"

# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Flat metalens FDTD simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wavelength",  type=float, default=WAVELENGTH,
                   help="Free-space wavelength in μm")
    p.add_argument("--focal-len",   type=float, default=FOCAL_LEN,
                   dest="focal_len", help="Focal length in μm")
    p.add_argument("--lens-width",  type=float, default=LENS_WIDTH,
                   dest="lens_width", help="Lens aperture (lateral width) in μm")
    p.add_argument("--pillar-h",    type=float, default=PILLAR_H,
                   dest="pillar_h", help="Pillar height in μm")
    p.add_argument("--n-glass",     type=float, default=N_GLASS,
                   dest="n_glass",  help="Substrate refractive index")
    p.add_argument("--resolution",  type=int,   default=RESOLUTION,
                   help="MEEP pixels per μm")
    p.add_argument("--lib-path",    default=LIB_PATH, dest="lib_path",
                   help='Path to phase_library.npz, or "auto"')
    p.add_argument("--n-asm",       type=int,   default=N_ASM,
                   dest="n_asm", help="Number of y-planes for ASM field map")
    p.add_argument("--outdir",      default=OUT_DIR,
                   help="Output directory (relative to this script)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Phase library
# ─────────────────────────────────────────────────────────────────────────────

def load_phase_library(lib_path_arg):
    """Resolve lib_path_arg and return a PhaseLibrary."""
    if lib_path_arg == "auto":
        path = os.path.normpath(os.path.join(
            _HERE, "..", "01_beam_steering", "results", "phase_library.npz"
        ))
    else:
        path = lib_path_arg
    try:
        return PhaseLibrary.load(path)
    except FileNotFoundError as e:
        sys.exit(
            f"\n[error] {e}\n"
            "          python run.py 01_beam_steering/unit_cell_sweep.py\n"
        )


def assign_pillar_widths(lib, target_phases):
    """Delegate to PhaseLibrary.assign_widths (kept for call-site compat)."""
    return lib.assign_widths(target_phases)


# ─────────────────────────────────────────────────────────────────────────────
# Metalens phase profile
# ─────────────────────────────────────────────────────────────────────────────

def metalens_phase(x_positions, focal_len, wavelength):
    """
    Compute the ideal metalens phase profile.

    φ(x) = -k₀ · (√(x² + f²) − f)

    Derivation: require equal optical path length from each lens point
    to the focal point at (0, f). The "-f" term sets φ(0) = 0.

    Parameters
    ----------
    x_positions : array  pillar x-coordinates in μm (centred at 0)
    focal_len   : float  focal length in μm
    wavelength  : float  free-space wavelength in μm

    Returns
    -------
    phases : array  required phase at each x in radians (all ≤ 0)
    """
    k0 = 2 * np.pi / wavelength
    return -k0 * (np.sqrt(x_positions**2 + focal_len**2) - focal_len)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builder
# ─────────────────────────────────────────────────────────────────────────────

def build_geometry(x_positions, pillar_widths, pillar_height,
                   period, pillar_medium, n_glass, sy, y_subs_top):
    """Return list of mp.GeometricObject for the full metalens supercell."""
    import meep as mp

    glass = mp.Medium(index=n_glass)

    geom = [
        mp.Block(
            size=mp.Vector3(mp.inf, y_subs_top + sy / 2, mp.inf),
            center=mp.Vector3(0, -(sy / 2 - (y_subs_top + sy / 2) / 2)),
            material=glass,
        )
    ]

    y_pillar_ctr = y_subs_top + pillar_height / 2
    for x0, w in zip(x_positions, pillar_widths):
        if w > 0.02 * period:
            geom.append(
                mp.Block(
                    size=mp.Vector3(w, pillar_height, mp.inf),
                    center=mp.Vector3(x0, y_pillar_ctr),
                    material=pillar_medium,
                )
            )
    return geom


# ─────────────────────────────────────────────────────────────────────────────
# Angular Spectrum Method (ASM)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_asm(Ez_near, sx, wavelength, delta_y):
    """
    Propagate Ez field by delta_y (in μm) using the Angular Spectrum Method.

    H(kx) = exp(i·kz·Δy),  kz = √(k₀² − kx²)
    Evanescent components (|kx| > k₀) decay automatically via Im(kz) > 0.

    Parameters
    ----------
    Ez_near   : 1D complex array  near-field Ez
    sx        : float             physical width in μm (= cell size in x)
    wavelength: float             free-space wavelength in μm
    delta_y   : float             propagation distance in μm

    Returns
    -------
    Ez_prop : 1D complex array at y + delta_y
    """
    Nx = len(Ez_near)
    dx = sx / Nx
    k0 = 2 * np.pi / wavelength

    Ez_k = np.fft.fft(Ez_near)
    kx   = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)   # rad/μm

    kz2  = k0**2 - kx**2
    # Evanescent: kz2 < 0 → kz is purely imaginary (decaying)
    kz   = np.where(kz2 >= 0,
                    np.sqrt(kz2.astype(complex)),
                    1j * np.sqrt((-kz2).astype(complex)))

    H = np.exp(1j * kz * delta_y)
    return np.fft.ifft(Ez_k * H)


def propagate_asm_stack(Ez_near, sx, wavelength, delta_y_array):
    """
    Propagate Ez to multiple planes (vectorised over delta_y_array).

    Returns
    -------
    stack : 2D complex array, shape (len(delta_y_array), Nx)
    """
    Nx = len(Ez_near)
    dx = sx / Nx
    k0 = 2 * np.pi / wavelength

    Ez_k = np.fft.fft(Ez_near)
    kx   = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)

    kz2  = k0**2 - kx**2
    kz   = np.where(kz2 >= 0,
                    np.sqrt(kz2.astype(complex)),
                    1j * np.sqrt((-kz2).astype(complex)))

    # delta_y_array: shape (M,), kz: shape (Nx,)
    # H[m, n] = exp(i·kz[n]·delta_y[m])
    dy   = np.asarray(delta_y_array)[:, np.newaxis]   # (M, 1)
    H    = np.exp(1j * kz[np.newaxis, :] * dy)        # (M, Nx)
    return np.fft.ifft(Ez_k[np.newaxis, :] * H, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# FWHM
# ─────────────────────────────────────────────────────────────────────────────

def compute_fwhm(x, intensity):
    """
    Compute the FWHM of a unimodal peak via linear interpolation.

    Returns None if the half-maximum crossings cannot be found.
    """
    i_max    = int(np.argmax(intensity))
    half_max = intensity[i_max] / 2.0

    # Left crossing
    left_x = None
    for i in range(i_max, 0, -1):
        if intensity[i - 1] <= half_max:
            t = (half_max - intensity[i]) / (intensity[i - 1] - intensity[i])
            left_x = x[i] + t * (x[i - 1] - x[i])
            break

    # Right crossing
    right_x = None
    for i in range(i_max, len(intensity) - 1):
        if intensity[i + 1] <= half_max:
            t = (half_max - intensity[i]) / (intensity[i + 1] - intensity[i])
            right_x = x[i] + t * (x[i + 1] - x[i])
            break

    if left_x is None or right_x is None:
        return None
    return right_x - left_x


# ─────────────────────────────────────────────────────────────────────────────
# MEEP simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_sim(args, lib, outdir):
    import meep as mp

    wavelength  = args.wavelength
    focal_len   = args.focal_len
    lens_width  = args.lens_width
    pillar_h    = args.pillar_h
    n_glass     = args.n_glass
    resolution  = args.resolution

    freq   = 1.0 / wavelength
    period = float(lib["period"])

    # Number of unit cells that fit in the lens aperture
    n_cells     = max(2, int(np.round(lens_width / period)))
    sx          = n_cells * period

    print(f"\n[sim] ── Metalens design ────────────────────────────")
    print(f"[sim]   Wavelength     : {wavelength*1000:.0f} nm")
    print(f"[sim]   Focal length   : {focal_len:.1f} μm")
    print(f"[sim]   Lens aperture  : {lens_width:.2f} μm  ({n_cells} unit cells)")
    print(f"[sim]   Actual width   : {sx:.3f} μm")
    print(f"[sim]   Unit cell Λ    : {period*1000:.0f} nm")

    # Numerical aperture and diffraction-limited FWHM
    half_d  = sx / 2.0
    na      = half_d / np.sqrt(half_d**2 + focal_len**2)   # sin(arctan(D/2f))
    fwhm_dl = 0.5 * wavelength / na                         # diffraction limit
    print(f"[sim]   NA             : {na:.3f}")
    print(f"[sim]   FWHM (limit)  : {fwhm_dl*1000:.0f} nm")

    # ── pillar x-positions (centred at 0) ─────────────────────────────────────
    x_positions   = (np.arange(n_cells) - (n_cells - 1) / 2.0) * period

    # ── metalens phase profile ────────────────────────────────────────────────
    target_phases = metalens_phase(x_positions, focal_len, wavelength)

    pillar_widths, phase_errors = assign_pillar_widths(lib, target_phases)

    print(f"\n[sim] ── Pillar assignment ─────────────────────────")
    print(f"[sim]   Width range : {pillar_widths.min()*1000:.1f} – "
                                  f"{pillar_widths.max()*1000:.1f} nm")
    print(f"[sim]   Phase error : mean {phase_errors.mean():.1f}°  "
                                 f"max {phase_errors.max():.1f}°")

    # ── load pillar material ──────────────────────────────────────────────────
    mat_name      = str(lib["material"])
    pillar_mat    = load_material(mat_name)
    pillar_medium = pillar_mat(wavelength)

    # ── cell layout ───────────────────────────────────────────────────────────
    dpml        = wavelength
    subs_thick  = 1.5 * wavelength
    # propagation height: focal_len + 2λ above pillar tops
    prop_height = focal_len + 2.0 * wavelength
    sy          = dpml + subs_thick + pillar_h + prop_height + dpml

    y_bot       = -sy / 2
    y_subs_top  = y_bot + dpml + subs_thick
    y_src       = y_bot + dpml + 0.4 * wavelength
    y_near      = y_subs_top + pillar_h + 0.4 * wavelength  # DFT monitor

    # ── build MEEP objects ────────────────────────────────────────────────────
    geometry = build_geometry(
        x_positions, pillar_widths, pillar_h,
        period, pillar_medium, n_glass,
        sy, y_subs_top,
    )

    sources = [
        mp.Source(
            mp.GaussianSource(frequency=freq, fwidth=0.05 * freq),
            component=mp.Ez,
            center=mp.Vector3(0, y_src),
            size=mp.Vector3(sx, 0),
        )
    ]

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=sources,
        k_point=mp.Vector3(),     # periodic BC in x, normal incidence
        resolution=resolution,
        eps_averaging=True,
    )

    # 1D DFT monitor just above pillars
    dft_near = sim.add_dft_fields(
        [mp.Ez], freq, freq, 1,
        center=mp.Vector3(0, y_near),
        size=mp.Vector3(sx, 0),
    )

    decay_pt = mp.Vector3(0, y_near)

    print(f"\n[sim] ── Running MEEP (resolution={resolution}) ────")

    if mp.am_master():
        plot_epsilon(
            sim,
            filename=os.path.join(outdir, "epsilon_map.png"),
            title=f"Permittivity map  |  λ={wavelength*1000:.0f} nm  "
                  f"|  f={focal_len:.1f} μm  |  NA={na:.3f}",
        )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ez, decay_pt, 1e-7
        )
    )

    ez_near = sim.get_dft_array(dft_near, mp.Ez, 0)

    return (ez_near, sx, y_near, y_subs_top + pillar_h,
            na, fwhm_dl,
            x_positions, pillar_widths, target_phases,
            period)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def analyse_focal_spot(ez_near, sx, wavelength, focal_len, n_asm):
    """
    ASM-propagate the near-field to the focal plane and to N intermediate
    planes, returning the focal-plane profile and the 2D field map.

    Returns
    -------
    x_um       : 1D array  x positions in μm
    focal_int  : 1D array  |Ez|² at focal plane
    y_prop     : 1D array  y positions for field map (relative to y_near)
    field_2d   : 2D array  |Ez|² shape (n_asm, Nx)
    fwhm_um    : float or None
    """
    Nx   = len(ez_near)
    x_um = np.linspace(-sx / 2, sx / 2, Nx)

    # Focal-plane field
    ez_focal   = propagate_asm(ez_near, sx, wavelength, focal_len)
    focal_int  = np.abs(ez_focal) ** 2

    # 2D field map: y from 0.5λ to focal_len + 1.5λ above y_near
    y_prop   = np.linspace(0.5 * wavelength,
                           focal_len + 1.5 * wavelength, n_asm)
    stack    = propagate_asm_stack(ez_near, sx, wavelength, y_prop)
    field_2d = np.abs(stack) ** 2

    fwhm_um = compute_fwhm(x_um, focal_int)
    return x_um, focal_int, y_prop, field_2d, fwhm_um


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print_platform_info()

    lib = load_phase_library(args.lib_path)

    (ez_near, sx, y_near, y_pillar_top,
     na, fwhm_dl,
     x_positions, pillar_widths, target_phases,
     period) = run_sim(args, lib, outdir)

    import meep as mp
    if not mp.am_master():
        return

    # ── pillar layout ─────────────────────────────────────────────────────────
    plot_pillar_layout(
        x_positions, pillar_widths,
        pillar_height=args.pillar_h,
        period=period,
        target_phases=target_phases,
        filename=os.path.join(outdir, "metalens_layout.png"),
    )

    # ── ASM propagation ───────────────────────────────────────────────────────
    print(f"\n[post] Propagating near-field via ASM  "
          f"(focal_len={args.focal_len} μm, n_planes={args.n_asm}) ...")

    x_um, focal_int, y_prop, field_2d, fwhm_um = analyse_focal_spot(
        ez_near, sx, args.wavelength, args.focal_len, args.n_asm
    )

    # ── focal spot plot ───────────────────────────────────────────────────────
    plot_focal_spot(
        x_um, focal_int,
        fwhm_um=fwhm_um,
        diffraction_limit_um=fwhm_dl,
        focal_len=args.focal_len,
        wavelength=args.wavelength,
        filename=os.path.join(outdir, "focal_spot.png"),
    )

    # ── 2D field-propagation map ──────────────────────────────────────────────
    # y_prop is relative to y_near; convert to absolute μm for the plot
    y_abs = y_pillar_top + y_prop
    plot_field_propagation(
        x_um, y_abs, field_2d,
        focal_len=args.focal_len,
        y_pillar_top=y_pillar_top,
        filename=os.path.join(outdir, "field_propagation.png"),
    )

    # ── save raw data ─────────────────────────────────────────────────────────
    np.savez(
        os.path.join(outdir, "focal_spot.npz"),
        x_um              = x_um,
        focal_intensity   = focal_int,
        fwhm_um           = fwhm_um if fwhm_um is not None else np.nan,
        fwhm_dl_um        = fwhm_dl,
        na                = na,
        focal_len         = args.focal_len,
        wavelength        = args.wavelength,
        y_prop            = y_prop,
        field_2d          = field_2d,
    )
    print(f"[post] Raw data saved → {outdir}/focal_spot.npz")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n[result] ── Focal spot summary ─────────────────────")
    print(f"[result]   NA                 : {na:.3f}")
    print(f"[result]   Diffraction limit  : {fwhm_dl*1000:.0f} nm")
    if fwhm_um is not None:
        strehl_proxy = focal_int.max() / focal_int.max()   # normalised to 1
        print(f"[result]   Measured FWHM     : {fwhm_um*1000:.0f} nm")
        print(f"[result]   FWHM / limit      : {fwhm_um/fwhm_dl:.2f}x  "
              f"({'near diffraction limit' if fwhm_um/fwhm_dl < 1.5 else 'above limit — consider more phase coverage'})")
    else:
        print("[result]   FWHM: could not find half-maximum crossings "
              "(increase --lens-width or check phase library coverage)")

    print("\n[done] All outputs in:", outdir)


if __name__ == "__main__":
    main()
