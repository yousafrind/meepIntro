"""
RCWA unit-cell phase/amplitude sweep using torcwa.

Performs the same pillar-width sweep as 01_beam_steering/unit_cell_sweep.py but
using Rigorous Coupled-Wave Analysis (RCWA) instead of MEEP FDTD.  RCWA expands
fields as Fourier harmonics in the lateral directions and propagates them through
the layer stack analytically — no time-stepping, no PML.  It is orders of
magnitude faster than FDTD for periodic structures.

The output is a PhaseLibrary-compatible .npz file that can be loaded with
PhaseLibrary.load() and used in any Phase 1 metasurface sim.

Usage
-----
    # Default geometry (matches 01_beam_steering defaults)
    python run.py 05_solver_comparison/rcwa_sim.py

    # Custom geometry
    python run.py 05_solver_comparison/rcwa_sim.py \\
        --period 0.30 --height 0.70 --n-widths 80

    # Higher Fourier truncation for better convergence
    python run.py 05_solver_comparison/rcwa_sim.py --fourier-order 20

    Note: single-process only.  Recommend: MEEP_NPROCS=1 python run.py ...

Options
-------
    --material       TiO2     pillar material (bundled name or n,k file path)
    --wavelength     0.532    free-space wavelength in μm
    --period         0.25     unit cell period in μm
    --height         0.60     pillar height in μm
    --n-glass        1.5      substrate refractive index
    --n-widths       50       number of width samples
    --width-min      0.05     min pillar width as fraction of period
    --width-max      0.90     max pillar width as fraction of period
    --fourier-order  15       RCWA Fourier truncation order N (total = (2N+1)²)
    --geo-resolution 128      spatial grid points per axis for geometry mask
    --outdir         results  output directory
    --no-plot                 skip diagnostic plot

Output
------
    results/rcwa_phase_library.npz   PhaseLibrary-compatible data + sweep_time_s
    results/rcwa_phase_library.png   4-panel diagnostic plot (|T| and ∠T)
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.device    import print_platform_info
from utils.materials import load_material
from utils.sweep     import PhaseLibrary
from utils.viz       import plot_phase_library


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

WAVELENGTH     = 0.532   # μm
PERIOD         = 0.25    # μm
HEIGHT         = 0.60    # μm
N_GLASS        = 1.5
N_WIDTHS       = 50
WIDTH_MIN_FRAC = 0.05
WIDTH_MAX_FRAC = 0.90
FOURIER_ORDER  = 15      # N → total plane waves = (2N+1)²
GEO_RESOLUTION = 128     # spatial grid pts per axis for pillar mask
MATERIAL       = "TiO2"
OUT_DIR        = "results"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="RCWA unit-cell sweep → PhaseLibrary (torcwa).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--material",       default=MATERIAL)
    p.add_argument("--wavelength",     type=float, default=WAVELENGTH,
                   help="Free-space wavelength in μm")
    p.add_argument("--period",         type=float, default=PERIOD)
    p.add_argument("--height",         type=float, default=HEIGHT)
    p.add_argument("--n-glass",        type=float, default=N_GLASS, dest="n_glass")
    p.add_argument("--n-widths",       type=int,   default=N_WIDTHS, dest="n_widths")
    p.add_argument("--width-min",      type=float, default=WIDTH_MIN_FRAC,
                   dest="width_min", help="Min width as fraction of period")
    p.add_argument("--width-max",      type=float, default=WIDTH_MAX_FRAC,
                   dest="width_max", help="Max width as fraction of period")
    p.add_argument("--fourier-order",  type=int,   default=FOURIER_ORDER,
                   dest="fourier_order",
                   help="Fourier truncation order N; total orders = (2N+1)²")
    p.add_argument("--geo-resolution", type=int,   default=GEO_RESOLUTION,
                   dest="geo_resolution",
                   help="Spatial grid pts per axis for geometry mask")
    p.add_argument("--outdir",         default=OUT_DIR)
    p.add_argument("--no-plot",        action="store_true", dest="no_plot")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  RCWA SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def rcwa_sweep(
    material       = MATERIAL,
    wavelength     = WAVELENGTH,
    period         = PERIOD,
    height         = HEIGHT,
    n_glass        = N_GLASS,
    n_widths       = N_WIDTHS,
    width_min_frac = WIDTH_MIN_FRAC,
    width_max_frac = WIDTH_MAX_FRAC,
    fourier_order  = FOURIER_ORDER,
    geo_resolution = GEO_RESOLUTION,
    device         = None,
    verbose        = True,
):
    """
    Sweep TiO2 pillar width and compute transmission amplitude/phase via RCWA.

    All length arguments in μm.  Internal torcwa calculations use nm.

    Returns
    -------
    PhaseLibrary  with widths, phases, amplitudes matching the MEEP convention
    float         elapsed wall time in seconds
    """
    try:
        import torch
        import torcwa
    except ImportError as e:
        raise ImportError(
            f"torcwa is required for Phase 3.  "
            f"Install with: pip install torcwa\n(original error: {e})"
        ) from e

    if device is None:
        try:
            from utils.device import get_torch_device
            device = get_torch_device() or torch.device("cpu")
        except Exception:
            device = torch.device("cpu")

    if verbose:
        print(f"[rcwa_sim] device = {device}")

    # Load material refractive index (same path as MEEP sweep)
    mat    = load_material(material)
    n_pil  = mat.n(wavelength)
    k_pil  = mat.k(wavelength) if hasattr(mat, "k") else 0.0
    eps_pil_val = complex(n_pil**2 - k_pil**2, 2 * n_pil * k_pil)

    if verbose:
        print(f"[rcwa_sim] {material}  n={n_pil:.4f}  k={k_pil:.4g}  "
              f"eps={eps_pil_val:.4f}")

    # Convert to nm (torcwa internal unit)
    lam_nm    = wavelength * 1000.0
    period_nm = period     * 1000.0
    height_nm = height     * 1000.0

    # Configure geometry singleton (period and grid are constant across sweep)
    torcwa.rcwa_geo.Lx = period_nm
    torcwa.rcwa_geo.Ly = period_nm
    torcwa.rcwa_geo.nx = geo_resolution
    torcwa.rcwa_geo.ny = geo_resolution
    torcwa.rcwa_geo.grid()          # initialise internal grid arrays

    # Permittivity tensors (dtype = complex64 on the target device)
    dtype = torch.complex64
    eps_pil = torch.tensor(eps_pil_val, dtype=dtype, device=device)
    eps_air = torch.tensor(1.0 + 0j,   dtype=dtype, device=device)
    eps_sub = torch.tensor(n_glass**2 + 0j, dtype=dtype, device=device)

    widths = np.linspace(width_min_frac * period,
                         width_max_frac * period,
                         n_widths)

    amplitudes = np.zeros(n_widths)
    phases     = np.zeros(n_widths)

    t_start = time.perf_counter()

    for i, w in enumerate(widths):
        w_nm = w * 1000.0

        # Pillar geometry mask (1 inside pillar, 0 outside)
        geo = torcwa.rcwa_geo.rectangle(
            Wx=w_nm, Wy=w_nm,
            Cx=period_nm / 2,
            Cy=period_nm / 2,
        ).to(device=device, dtype=dtype)

        # Layer permittivity: pillar material inside, air outside
        eps_layer = geo * eps_pil + (1.0 - geo) * eps_air

        # Create RCWA simulation
        sim = torcwa.rcwa(
            freq      = 1.0 / lam_nm,
            order     = [fourier_order, fourier_order],
            L         = [period_nm, period_nm],
            dtype     = dtype,
            device    = device,
        )

        # Layer stack: glass substrate → TiO2 pillar → air (default output)
        sim.add_input_layer(eps=eps_sub)
        sim.add_layer(thickness=height_nm, eps=eps_layer)
        # Output layer is air (default)

        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
        sim.solve_global_smatrix()

        # Zeroth-order forward transmission (xx = input x → output x polarisation)
        t00 = sim.S_parameters(
            orders       = [0, 0],
            direction    = "forward",
            port         = "transmission",
            polarization = "xx",
            ref_order    = [0, 0],
        )

        amplitudes[i] = float(torch.abs(t00).cpu())
        phases[i]     = float(torch.angle(t00).cpu())

        if verbose and (i == 0 or (i + 1) % max(1, n_widths // 10) == 0):
            print(f"[rcwa_sim]   {i+1:3d}/{n_widths}  "
                  f"w={w*1000:.1f} nm  "
                  f"|T|={amplitudes[i]:.3f}  "
                  f"∠T={np.degrees(phases[i]):+.1f}°")

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\n[rcwa_sim] Sweep done in {elapsed:.1f} s  "
              f"({elapsed/n_widths:.2f} s/point)")
        coverage = abs(float(np.degrees(
            np.unwrap(phases)[-1] - np.unwrap(phases)[0]
        )))
        print(f"[rcwa_sim] Phase coverage: {coverage:.1f}°")

    lib = PhaseLibrary(
        widths       = widths,
        phases       = phases,
        amplitudes   = amplitudes,
        wavelength   = wavelength,
        period       = period,
        pillar_height= height,
        n_glass      = n_glass,
        material     = material,
    )
    return lib, elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print_platform_info()

    mat = load_material(args.material)
    print(f"\n[rcwa_sim] ── Parameters ──────────────────────────────")
    print(f"[rcwa_sim]   material      : {args.material}  "
          f"n = {mat.n(args.wavelength):.3f}")
    print(f"[rcwa_sim]   period        : {args.period*1000:.0f} nm")
    print(f"[rcwa_sim]   height        : {args.height*1000:.0f} nm")
    print(f"[rcwa_sim]   wavelength    : {args.wavelength*1000:.0f} nm")
    print(f"[rcwa_sim]   n_glass       : {args.n_glass}")
    print(f"[rcwa_sim]   n_widths      : {args.n_widths}")
    print(f"[rcwa_sim]   fourier_order : {args.fourier_order}  "
          f"(total orders = {(2*args.fourier_order+1)**2})")
    print(f"[rcwa_sim]   geo_resolution: {args.geo_resolution}")

    lib, elapsed = rcwa_sweep(
        material       = args.material,
        wavelength     = args.wavelength,
        period         = args.period,
        height         = args.height,
        n_glass        = args.n_glass,
        n_widths       = args.n_widths,
        width_min_frac = args.width_min,
        width_max_frac = args.width_max,
        fourier_order  = args.fourier_order,
        geo_resolution = args.geo_resolution,
    )

    # Save PhaseLibrary
    lib_path = os.path.join(outdir, "rcwa_phase_library.npz")
    lib.save(lib_path)

    # Append timing metadata to the .npz (outside PhaseLibrary schema)
    data = dict(np.load(lib_path, allow_pickle=True))
    data["sweep_time_s"] = np.float64(elapsed)
    np.savez(lib_path, **data)
    print(f"[rcwa_sim] Saved → {lib_path}  (sweep_time_s = {elapsed:.1f} s)")

    # Diagnostic plot (same 4-panel style as MEEP phase library)
    if not args.no_plot:
        plot_path = os.path.join(outdir, "rcwa_phase_library.png")
        plot_phase_library(
            widths_um    = lib.widths,
            amplitudes   = lib.amplitudes,
            phases_rad   = lib.phases,
            period_um    = lib.period,
            wavelength_um= lib.wavelength,
            material     = lib.material,
            filename     = plot_path,
        )

    print("\n[rcwa_sim] Done.")


if __name__ == "__main__":
    main()
