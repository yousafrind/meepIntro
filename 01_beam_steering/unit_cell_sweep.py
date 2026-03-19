"""
Unit cell phase library sweep — beam steering metasurface.

Simulates a single 2D silicon/TiO2 pillar unit cell with Bloch periodic
boundary conditions. Sweeps pillar width and records the complex
transmission coefficient at each width.

Run this ONCE to generate the phase library before running full_array_sim.py.

Usage
-----
    python run.py 01_beam_steering/unit_cell_sweep.py [options]

    --material   TiO2                  # bundled name or path to n,k file
    --wavelength 0.532                 # free-space wavelength in μm
    --period     0.25                  # unit cell period in μm
    --height     0.60                  # pillar height in μm
    --n-glass    1.5                   # substrate refractive index
    --resolution 64                    # MEEP pixels per μm
    --n-widths   50                    # number of width samples
    --outdir     results               # where to save library + plots

Output
------
    results/phase_library.npz          # data (load in full_array_sim.py)
    results/phase_library.png          # 4-panel plot

Physics notes
-------------
* 2D FDTD, TM polarisation (Ez source).
* Bloch periodic BC in x (k_point = 0 = normal incidence).
* PML absorbers at top and bottom in y.
* Complex transmission T = E_pillar / E_ref (spatial average at monitor).
* Phase is computed from angle(T), unwrapped for plotting.
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
from utils.viz       import plot_phase_library


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Unit cell sweep: build phase library for beam steering."
    )
    p.add_argument("--material",   default="TiO2",
                   help="Pillar material: bundled name (TiO2, SiO2) "
                        "or path to n,k file  [default: TiO2]")
    p.add_argument("--wavelength", type=float, default=0.532,
                   help="Free-space wavelength in μm  [default: 0.532]")
    p.add_argument("--period",     type=float, default=0.25,
                   help="Unit cell period in μm  [default: 0.25]")
    p.add_argument("--height",     type=float, default=0.60,
                   help="Pillar height in μm  [default: 0.60]")
    p.add_argument("--n-glass",    type=float, default=1.5,
                   help="Substrate refractive index  [default: 1.5]")
    p.add_argument("--resolution", type=int,   default=64,
                   help="MEEP resolution in pixels/μm  [default: 64]")
    p.add_argument("--n-widths",   type=int,   default=50,
                   help="Number of pillar widths to sweep  [default: 50]")
    p.add_argument("--outdir",     default="results",
                   help="Output directory  [default: results]")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MEEP unit cell simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_cell(pillar_width, period, pillar_height,
                  wavelength, pillar_medium, n_glass, resolution):
    """
    2D FDTD simulation of one unit cell.

    Cell layout (y axis = propagation direction, upward):

        ┌──────────────┐  ← sy/2
        │   PML (top)  │
        ├──────────────┤  ← y_mon  (DFT monitor)
        │   air above  │
        ├──╔════╗──────┤  ← y_subs_top (pillar base = substrate top)
        │  ║ Si ║      │    pillar sits here
        ├──╚════╝──────┤  ← y_subs_top (conceptually)
        │  glass subs  │
        ├──────────────┤
        │  PML (bot)   │
        └──────────────┘  ← -sy/2
          source here ──→ y_src

    Returns
    -------
    complex : spatially-averaged Ez DFT at monitor frequency
    """
    import meep as mp

    freq   = 1.0 / wavelength
    fwidth = 0.15 * freq

    dpml       = wavelength            # PML thickness
    subs_thick = 1.5 * wavelength      # substrate layer thickness
    air_thick  = 1.5 * wavelength      # air above pillar

    sy = dpml + subs_thick + pillar_height + air_thick + dpml
    sx = period

    y_bot      = -sy / 2
    y_subs_top = y_bot + dpml + subs_thick     # pillar base
    y_src      = y_bot + dpml + 0.4 * wavelength  # source in substrate
    y_mon      = y_subs_top + pillar_height + 0.4 * wavelength  # monitor in air

    glass = mp.Medium(index=n_glass)

    geometry = [
        # Glass substrate from cell bottom to pillar base
        mp.Block(
            size=mp.Vector3(mp.inf, dpml + subs_thick, mp.inf),
            center=mp.Vector3(0, y_bot + (dpml + subs_thick) / 2),
            material=glass,
        )
    ]
    if pillar_width > 0:
        geometry.append(
            mp.Block(
                size=mp.Vector3(pillar_width, pillar_height, mp.inf),
                center=mp.Vector3(0, y_subs_top + pillar_height / 2),
                material=pillar_medium,
            )
        )

    sources = [
        mp.Source(
            mp.GaussianSource(frequency=freq, fwidth=fwidth),
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
        k_point=mp.Vector3(),     # Γ point → normal-incidence Bloch BC
        resolution=resolution,
        eps_averaging=True,
    )

    dft = sim.add_dft_fields(
        [mp.Ez], freq, freq, 1,
        center=mp.Vector3(0, y_mon),
        size=mp.Vector3(sx, 0),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ez, mp.Vector3(0, y_mon), 1e-7
        )
    )

    ez_arr = sim.get_dft_array(dft, mp.Ez, 0)
    # Spatial average = 0th Fourier order = plane-wave component
    return complex(np.mean(ez_arr))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep driver
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(args, pillar_mat):
    wavelength   = args.wavelength
    period       = args.period
    pillar_height = args.height
    n_glass      = args.n_glass
    resolution   = args.resolution
    n_widths     = args.n_widths

    pillar_medium = pillar_mat(wavelength)

    # Width range: 5% – 90% of period (avoid degenerate cases)
    widths = np.linspace(0.05 * period, 0.90 * period, n_widths)

    phases     = np.zeros(n_widths)
    amplitudes = np.zeros(n_widths)

    # ── reference simulation (no pillar) ─────────────────────────────────────
    print("\n[sweep] Reference simulation (empty unit cell) ...")
    ez_ref = run_unit_cell(
        pillar_width=0,
        period=period,
        pillar_height=pillar_height,
        wavelength=wavelength,
        pillar_medium=pillar_medium,
        n_glass=n_glass,
        resolution=resolution,
    )
    print(f"[sweep] |E_ref| = {abs(ez_ref):.4f}")

    # ── pillar sweep ──────────────────────────────────────────────────────────
    for i, w in enumerate(widths):
        print(f"\n[sweep] [{i+1:02d}/{n_widths}] width = {w*1000:.1f} nm")
        ez = run_unit_cell(
            pillar_width=w,
            period=period,
            pillar_height=pillar_height,
            wavelength=wavelength,
            pillar_medium=pillar_medium,
            n_glass=n_glass,
            resolution=resolution,
        )
        T = ez / ez_ref
        phases[i]     = np.angle(T)
        amplitudes[i] = abs(T)
        print(f"[sweep]   |T| = {amplitudes[i]:.4f}   φ = {np.degrees(phases[i]):+.1f}°")

    return widths, phases, amplitudes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve outdir relative to this script's location
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print_platform_info()
    print(f"\n[sweep] wavelength = {args.wavelength*1000:.0f} nm")
    print(f"[sweep] period     = {args.period*1000:.0f} nm")
    print(f"[sweep] height     = {args.height*1000:.0f} nm")
    print(f"[sweep] material   = {args.material}")
    print(f"[sweep] resolution = {args.resolution} px/μm")
    print(f"[sweep] n_widths   = {args.n_widths}")

    pillar_mat = load_material(args.material)

    widths, phases, amplitudes = run_sweep(args, pillar_mat)

    # Unwrap phase (removes ±π jumps, useful for plotting & interpolation)
    phases_unwrapped = np.unwrap(phases)

    # ── save library ──────────────────────────────────────────────────────────
    lib_path = os.path.join(outdir, "phase_library.npz")
    np.savez(
        lib_path,
        widths            = widths,
        phases            = phases,               # wrapped, in [-π, π]
        phases_unwrapped  = phases_unwrapped,
        amplitudes        = amplitudes,
        wavelength        = args.wavelength,
        period            = args.period,
        pillar_height     = args.height,
        n_glass           = args.n_glass,
        material          = args.material,
    )
    print(f"\n[sweep] Phase library saved → {lib_path}")

    # Phase coverage check
    phase_range = phases_unwrapped[-1] - phases_unwrapped[0]
    print(f"[sweep] Phase coverage = {np.degrees(phase_range):.0f}°  "
          f"({'GOOD: ≥ 300°' if abs(phase_range) >= 300*np.pi/180 else 'LOW — consider increasing height'})")

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_phase_library(
        widths, phases_unwrapped, amplitudes,
        period=args.period,
        wavelength=args.wavelength,
        filename=os.path.join(outdir, "phase_library.png"),
    )

    print("\n[sweep] Done.")


if __name__ == "__main__":
    main()
