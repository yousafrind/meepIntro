"""
Metasurface hologram — Gerchberg-Saxton design + MEEP FDTD validation.

A phase-only hologram encodes a target multi-spot farfield pattern into the
pillar phase distribution.  The Gerchberg-Saxton (GS) algorithm iteratively
finds the hologram phase that reconstructs the desired farfield amplitude.

Workflow
--------
1. Load the phase library from 01_beam_steering/unit_cell_sweep.py.
2. Define target farfield: intensity peaks at user-specified angles.
3. Run GS phase retrieval → hologram phase per pillar.
4. Assign pillar widths via nearest-neighbour lookup in the library.
5. Run 2D FDTD (MEEP, TM polarisation).
6. Extract DFT near-field → farfield via FFT.
7. Compare simulated farfield to GS target.

Usage
-----
    # Step 1: build phase library (once)
    python run.py 01_beam_steering/unit_cell_sweep.py

    # Step 2: simulate the hologram
    python run.py 03_holography/hologram_sim.py [options]

    --wavelength    0.532          free-space wavelength in μm
    --holo-width    10.0           hologram aperture in μm
    --pillar-h      0.60           pillar height in μm (must match sweep)
    --n-glass       1.5            substrate refractive index
    --resolution    32             MEEP pixels per μm
    --target-angles -30 30         farfield target angles in degrees
                                   (pass space-separated, e.g. -30 0 30)
    --n-gs          100            Gerchberg-Saxton iterations
    --lib-path      auto           path to phase_library.npz
    --outdir        results        output directory

Output
------
    results/epsilon_map.png        permittivity cross-section
    results/hologram_layout.png    pillar widths + GS phase profile
    results/hologram_farfield.png  simulated farfield vs target markers
    results/hologram_comparison.png  combined farfield + phase panel
    results/hologram.npz           raw data arrays

Physics notes
-------------
* Phase-only hologram: each pillar imparts a phase delay only (|T| ≈ 1).
  Amplitude control would require independent control of pillar height AND
  width — left for Phase 4 (inverse-design).
* GS algorithm convergence: the efficiency (power in target spots / total)
  typically saturates after ~50 iterations for a small number of spots.
* Quantisation penalty: the continuous GS phases are rounded to the nearest
  library phase, reducing efficiency.  Better 2π coverage → lower penalty.
* MEEP vs GS discrepancy: GS assumes ideal unit-amplitude pillars.  Actual
  pillar amplitudes deviate from 1, adding a residual background.
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
from utils.viz       import (plot_epsilon, plot_pillar_layout,
                              plot_hologram_comparison)


# ══════════════════════════════════════════════════════════════════════════════
#  USER PARAMETERS  ← edit these or override via CLI flags
# ══════════════════════════════════════════════════════════════════════════════

WAVELENGTH    = 0.532        # μm
HOLO_WIDTH    = 10.0         # μm  — hologram aperture
PILLAR_H      = 0.60         # μm  — must match unit_cell_sweep
N_GLASS       = 1.5          #     — substrate refractive index
RESOLUTION    = 32           #     — MEEP pixels/μm
TARGET_ANGLES = [-30.0, 30.0]  # °  — farfield target spot angles
N_GS          = 100          #     — Gerchberg-Saxton iterations
LIB_PATH      = "auto"       #     — "auto" → 01_beam_steering/results/
OUT_DIR       = "results"

# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Metasurface hologram: GS design + MEEP validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wavelength",    type=float, default=WAVELENGTH)
    p.add_argument("--holo-width",    type=float, default=HOLO_WIDTH,
                   dest="holo_width", help="Hologram aperture in μm")
    p.add_argument("--pillar-h",      type=float, default=PILLAR_H,
                   dest="pillar_h")
    p.add_argument("--n-glass",       type=float, default=N_GLASS,
                   dest="n_glass")
    p.add_argument("--resolution",    type=int,   default=RESOLUTION)
    p.add_argument("--target-angles", type=float, nargs="+",
                   default=TARGET_ANGLES, dest="target_angles",
                   help="Farfield target angles in degrees (space-separated)")
    p.add_argument("--n-gs",          type=int,   default=N_GS,
                   dest="n_gs", help="Gerchberg-Saxton iterations")
    p.add_argument("--lib-path",      default=LIB_PATH, dest="lib_path")
    p.add_argument("--outdir",        default=OUT_DIR)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Phase library
# ─────────────────────────────────────────────────────────────────────────────

def load_phase_library(lib_path_arg):
    if lib_path_arg == "auto":
        path = os.path.normpath(os.path.join(
            _HERE, "..", "01_beam_steering", "results", "phase_library.npz"
        ))
    else:
        path = lib_path_arg
    if not os.path.exists(path):
        sys.exit(
            f"\n[error] Phase library not found: {path}\n"
            "        Run unit_cell_sweep.py first:\n"
            "          python run.py 01_beam_steering/unit_cell_sweep.py\n"
        )
    data = np.load(path, allow_pickle=True)
    print(f"[library] Loaded: {path}")
    return data


def assign_pillar_widths(lib, target_phases):
    """Nearest-neighbour phase → width lookup. Same as 01 and 02."""
    w_lib       = lib["widths"]
    ph_lib_norm = lib["phases"] % (2 * np.pi)

    assigned = np.zeros(len(target_phases))
    errors   = np.zeros(len(target_phases))

    for i, phi in enumerate(target_phases):
        phi_norm = phi % (2 * np.pi)
        diff     = np.abs(ph_lib_norm - phi_norm)
        diff     = np.minimum(diff, 2 * np.pi - diff)
        idx      = int(np.argmin(diff))
        assigned[i] = w_lib[idx]
        errors[i]   = np.degrees(diff[idx])

    return assigned, errors


# ─────────────────────────────────────────────────────────────────────────────
# Gerchberg-Saxton phase retrieval
# ─────────────────────────────────────────────────────────────────────────────

def make_target_intensity(n_cells, period, wavelength, target_angles_deg):
    """
    Build a 1D target intensity array on the N_cells FFT grid.

    Each target angle maps to the nearest FFT bin:
        bin = round( sin(θ) · W / λ )   where W = n_cells · period

    Bins outside the propagating range (|sin θ| > 1) are silently skipped.

    Returns
    -------
    target : ndarray shape (n_cells,), values in {0, 1}
    bin_angles : list of actual angles achieved (degrees)
    """
    W      = n_cells * period
    target = np.zeros(n_cells)
    bin_angles = []

    for ang in target_angles_deg:
        sin_th = np.sin(np.radians(ang))
        if abs(sin_th) > 1.0:
            print(f"[GS] WARNING: angle {ang:.1f}° is evanescent — skipped.")
            continue
        n_bin      = sin_th * W / wavelength
        idx        = int(round(n_bin)) % n_cells
        target[idx] = 1.0
        actual_ang = np.degrees(np.arcsin(round(n_bin) * wavelength / W))
        bin_angles.append(actual_ang)

    return target, bin_angles


def gerchberg_saxton(target_intensity, n_cells, n_iter=100, seed=42):
    """
    1D phase-only Gerchberg-Saxton algorithm.

    Alternates between the hologram plane (phase-only constraint) and the
    farfield (amplitude constraint = sqrt(target_intensity)).

    Parameters
    ----------
    target_intensity : ndarray (n_cells,)  desired |farfield|²
    n_cells          : int  number of hologram pixels (pillars)
    n_iter           : int  number of GS iterations
    seed             : int  random seed for initial phase

    Returns
    -------
    phases    : ndarray (n_cells,)  hologram phases in radians ∈ [-π, π]
    eff_curve : ndarray (n_iter,)   efficiency vs iteration (for diagnostics)
    """
    rng        = np.random.default_rng(seed)
    target_amp = np.sqrt(target_intensity)
    total_tgt  = target_intensity.sum()

    # Initialise: random phases, unit amplitude
    hologram = np.exp(1j * rng.uniform(0, 2 * np.pi, n_cells))

    eff_curve = np.zeros(n_iter)

    for it in range(n_iter):
        # ── Forward: hologram plane → farfield ───────────────────────────────
        farfield = np.fft.fft(hologram)

        # Efficiency metric: fraction of power in target bins
        ff_int      = np.abs(farfield) ** 2
        eff_curve[it] = ff_int[target_intensity > 0].sum() / (ff_int.sum() + 1e-30)

        # ── Apply farfield amplitude constraint ───────────────────────────────
        ff_phase     = np.angle(farfield)
        farfield_new = target_amp * np.exp(1j * ff_phase)

        # ── Backward: farfield → hologram plane ──────────────────────────────
        h_back = np.fft.ifft(farfield_new)

        # ── Apply hologram constraint: phase-only (unit amplitude) ────────────
        hologram = np.exp(1j * np.angle(h_back))

    return np.angle(hologram), eff_curve


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builder  (identical pattern to 01 and 02)
# ─────────────────────────────────────────────────────────────────────────────

def build_geometry(x_positions, pillar_widths, pillar_height,
                   period, pillar_medium, n_glass, sy, y_subs_top):
    import meep as mp
    glass = mp.Medium(index=n_glass)
    geom  = [
        mp.Block(
            size=mp.Vector3(mp.inf, y_subs_top + sy / 2, mp.inf),
            center=mp.Vector3(0, -(sy / 2 - (y_subs_top + sy / 2) / 2)),
            material=glass,
        )
    ]
    y_ctr = y_subs_top + pillar_height / 2
    for x0, w in zip(x_positions, pillar_widths):
        if w > 0.02 * period:
            geom.append(
                mp.Block(
                    size=mp.Vector3(w, pillar_height, mp.inf),
                    center=mp.Vector3(x0, y_ctr),
                    material=pillar_medium,
                )
            )
    return geom


# ─────────────────────────────────────────────────────────────────────────────
# Farfield from DFT near-field  (same as 01_beam_steering/full_array_sim.py)
# ─────────────────────────────────────────────────────────────────────────────

def angular_spectrum(ez_dft, sx, wavelength, pad_factor=8):
    """FFT-based angular spectrum from 1D DFT near-field."""
    Nx     = len(ez_dft)
    dx     = sx / Nx
    Ez_fft = np.fft.fftshift(np.fft.fft(ez_dft, n=Nx * pad_factor))
    kx     = np.fft.fftshift(np.fft.fftfreq(Nx * pad_factor, d=dx))
    kx_rad = 2 * np.pi * kx
    k0_rad = 2 * np.pi / wavelength
    sin_th = kx_rad / k0_rad
    prop   = np.abs(sin_th) <= 1.0
    theta  = np.degrees(np.arcsin(sin_th[prop]))
    intens = np.abs(Ez_fft[prop]) ** 2
    order  = np.argsort(theta)
    return theta[order], intens[order]


# ─────────────────────────────────────────────────────────────────────────────
# MEEP simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_sim(args, lib, hologram_phases, x_positions, outdir):
    import meep as mp

    wavelength = args.wavelength
    pillar_h   = args.pillar_h
    n_glass    = args.n_glass
    resolution = args.resolution
    period     = float(lib["period"])
    freq       = 1.0 / wavelength

    sx = len(x_positions) * period

    mat_name      = str(lib["material"])
    pillar_mat    = load_material(mat_name)
    pillar_medium = pillar_mat(wavelength)

    pillar_widths, phase_errors = assign_pillar_widths(lib, hologram_phases)

    print(f"\n[sim] ── Pillar assignment ─────────────────────────")
    print(f"[sim]   Width range : {pillar_widths.min()*1000:.1f} – "
                                  f"{pillar_widths.max()*1000:.1f} nm")
    print(f"[sim]   Phase error : mean {phase_errors.mean():.1f}°  "
                                 f"max {phase_errors.max():.1f}°")

    dpml       = wavelength
    subs_thick = 1.5 * wavelength
    air_thick  = 2.0 * wavelength
    sy         = dpml + subs_thick + pillar_h + air_thick + dpml

    y_bot      = -sy / 2
    y_subs_top = y_bot + dpml + subs_thick
    y_src      = y_bot + dpml + 0.4 * wavelength
    y_mon      = y_subs_top + pillar_h + 0.4 * wavelength

    geometry = build_geometry(
        x_positions, pillar_widths, pillar_h,
        period, pillar_medium, n_glass, sy, y_subs_top,
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
        k_point=mp.Vector3(),
        resolution=resolution,
        eps_averaging=True,
    )

    dft_mon = sim.add_dft_fields(
        [mp.Ez], freq, freq, 1,
        center=mp.Vector3(0, y_mon),
        size=mp.Vector3(sx, 0),
    )

    if mp.am_master():
        plot_epsilon(
            sim,
            filename=os.path.join(outdir, "epsilon_map.png"),
            title=f"Permittivity map  |  λ={wavelength*1000:.0f} nm  hologram",
        )

    print(f"\n[sim] ── Running MEEP (resolution={resolution}) ────")
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ez, mp.Vector3(0, y_mon), 1e-7
        )
    )

    ez_dft = sim.get_dft_array(dft_mon, mp.Ez, 0)
    return ez_dft, sx, pillar_widths, phase_errors


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_spot_efficiency(theta_deg, intensity, target_angles, window=5.0):
    """
    Fraction of total farfield power within ±window° of each target angle.

    Returns list of per-spot efficiencies and the combined efficiency.
    """
    total   = intensity.sum()
    per_spot = []
    for ang in target_angles:
        in_win = intensity[np.abs(theta_deg - ang) <= window].sum()
        per_spot.append(in_win / total * 100 if total > 0 else 0)
    combined = sum(
        intensity[np.abs(theta_deg - ang) <= window].sum()
        for ang in target_angles
    ) / total * 100 if total > 0 else 0
    return per_spot, combined


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print_platform_info()

    lib    = load_phase_library(args.lib_path)
    period = float(lib["period"])

    # Number of unit cells in the hologram aperture
    n_cells     = max(4, int(np.round(args.holo_width / period)))
    sx          = n_cells * period
    x_positions = (np.arange(n_cells) - (n_cells - 1) / 2.0) * period

    print(f"\n[hologram] ── Design parameters ─────────────────────")
    print(f"[hologram]   Wavelength     : {args.wavelength*1000:.0f} nm")
    print(f"[hologram]   Aperture       : {args.holo_width:.1f} μm  ({n_cells} pillars)")
    print(f"[hologram]   Target angles  : {args.target_angles}")
    print(f"[hologram]   GS iterations  : {args.n_gs}")

    # ── Gerchberg-Saxton phase retrieval ──────────────────────────────────────
    target_intensity, actual_angles = make_target_intensity(
        n_cells, period, args.wavelength, args.target_angles
    )
    n_spots = int(target_intensity.sum())
    print(f"\n[GS] {n_spots} target spot(s) placed at: "
          f"{[f'{a:.1f}°' for a in actual_angles]}")
    print(f"[GS] Running {args.n_gs} iterations ...")

    hologram_phases, eff_curve = gerchberg_saxton(
        target_intensity, n_cells, n_iter=args.n_gs
    )

    print(f"[GS] Final GS efficiency : {eff_curve[-1]*100:.1f}%  "
          f"(power in target bins / total)")

    # ── MEEP simulation ───────────────────────────────────────────────────────
    ez_dft, sx, pillar_widths, phase_errors = run_sim(
        args, lib, hologram_phases, x_positions, outdir
    )

    import meep as mp
    if not mp.am_master():
        return

    # ── Pillar layout ─────────────────────────────────────────────────────────
    plot_pillar_layout(
        x_positions, pillar_widths,
        pillar_height=args.pillar_h,
        period=period,
        target_phases=hologram_phases,
        filename=os.path.join(outdir, "hologram_layout.png"),
    )

    # ── Farfield ──────────────────────────────────────────────────────────────
    theta_deg, intensity = angular_spectrum(ez_dft, sx, args.wavelength)

    plot_hologram_comparison(
        theta_deg, intensity,
        target_angles_deg=actual_angles,
        hologram_phases=hologram_phases,
        x_positions=x_positions,
        wavelength=args.wavelength,
        filename=os.path.join(outdir, "hologram_comparison.png"),
    )

    # ── Efficiency summary ────────────────────────────────────────────────────
    per_spot, combined = compute_spot_efficiency(
        theta_deg, intensity, actual_angles
    )

    print(f"\n[result] ── Efficiency summary ─────────────────────")
    print(f"[result]   GS design efficiency  : {eff_curve[-1]*100:.1f}%")
    for ang, eff in zip(actual_angles, per_spot):
        print(f"[result]   MEEP spot @ {ang:+.1f}°    : {eff:.1f}%")
    print(f"[result]   Combined MEEP eff     : {combined:.1f}%  "
          f"(power within ±5° of targets)")
    if combined < 20:
        print("[result]   TIP: increase pillar height for better 2π coverage "
              "→ higher efficiency")

    # ── Save raw data ──────────────────────────────────────────────────────────
    np.savez(
        os.path.join(outdir, "hologram.npz"),
        theta_deg        = theta_deg,
        intensity        = intensity,
        hologram_phases  = hologram_phases,
        x_positions      = x_positions,
        pillar_widths    = pillar_widths,
        target_angles    = np.array(actual_angles),
        eff_curve        = eff_curve,
        wavelength       = args.wavelength,
    )
    print(f"[post] Raw data saved → {outdir}/hologram.npz")

    print("\n[done] All outputs in:", outdir)


if __name__ == "__main__":
    main()
