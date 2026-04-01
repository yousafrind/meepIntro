"""
End-to-end validation: run the achromatic surrogate design through MEEP FDTD.

Loads the optimised pillar widths from 07_broadband/achromatic_design.npz,
builds a full MEEP supercell for each wavelength, and measures the focal spot
quality (FWHM, peak intensity, Strehl ratio) via the Angular Spectrum Method.

Compares three designs at each wavelength:
  1. Achromatic (surrogate-optimised across all λ simultaneously)
  2. NN baseline (nearest-neighbour from the phase library at the central λ)
  3. Single-λ NN (nearest-neighbour optimised independently per wavelength)

This closes the design loop: surrogate → FDTD ground truth.

Usage
-----
    python run.py 08_validation/validate_design.py

    # Use a specific achromatic design file
    python run.py 08_validation/validate_design.py \\
        --design 07_broadband/results/achromatic_design.npz

    # Only validate a subset of wavelengths
    python run.py 08_validation/validate_design.py \\
        --wavelengths 0.45 0.532

    Options
    -------
    --design       07_broadband/results/achromatic_design.npz
    --lib          01_beam_steering/results/phase_library.npz  (for per-λ NN)
    --wavelengths  all     wavelengths to validate (default: all from design file)
    --resolution   32      MEEP pixels per μm (32=fast, 64=accurate)
    --n-asm        200     ASM propagation planes for field map
    --outdir       results

Output
------
    results/validation_summary.png     FWHM + Strehl comparison per λ
    results/focal_spots_<WL>nm.png     focal spot profiles at each wavelength
    results/validation.npz             all numerical results
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "02_metalens"))

from utils.device    import print_platform_info
from utils.materials import load_material
from utils.sweep     import PhaseLibrary
from utils.viz       import plot_validation_summary

# Reuse ASM and geometry helpers from metalens_sim (avoid duplication)
from metalens_sim import (
    build_geometry, propagate_asm, propagate_asm_stack, compute_fwhm
)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_DESIGN = os.path.join(
    _ROOT, "07_broadband", "results", "achromatic_design.npz"
)
_DEFAULT_LIB = os.path.join(
    _ROOT, "01_beam_steering", "results", "phase_library.npz"
)
RESOLUTION = 32
N_ASM      = 200
OUT_DIR    = "results"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate achromatic design via MEEP FDTD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--design",      default=_DEFAULT_DESIGN,
                   help="Path to achromatic_design.npz")
    p.add_argument("--lib",         default=_DEFAULT_LIB,
                   help="Path to PhaseLibrary for per-λ NN baseline")
    p.add_argument("--wavelengths", nargs="*", type=float, default=None,
                   help="Subset of wavelengths to validate (default: all)")
    p.add_argument("--resolution",  type=int, default=RESOLUTION,
                   help="MEEP pixels/μm. 32=preview, 64=production. Runtime ∝ r³.")
    p.add_argument("--n-asm",       type=int, default=N_ASM, dest="n_asm")
    p.add_argument("--outdir",      default=OUT_DIR)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MEEP FOCAL-SPOT MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_metalens_meep(widths, x_positions, wavelength, period,
                      pillar_h, n_glass, material, focal_len,
                      resolution, n_asm, label=""):
    """
    Run a single-wavelength MEEP metalens simulation with the given widths.

    Parameters
    ----------
    widths       : ndarray [N]  pillar widths in μm
    x_positions  : ndarray [N]  pillar x-coordinates in μm (centred at 0)
    wavelength   : float        free-space wavelength in μm
    period       : float        unit cell period in μm
    pillar_h     : float        pillar height in μm
    n_glass      : float        substrate refractive index
    material     : str          pillar material name
    focal_len    : float        focal length in μm
    resolution   : int          MEEP pixels/μm
    n_asm        : int          ASM propagation planes
    label        : str          description for progress prints

    Returns
    -------
    dict with: fwhm_um, fwhm_dl_um, strehl, peak_intensity,
               x_um, focal_intensity, field_2d (2D ASM map)
    """
    import meep as mp

    freq        = 1.0 / wavelength
    n_cells     = len(widths)
    sx          = n_cells * period

    pillar_mat    = load_material(material)
    pillar_medium = pillar_mat(wavelength)

    # Cell layout (matches metalens_sim.py conventions)
    dpml        = wavelength
    subs_thick  = 1.5 * wavelength
    prop_height = focal_len + 2.0 * wavelength
    sy          = dpml + subs_thick + pillar_h + prop_height + dpml

    y_bot       = -sy / 2
    y_subs_top  = y_bot + dpml + subs_thick
    y_src       = y_bot + dpml + 0.4 * wavelength
    y_near      = y_subs_top + pillar_h + 0.4 * wavelength

    if mp.am_master():
        print(f"  [{label}] λ={wavelength*1000:.0f} nm  "
              f"sx={sx:.2f} μm  sy={sy:.2f} μm  res={resolution}")

    geometry = build_geometry(
        x_positions, widths, pillar_h,
        period, pillar_medium, n_glass, sy, y_subs_top,
    )

    sources = [mp.Source(
        mp.GaussianSource(frequency=freq, fwidth=0.05 * freq),
        component=mp.Ez,
        center=mp.Vector3(0, y_src),
        size=mp.Vector3(sx, 0),
    )]

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=sources,
        k_point=mp.Vector3(),
        resolution=resolution,
        eps_averaging=True,
        symmetries=[mp.Mirror(mp.X)],   # valid for centred pillars, normal incidence
    )

    dft_near = sim.add_dft_fields(
        [mp.Ez], freq, freq, 1,
        center=mp.Vector3(0, y_near),
        size=mp.Vector3(sx, 0),
    )

    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ez, mp.Vector3(0, y_near), 1e-7
    ))

    ez_near    = sim.get_dft_array(dft_near, mp.Ez, 0)
    Nx         = len(ez_near)
    x_um       = np.linspace(-sx / 2, sx / 2, Nx)

    # ASM to focal plane
    ez_focal   = propagate_asm(ez_near, sx, wavelength, focal_len)
    focal_int  = np.abs(ez_focal) ** 2

    # 2D field map
    y_prop  = np.linspace(0.5 * wavelength, focal_len + 1.5 * wavelength, n_asm)
    stack   = propagate_asm_stack(ez_near, sx, wavelength, y_prop)
    field_2d= np.abs(stack) ** 2

    # Metrics
    fwhm_um = compute_fwhm(x_um, focal_int)
    half_d  = sx / 2.0
    na      = half_d / np.sqrt(half_d**2 + focal_len**2)
    fwhm_dl = 0.5 * wavelength / na

    peak    = float(focal_int.max())
    strehl  = float(peak / focal_int.max()) if peak > 0 else 0.0  # relative to self

    if mp.am_master():
        fwhm_str = f"{fwhm_um*1000:.0f} nm" if fwhm_um else "N/A"
        print(f"  [{label}] FWHM={fwhm_str}  DL={fwhm_dl*1000:.0f} nm  "
              f"peak={peak:.3f}")

    return dict(
        fwhm_um       = fwhm_um,
        fwhm_dl_um    = fwhm_dl,
        peak_intensity= peak,
        strehl        = strehl,
        x_um          = x_um,
        focal_intensity= focal_int,
        field_2d      = field_2d,
        y_prop_um     = y_prop,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print_platform_info()

    try:
        import meep as mp
    except ImportError:
        raise ImportError("MEEP is required for validation. conda activate meep")

    # ── Load achromatic design ────────────────────────────────────────────────
    print(f"\n[validate] Loading design: {args.design}")
    d = np.load(args.design, allow_pickle=True)

    wavelengths   = list(d["wavelengths"])
    widths_opt    = d["widths_opt"]          # [N] shared across λ
    widths_nn     = d["widths_nn"][0]        # [n_λ, N] → take first λ as central NN
    x_um          = d["x_um"]
    period        = float(d["period"])
    target_desc   = str(d["target_desc"])
    focal_len     = _infer_focal_len(target_desc)

    # Filter to requested wavelengths
    if args.wavelengths:
        wavelengths = [w for w in wavelengths if w in args.wavelengths]
        if not wavelengths:
            sys.exit("[validate] None of the requested wavelengths found in design.")

    print(f"[validate] Design     : {target_desc}")
    print(f"[validate] Wavelengths: {[f'{w*1000:.0f} nm' for w in wavelengths]}")
    print(f"[validate] N pillars  : {len(widths_opt)}  period={period*1000:.0f} nm")
    print(f"[validate] Focal len  : {focal_len:.1f} μm")
    print(f"[validate] Resolution : {args.resolution} px/μm")

    # Load phase library for per-λ NN baselines
    lib = PhaseLibrary.load(args.lib)

    # Infer geometry from the library
    pillar_h = float(lib.pillar_height)
    n_glass  = float(lib.n_glass)
    material = str(lib.material)

    # ── Validate each wavelength ──────────────────────────────────────────────
    results_ach = {}
    results_nn  = {}
    results_perwl = {}

    for wl in wavelengths:
        wl_nm = int(round(wl * 1000))
        print(f"\n[validate] ══ λ = {wl_nm} nm ══════════════════════════════")

        # Per-λ NN design (optimal NN baseline for this specific wavelength)
        target_phases = _metalens_phases(x_um, focal_len, wl)
        widths_perwl, _ = lib.assign_widths(target_phases)

        # ── Achromatic design ─────────────────────────────────────────────────
        print(f"[validate] Running achromatic design ...")
        res_ach = run_metalens_meep(
            widths_opt, x_um, wl, period, pillar_h, n_glass, material,
            focal_len, args.resolution, args.n_asm, label="achromatic"
        )
        results_ach[wl_nm] = res_ach

        # ── Central-λ NN design ────────────────────────────────────────────────
        print(f"[validate] Running central-λ NN design ...")
        res_nn = run_metalens_meep(
            widths_nn, x_um, wl, period, pillar_h, n_glass, material,
            focal_len, args.resolution, args.n_asm, label="NN central"
        )
        results_nn[wl_nm] = res_nn

        # ── Per-λ NN design ────────────────────────────────────────────────────
        print(f"[validate] Running per-λ NN design ...")
        res_perwl = run_metalens_meep(
            widths_perwl, x_um, wl, period, pillar_h, n_glass, material,
            focal_len, args.resolution, args.n_asm, label="NN per-λ"
        )
        results_perwl[wl_nm] = res_perwl

        # Per-wavelength focal spot plot (master only)
        if mp.am_master():
            _plot_focal_comparison(
                wl_nm, res_ach, res_nn, res_perwl,
                outdir=outdir,
            )

    if not mp.am_master():
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    _print_summary_table(wavelengths, results_ach, results_nn, results_perwl)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        os.path.join(outdir, "validation.npz"),
        wavelengths      = np.array(wavelengths),
        fwhm_ach         = np.array([results_ach[int(w*1000)]["fwhm_um"] or np.nan
                                     for w in wavelengths]),
        fwhm_nn          = np.array([results_nn[int(w*1000)]["fwhm_um"]  or np.nan
                                     for w in wavelengths]),
        fwhm_perwl       = np.array([results_perwl[int(w*1000)]["fwhm_um"] or np.nan
                                     for w in wavelengths]),
        fwhm_dl          = np.array([results_ach[int(w*1000)]["fwhm_dl_um"]
                                     for w in wavelengths]),
        peak_ach         = np.array([results_ach[int(w*1000)]["peak_intensity"]
                                     for w in wavelengths]),
        peak_nn          = np.array([results_nn[int(w*1000)]["peak_intensity"]
                                     for w in wavelengths]),
        peak_perwl       = np.array([results_perwl[int(w*1000)]["peak_intensity"]
                                     for w in wavelengths]),
    )
    print(f"\n[validate] Saved → {outdir}/validation.npz")

    # Summary plot
    plot_validation_summary(
        wavelengths  = wavelengths,
        results_ach  = results_ach,
        results_nn   = results_nn,
        results_perwl= results_perwl,
        filename     = os.path.join(outdir, "validation_summary.png"),
    )

    print(f"\n[validate] Done.  Outputs in: {outdir}")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _metalens_phases(x_um, focal_len, wavelength):
    k0 = 2 * np.pi / wavelength
    return k0 * (focal_len - np.sqrt(x_um**2 + focal_len**2))


def _infer_focal_len(target_desc):
    """Parse focal length from target_desc string like 'metalens  f=10 μm'."""
    import re
    m = re.search(r"f\s*=\s*([\d.]+)", target_desc)
    if m:
        return float(m.group(1))
    return 10.0  # fallback


def _plot_focal_comparison(wl_nm, res_ach, res_nn, res_perwl, outdir):
    """Three-curve focal-spot overlay at one wavelength."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Validation  |  λ = {wl_nm} nm", fontsize=13)

    x_ach = res_ach["x_um"] * 1000
    x_nn  = res_nn["x_um"]  * 1000
    x_pw  = res_perwl["x_um"] * 1000

    # Normalise each to its own peak for shape comparison
    def norm(arr):
        mx = arr.max()
        return arr / mx if mx > 0 else arr

    ax = axes[0]
    ax.plot(x_ach, norm(res_ach["focal_intensity"]),    "b-",   lw=2,
            label=f"Achromatic  FWHM="
                  f"{res_ach['fwhm_um']*1000:.0f} nm" if res_ach["fwhm_um"]
                  else "Achromatic")
    ax.plot(x_nn,  norm(res_nn["focal_intensity"]),     "r--",  lw=1.5,
            label=f"NN central-λ  FWHM="
                  f"{res_nn['fwhm_um']*1000:.0f} nm" if res_nn["fwhm_um"]
                  else "NN central-λ")
    ax.plot(x_pw,  norm(res_perwl["focal_intensity"]),  "g-.",  lw=1.5,
            label=f"NN per-λ  FWHM="
                  f"{res_perwl['fwhm_um']*1000:.0f} nm" if res_perwl["fwhm_um"]
                  else "NN per-λ")
    dl_nm = res_ach["fwhm_dl_um"] * 1000
    ax.axvline(-dl_nm / 2, color="k", lw=0.8, ls=":", alpha=0.5)
    ax.axvline( dl_nm / 2, color="k", lw=0.8, ls=":", alpha=0.5,
                label=f"DL = {dl_nm:.0f} nm")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("|Ez|² (normalised)")
    ax.set_title("Focal-plane intensity (normalised)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2D field map for achromatic design
    ax = axes[1]
    field = res_ach["field_2d"]
    y_um  = res_ach["y_prop_um"]
    x_plot= res_ach["x_um"] * 1000
    im = ax.imshow(
        field / field.max(),
        extent=[x_plot[0], x_plot[-1], y_um[0], y_um[-1]],
        origin="lower", aspect="auto", cmap="inferno",
    )
    plt.colorbar(im, ax=ax, label="|Ez|² (normalised)")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y above pillars (μm)")
    ax.set_title("Achromatic — 2D field map")

    plt.tight_layout()
    fname = os.path.join(outdir, f"focal_spots_{wl_nm}nm.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[validate] Saved: {fname}")


def _print_summary_table(wavelengths, results_ach, results_nn, results_perwl):
    header = (f"{'λ (nm)':>8}  {'Design':>14}  "
              f"{'FWHM (nm)':>10}  {'DL (nm)':>8}  {'FWHM/DL':>8}  "
              f"{'Peak':>8}")
    sep = "─" * len(header)
    print(f"\n[validate] ── Focal Spot Summary {'─'*30}")
    print(header); print(sep)

    def row(wl_nm, label, res):
        fwhm = res["fwhm_um"]
        dl   = res["fwhm_dl_um"]
        fwhm_str = f"{fwhm*1000:.0f}" if fwhm else " N/A"
        ratio    = f"{fwhm/dl:.2f}"   if fwhm else "  — "
        print(f"{wl_nm:>8}  {label:>14}  "
              f"{fwhm_str:>10}  {dl*1000:>8.0f}  {ratio:>8}  "
              f"{res['peak_intensity']:>8.3f}")

    for wl in wavelengths:
        wl_nm = int(round(wl * 1000))
        row(wl_nm, "Achromatic",   results_ach[wl_nm])
        row("",    "NN central-λ", results_nn[wl_nm])
        row("",    "NN per-λ",     results_perwl[wl_nm])
        print(sep)


if __name__ == "__main__":
    main()
