"""
Resonant metasurface absorption spectrum + harminv Q-factor extraction.

A single TiO2 pillar unit cell with Bloch periodic BCs is simulated twice:

  Stage 1 — Broadband DFT flux
    A wide-bandwidth Gaussian source drives the structure.  DFT flux monitors
    above and below the pillar record T(ω) and R(ω) across the visible range.
    Absorption A = 1 − T − R is derived.  The spectrum reveals resonances as
    dips in T (and peaks in A).

  Stage 2 — harminv resonance extraction
    A narrow Gaussian source is centred on the strongest absorption peak found
    in Stage 1.  MEEP's built-in harminv analyses the time-domain Ez field and
    returns resonance frequencies, decay rates, and Q-factors.

Physics: TiO2 pillars on a glass substrate support Mie resonances (magnetic
and electric dipoles) whose wavelengths are controlled by pillar geometry.
Near the Wood's anomaly wavelength (λ_W = n_sub · period) guided-mode
resonances couple to diffraction orders, producing sharp Fano features.

Usage
-----
    # Quick preview (low resolution)
    python run.py 04_absorption/absorption_sim.py

    # Accurate run
    python run.py 04_absorption/absorption_sim.py --resolution 64

    # Scan the full visible range with a custom geometry
    python run.py 04_absorption/absorption_sim.py \\
        --period 0.40 --height 0.45 --width 0.28 --resolution 64

    # Skip harminv (broadband spectrum only)
    python run.py 04_absorption/absorption_sim.py --no-harminv

    Options:
    --material   TiO2    pillar material
    --wavelength 0.55    centre wavelength in μm (harminv search centre)
    --bw         0.30    half-bandwidth in μm for broadband source
    --period     0.35    unit cell period in μm
    --height     0.40    pillar height in μm
    --width      0.21    pillar width in μm
    --n-glass    1.5     substrate refractive index
    --resolution 32      MEEP pixels per μm
    --nfreq      150     DFT frequency samples
    --no-harminv         skip Stage 2
    --outdir     results output directory

Output
------
    results/epsilon_map.png          permittivity cross-section
    results/transmission_spectrum.png  T / R / A vs wavelength
    results/absorption.npz           raw spectrum arrays + resonance data
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.device    import print_platform_info
from utils.materials import load_material
from utils.viz       import plot_epsilon, plot_absorption_spectrum


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

WAVELENGTH  = 0.55    # μm  — harminv search centre (approx. resonance)
BW          = 0.30    # μm  — half-bandwidth of broadband source
PERIOD      = 0.35    # μm
HEIGHT      = 0.40    # μm
WIDTH       = 0.21    # μm  — 60 % fill factor
N_GLASS     = 1.5
RESOLUTION  = 32
NFREQ       = 150     # DFT frequency samples over the broadband range
MATERIAL    = "TiO2"
OUT_DIR     = "results"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Resonant absorption spectrum + harminv Q-factor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--material",   default=MATERIAL)
    p.add_argument("--wavelength", type=float, default=WAVELENGTH,
                   help="Centre wavelength in μm (harminv search centre)")
    p.add_argument("--bw",         type=float, default=BW,
                   help="Half-bandwidth of broadband source in μm")
    p.add_argument("--period",     type=float, default=PERIOD)
    p.add_argument("--height",     type=float, default=HEIGHT)
    p.add_argument("--width",      type=float, default=WIDTH)
    p.add_argument("--n-glass",    type=float, default=N_GLASS, dest="n_glass")
    p.add_argument("--resolution", type=int,   default=RESOLUTION,
                   help="Pixels/μm. 32=preview, 64=production. Runtime ∝ r³.")
    p.add_argument("--nfreq",      type=int,   default=NFREQ,
                   help="Number of DFT frequency points")
    p.add_argument("--no-harminv", action="store_true", dest="no_harminv",
                   help="Skip Stage 2 harminv run")
    p.add_argument("--no-symmetry", action="store_false", dest="symmetry",
                   default=True,
                   help="Disable Mirror(X) symmetry (on by default, ~2× speedup)")
    p.add_argument("--outdir",     default=OUT_DIR)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _cell_layout(period, height, wavelength_center):
    """Return y-coordinate landmarks for the unit cell."""
    dpml       = max(wavelength_center, 0.5)   # PML thickness ≥ 0.5 μm
    subs_thick = 1.5 * wavelength_center
    air_thick  = 1.5 * wavelength_center

    sy         = dpml + subs_thick + height + air_thick + dpml
    y_bot      = -sy / 2
    y_subs_top = y_bot + dpml + subs_thick

    y_src      = y_bot + dpml + 0.4 * wavelength_center   # source in glass
    y_refl     = y_bot + dpml + 0.8 * wavelength_center   # refl monitor in glass
    y_tran     = y_subs_top + height + 0.4 * wavelength_center  # tran monitor in air

    return dict(
        sy=sy, sx=period,
        y_bot=y_bot, y_subs_top=y_subs_top,
        y_src=y_src, y_refl=y_refl, y_tran=y_tran,
    )


def _build_geometry(lyt, pillar_width, height, pillar_medium, n_glass):
    import meep as mp
    glass = mp.Medium(index=n_glass)
    geom  = [
        mp.Block(
            size=mp.Vector3(mp.inf, lyt["y_subs_top"] - lyt["y_bot"], mp.inf),
            center=mp.Vector3(0, lyt["y_bot"] + (lyt["y_subs_top"] - lyt["y_bot"]) / 2),
            material=glass,
        )
    ]
    if pillar_width > 0:
        geom.append(
            mp.Block(
                size=mp.Vector3(pillar_width, height, mp.inf),
                center=mp.Vector3(0, lyt["y_subs_top"] + height / 2),
                material=pillar_medium,
            )
        )
    return geom


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — BROADBAND T/R SPECTRUM
# ══════════════════════════════════════════════════════════════════════════════

def run_broadband(args, pillar_medium, lyt, use_symmetry=True):
    """
    Two MEEP runs: reference (no pillar) then structure.
    Returns (freqs, T, R) arrays.
    """
    import meep as mp
    symmetries = [mp.Mirror(mp.X)] if use_symmetry else []

    fcen   = 1.0 / args.wavelength           # centre frequency  [μm⁻¹]
    # Frequency range: fcen ± df/2 in wavelength → convert to freq width
    wl_lo  = args.wavelength - args.bw       # shortest wavelength end
    wl_hi  = args.wavelength + args.bw       # longest wavelength end
    fmin   = 1.0 / max(wl_hi, 0.1)
    fmax   = 1.0 / max(wl_lo, 0.1)
    df     = fmax - fmin                     # total freq span
    fcen_src = 0.5 * (fmin + fmax)
    nfreq  = args.nfreq

    sx, sy = lyt["sx"], lyt["sy"]
    dpml   = lyt["sy"] / 2 + lyt["y_bot"]   # = dpml stored in _cell_layout

    def _make_source(freq_center, freq_width):
        return [mp.Source(
            mp.GaussianSource(frequency=freq_center, fwidth=freq_width),
            component=mp.Ez,
            center=mp.Vector3(0, lyt["y_src"]),
            size=mp.Vector3(sx, 0),
        )]

    def _make_flux_monitors(sim):
        tran = sim.add_flux(
            fcen_src, df, nfreq,
            mp.FluxRegion(center=mp.Vector3(0, lyt["y_tran"]),
                          size=mp.Vector3(sx, 0)),
        )
        refl = sim.add_flux(
            fcen_src, df, nfreq,
            mp.FluxRegion(center=mp.Vector3(0, lyt["y_refl"]),
                          size=mp.Vector3(sx, 0)),
        )
        return tran, refl

    stop = mp.stop_when_fields_decayed(
        30, mp.Ez, mp.Vector3(0, lyt["y_tran"]), 1e-8
    )

    # ── Reference run (empty cell — no pillar) ────────────────────────────────
    if mp.am_master():
        print("\n[stage1] Reference run (no pillar) ...")

    sim_ref = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        boundary_layers=[mp.PML(abs(lyt["y_bot"] + sy / 2))],
        geometry=_build_geometry(lyt, 0, args.height, pillar_medium, args.n_glass),
        sources=_make_source(fcen_src, df),
        k_point=mp.Vector3(),
        resolution=args.resolution,
        symmetries=symmetries,
        eps_averaging=True,
    )
    tran_ref, refl_ref = _make_flux_monitors(sim_ref)
    sim_ref.run(until_after_sources=stop)

    ref_tran_flux  = np.array(mp.get_fluxes(tran_ref))
    ref_refl_data  = sim_ref.get_flux_data(refl_ref)
    freqs          = np.array(mp.get_flux_freqs(tran_ref))

    if mp.am_master():
        print(f"[stage1] Reference done.  "
              f"freq range: {freqs[0]:.3f}–{freqs[-1]:.3f} μm⁻¹  "
              f"(λ {1000/freqs[-1]:.0f}–{1000/freqs[0]:.0f} nm)")

    # ── Structure run (with pillar) ────────────────────────────────────────────
    if mp.am_master():
        print("[stage1] Structure run (with pillar) ...")

    sim_str = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        boundary_layers=[mp.PML(abs(lyt["y_bot"] + sy / 2))],
        geometry=_build_geometry(lyt, args.width, args.height,
                                 pillar_medium, args.n_glass),
        sources=_make_source(fcen_src, df),
        k_point=mp.Vector3(),
        resolution=args.resolution,
        symmetries=symmetries,
        eps_averaging=True,
    )
    tran_str, refl_str = _make_flux_monitors(sim_str)
    sim_str.load_minus_flux_data(refl_str, ref_refl_data)
    sim_str.run(until_after_sources=stop)

    T = np.array(mp.get_fluxes(tran_str)) / ref_tran_flux
    R = -np.array(mp.get_fluxes(refl_str)) / ref_tran_flux
    T = np.clip(T, 0, 1)
    R = np.clip(R, 0, 1)

    # Save epsilon map from structure sim
    return sim_str, freqs, T, R


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — harminv RESONANCE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def run_harminv(args, pillar_medium, lyt, fcen_res, df_harminv=None,
                use_symmetry=True):
    """
    Narrow-band MEEP run centred at fcen_res, with harminv at the monitor.

    Returns list of dicts: [{freq, wavelength_nm, Q, decay, amplitude}, ...]
    Modes with |err| > 0.1 or Q < 5 are filtered out.
    """
    import meep as mp
    symmetries = [mp.Mirror(mp.X)] if use_symmetry else []

    if df_harminv is None:
        df_harminv = 0.4 * fcen_res   # search window: ±20% around resonance

    sx, sy = lyt["sx"], lyt["sy"]

    sources = [mp.Source(
        mp.GaussianSource(frequency=fcen_res, fwidth=df_harminv),
        component=mp.Ez,
        center=mp.Vector3(0, lyt["y_src"]),
        size=mp.Vector3(sx, 0),
    )]

    harminv = mp.Harminv(
        mp.Ez,
        mp.Vector3(0, lyt["y_tran"]),
        fcen_res,
        df_harminv,
    )

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        boundary_layers=[mp.PML(abs(lyt["y_bot"] + sy / 2))],
        geometry=_build_geometry(lyt, args.width, args.height,
                                 pillar_medium, args.n_glass),
        sources=sources,
        k_point=mp.Vector3(),
        resolution=args.resolution,
        symmetries=symmetries,
        eps_averaging=True,
    )

    sim.run(
        harminv,
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez, mp.Vector3(0, lyt["y_tran"]), 1e-9
        ),
    )

    modes = []
    for m in harminv.modes:
        if abs(m.err) > 0.1 or m.Q < 5 or m.freq <= 0:
            continue
        modes.append(dict(
            freq          = m.freq,
            wavelength_nm = 1000.0 / m.freq,
            Q             = m.Q,
            decay         = m.decay,
            amplitude     = abs(m.amp),
        ))

    # Sort by amplitude (strongest first)
    modes.sort(key=lambda x: -x["amplitude"])
    return modes


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print_platform_info()

    mat    = load_material(args.material)
    medium = mat(args.wavelength)

    print(f"\n[absorption] ── Parameters ──────────────────────────────")
    print(f"[absorption]   material   : {args.material}  n = {mat.n(args.wavelength):.3f}")
    print(f"[absorption]   period     : {args.period*1000:.0f} nm")
    print(f"[absorption]   height     : {args.height*1000:.0f} nm")
    print(f"[absorption]   width      : {args.width*1000:.0f} nm  "
          f"({args.width/args.period*100:.0f}% fill)")
    print(f"[absorption]   λ_center   : {args.wavelength*1000:.0f} nm")
    print(f"[absorption]   λ_range    : "
          f"{(args.wavelength - args.bw)*1000:.0f}–"
          f"{(args.wavelength + args.bw)*1000:.0f} nm")
    print(f"[absorption]   n_glass    : {args.n_glass}")
    print(f"[absorption]   Wood's λ   : {args.n_glass*args.period*1000:.0f} nm  "
          f"(first diff. order at λ = n·period)")
    print(f"[absorption]   resolution : {args.resolution} px/μm")

    lyt = _cell_layout(args.period, args.height, args.wavelength)

    # ── Stage 1: broadband spectrum ───────────────────────────────────────────
    sim_str, freqs, T, R = run_broadband(args, medium, lyt,
                                          use_symmetry=args.symmetry)
    A = np.clip(1.0 - T - R, 0, 1)

    try:
        import meep as mp
        if not mp.am_master():
            return
    except ImportError:
        pass

    wavelengths_nm = 1000.0 / freqs

    # Sort so wavelengths are ascending for plotting
    idx = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[idx]
    T = T[idx]; R = R[idx]; A = A[idx]

    # Summary stats
    peak_A_idx = int(np.argmax(A))
    peak_wl    = wavelengths_nm[peak_A_idx]
    print(f"\n[stage1] Peak absorption  : {A[peak_A_idx]*100:.1f}%  "
          f"at λ = {peak_wl:.1f} nm")
    print(f"[stage1] T at peak        : {T[peak_A_idx]*100:.1f}%")
    print(f"[stage1] R at peak        : {R[peak_A_idx]*100:.1f}%")

    # ── Stage 2: harminv ──────────────────────────────────────────────────────
    resonances = []
    if not args.no_harminv:
        fcen_res = 1.0 / (peak_wl / 1000.0)
        print(f"\n[stage2] harminv centred at λ = {peak_wl:.1f} nm "
              f"(f = {fcen_res:.3f} μm⁻¹) ...")

        modes = run_harminv(args, medium, lyt, fcen_res,
                            use_symmetry=args.symmetry)

        if modes:
            print(f"\n[stage2] ── Resonances found ({len(modes)}) ─────────────")
            print(f"{'λ (nm)':>10}  {'f (μm⁻¹)':>10}  {'Q':>8}  {'decay':>12}  {'amp':>8}")
            print("-" * 56)
            for m in modes:
                print(f"{m['wavelength_nm']:10.2f}  "
                      f"{m['freq']:10.5f}  "
                      f"{m['Q']:8.1f}  "
                      f"{m['decay']:12.4e}  "
                      f"{m['amplitude']:8.4f}")
            resonances = [
                dict(wavelength_nm=m["wavelength_nm"],
                     Q=m["Q"],
                     label=f"λ={m['wavelength_nm']:.0f} nm  Q={m['Q']:.0f}")
                for m in modes
            ]
        else:
            print("[stage2] No clean resonances found in the harminv window.")
            print("[stage2] TIP: try --wavelength closer to a T dip, "
                  "or increase --resolution.")

    # ── Epsilon map ───────────────────────────────────────────────────────────
    plot_epsilon(
        sim_str,
        filename=os.path.join(outdir, "epsilon_map.png"),
        title=(f"Permittivity  |  {args.material}  "
               f"period={args.period*1000:.0f} nm  "
               f"h={args.height*1000:.0f} nm  "
               f"w={args.width*1000:.0f} nm"),
    )

    # ── Spectrum plot ─────────────────────────────────────────────────────────
    plot_absorption_spectrum(
        wavelengths_nm, T, R, A,
        resonances=resonances if resonances else None,
        filename=os.path.join(outdir, "transmission_spectrum.png"),
    )

    # ── Save raw data ─────────────────────────────────────────────────────────
    np.savez(
        os.path.join(outdir, "absorption.npz"),
        wavelengths_nm = wavelengths_nm,
        freqs          = 1000.0 / wavelengths_nm,
        T              = T,
        R              = R,
        A              = A,
        resonances     = np.array(
            [(r["wavelength_nm"], r["Q"]) for r in resonances],
            dtype=[("wavelength_nm", float), ("Q", float)],
        ) if resonances else np.array([]),
        period         = args.period,
        height         = args.height,
        width          = args.width,
        material       = args.material,
        wavelength_center = args.wavelength,
    )
    print(f"\n[post] Saved → {outdir}/absorption.npz")

    print("\n[done] All outputs in:", outdir)


if __name__ == "__main__":
    main()
