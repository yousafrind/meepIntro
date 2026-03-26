"""
Shared plotting utilities for MEEP metasurface simulations.

All plots save to disk (no interactive display — works in WSL).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display required
import matplotlib.pyplot as plt


def _ensure_dir(filepath):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Phase library plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_phase_library(widths, phases, amplitudes, period, wavelength, filename):
    """
    Plot phase and amplitude vs pillar width from unit cell sweep.

    Parameters
    ----------
    widths      : array  pillar widths in μm
    phases      : array  phase in radians (may be unwrapped)
    amplitudes  : array  |T| transmission amplitude
    period      : float  unit cell period in μm
    wavelength  : float  free-space wavelength in μm
    filename    : str    output PNG path
    """
    _ensure_dir(filename)
    widths_nm = np.asarray(widths) * 1000
    fill_frac  = np.asarray(widths) / period

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Unit Cell Phase Library  |  λ = {wavelength*1000:.0f} nm  "
        f"|  period = {period*1000:.0f} nm",
        fontsize=13
    )

    # Phase vs width (nm)
    ax = axes[0, 0]
    ax.plot(widths_nm, np.degrees(phases), "b-o", ms=4, lw=1.5)
    ax.set_xlabel("Pillar width (nm)")
    ax.set_ylabel("Phase (degrees)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Transmission phase")

    # Amplitude vs width (nm)
    ax = axes[0, 1]
    ax.plot(widths_nm, amplitudes, "r-o", ms=4, lw=1.5)
    ax.set_xlabel("Pillar width (nm)")
    ax.set_ylabel("|T|")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Transmission amplitude")

    # Phase vs fill fraction
    ax = axes[1, 0]
    ax.plot(fill_frac * 100, np.degrees(phases), "b-o", ms=4, lw=1.5)
    ax.set_xlabel("Fill fraction (%)")
    ax.set_ylabel("Phase (degrees)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Phase vs fill fraction")

    # Scatter: amplitude vs phase
    ax = axes[1, 1]
    sc = ax.scatter(np.degrees(phases), amplitudes,
                    c=widths_nm, cmap="viridis", s=40, zorder=3)
    plt.colorbar(sc, ax=ax, label="Pillar width (nm)")
    ax.set_xlabel("Phase (degrees)")
    ax.set_ylabel("|T|")
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Amplitude–phase space")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Farfield / angular spectrum plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_angular_spectrum(theta_deg, intensity, target_angle, actual_angle,
                          wavelength, filename):
    """
    Plot angular spectrum (farfield intensity vs angle).

    Parameters
    ----------
    theta_deg    : array  angles in degrees
    intensity    : array  |E|² at each angle
    target_angle : float  user-requested steering angle (degrees)
    actual_angle : float  achievable angle given discrete unit cells (degrees)
    wavelength   : float  free-space wavelength in μm
    filename     : str    output PNG path
    """
    _ensure_dir(filename)
    norm = intensity / intensity.max() if intensity.max() > 0 else intensity

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Beam Steering Farfield  |  λ = {wavelength*1000:.0f} nm  "
        f"|  target = {target_angle:.1f}°  |  actual = {actual_angle:.1f}°"
    )

    # Linear plot
    ax1.plot(theta_deg, norm, "b-", lw=1.5)
    ax1.axvline(actual_angle,  color="r", ls="--", lw=1.2,
                label=f"Actual {actual_angle:.1f}°")
    ax1.axvline(-actual_angle, color="r", ls=":",  lw=0.8, alpha=0.5)
    ax1.axvline(0, color="k", ls=":", lw=0.8, alpha=0.4, label="Normal (0°)")
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Normalised intensity")
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Linear scale")

    # Log plot
    eps = 1e-4
    ax2.semilogy(theta_deg, norm + eps, "b-", lw=1.5)
    ax2.axvline(actual_angle,  color="r", ls="--", lw=1.2,
                label=f"Actual {actual_angle:.1f}°")
    ax2.axvline(0, color="k", ls=":", lw=0.8, alpha=0.4)
    ax2.set_xlabel("Angle (degrees)")
    ax2.set_ylabel("Normalised intensity (log)")
    ax2.set_xlim(-90, 90)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Log scale")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Near-field plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_fields(sim, component, filename, title=""):
    """
    Save a 2D field plot from a MEEP simulation object.
    Only call from MPI rank 0 (MEEP handles this internally).
    """
    import meep as mp
    _ensure_dir(filename)
    plt.figure(figsize=(9, 6))
    sim.plot2D(fields=component)
    if title:
        plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


def plot_epsilon(sim, filename, title="Permittivity"):
    """Save permittivity map from a MEEP simulation object."""
    _ensure_dir(filename)
    plt.figure(figsize=(9, 6))
    sim.plot2D()
    if title:
        plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Pillar layout visualisation
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Metalens focal spot plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_focal_spot(x_um, intensity, fwhm_um, diffraction_limit_um,
                   focal_len, wavelength, filename):
    """
    Plot the intensity profile at the focal plane of a metalens.

    Parameters
    ----------
    x_um               : array  x positions in μm
    intensity          : array  |Ez|² at focal plane
    fwhm_um            : float or None  measured FWHM in μm
    diffraction_limit_um : float  ideal FWHM = 0.5λ/NA in μm
    focal_len          : float  focal length in μm
    wavelength         : float  free-space wavelength in μm
    filename           : str    output PNG path
    """
    _ensure_dir(filename)
    norm = intensity / intensity.max() if intensity.max() > 0 else intensity

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_um, norm, "b-", lw=1.8, label="FDTD (MEEP)")
    ax.axhline(0.5, color="gray", ls="--", lw=1.0, label="Half-maximum")

    title_parts = [f"Focal Spot  |  λ = {wavelength*1000:.0f} nm",
                   f"f = {focal_len:.1f} μm"]
    if fwhm_um is not None:
        ax.axvspan(-fwhm_um/2, fwhm_um/2, alpha=0.12, color="blue",
                   label=f"FWHM = {fwhm_um*1000:.0f} nm")
        title_parts.append(f"FWHM = {fwhm_um*1000:.0f} nm  "
                           f"(limit = {diffraction_limit_um*1000:.0f} nm)")

    ax.set_xlabel("x (μm)")
    ax.set_ylabel("Normalised |Ez|²")
    ax.set_title("  |  ".join(title_parts))
    ax.set_xlim(x_um[0], x_um[-1])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Hologram plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_hologram_comparison(theta_deg, intensity, target_angles_deg,
                             hologram_phases, x_positions,
                             wavelength, filename):
    """
    Two-panel plot comparing the simulated hologram farfield to the target.

    Parameters
    ----------
    theta_deg        : array  angles in degrees from MEEP simulation
    intensity        : array  |Ez|² at each angle
    target_angles_deg: list   target spot angles in degrees
    hologram_phases  : array  GS-designed phase at each pillar (radians)
    x_positions      : array  pillar x-coordinates in μm
    wavelength       : float  free-space wavelength in μm
    filename         : str    output PNG path
    """
    _ensure_dir(filename)
    norm = intensity / intensity.max() if intensity.max() > 0 else intensity

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Metasurface Hologram  |  λ = {wavelength*1000:.0f} nm  "
        f"|  targets: {[f'{a:.0f}°' for a in target_angles_deg]}"
    )

    # Farfield comparison
    ax1.plot(theta_deg, norm, "b-", lw=1.5, label="FDTD (MEEP)")
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(target_angles_deg)))
    for ang, col in zip(target_angles_deg, colors):
        ax1.axvline(ang, color=col, ls="--", lw=1.3, label=f"Target {ang:.0f}°")
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Normalised intensity")
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Farfield (simulated vs targets)")

    # Hologram phase profile
    ax2.bar(x_positions, np.degrees(hologram_phases % (2 * np.pi)),
            width=(x_positions[1] - x_positions[0]) * 0.85,
            color="steelblue", alpha=0.75)
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("Assigned phase (°)")
    ax2.set_ylim(0, 370)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Hologram phase profile (GS design)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


def plot_field_propagation(x_um, y_um, intensity_2d,
                           focal_len, y_pillar_top, filename):
    """
    2D false-colour plot of |Ez|² showing beam focusing above the metalens.

    Parameters
    ----------
    x_um          : 1D array  x positions in μm (horizontal axis)
    y_um          : 1D array  propagation y positions in μm (vertical axis)
    intensity_2d  : 2D array  shape (len(y_um), len(x_um))  —  |Ez|²
    focal_len     : float     focal length in μm (draws a horizontal marker)
    y_pillar_top  : float     y coordinate of pillar tops in μm
    filename      : str       output PNG path
    """
    _ensure_dir(filename)

    fig, ax = plt.subplots(figsize=(8, 9))

    # Normalise per-row for better dynamic range visualisation
    vmax = intensity_2d.max()
    im = ax.pcolormesh(x_um, y_um, intensity_2d / (vmax + 1e-30),
                       cmap="inferno", shading="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalised |Ez|²")

    # Mark focal plane
    y_focal = y_pillar_top + focal_len
    ax.axhline(y_focal, color="cyan", ls="--", lw=1.2,
               label=f"Focal plane  (f = {focal_len:.1f} μm)")
    ax.axhline(y_pillar_top, color="white", ls=":", lw=0.8,
               label="Pillar top")

    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title("Metalens beam propagation  |  |Ez|²")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Pillar layout visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_pillar_layout(x_positions, widths, pillar_height, period,
                       target_phases, filename):
    """
    Visual map of the metasurface: pillar widths and assigned phases.

    Parameters
    ----------
    x_positions  : array  pillar centre x-coordinates in μm
    widths        : array  pillar widths in μm
    pillar_height : float  pillar height in μm
    period        : float  unit cell period in μm
    target_phases : array  required phase per pillar in radians
    filename      : str    output PNG path
    """
    _ensure_dir(filename)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(widths)*0.25), 6),
                                   sharex=True)
    fig.suptitle("Metasurface Pillar Layout")

    cmap = plt.cm.viridis
    colors = cmap((np.asarray(widths) - widths.min()) /
                  (widths.max() - widths.min() + 1e-12))

    for x0, w, c in zip(x_positions, widths, colors):
        rect = plt.Rectangle((x0 - w/2, 0), w, pillar_height,
                              color=c, alpha=0.85, linewidth=0.5,
                              edgecolor="k")
        ax1.add_patch(rect)

    ax1.set_xlim(x_positions[0] - period, x_positions[-1] + period)
    ax1.set_ylim(0, pillar_height * 1.4)
    ax1.set_ylabel("Height (μm)")
    ax1.set_title("Pillar cross-section (coloured by width)")

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=widths.min()*1000, vmax=widths.max()*1000)
    )
    plt.colorbar(sm, ax=ax1, orientation="vertical",
                 label="Pillar width (nm)", pad=0.01)

    ax2.bar(x_positions, np.degrees(np.unwrap(target_phases)),
            width=period * 0.8, alpha=0.7, color="steelblue")
    ax2.set_xlabel("x position (μm)")
    ax2.set_ylabel("Target phase (°)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Required phase profile")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Absorption spectrum + resonance markers
# ──────────────────────────────────────────────────────────────────────────────

def plot_absorption_spectrum(wavelengths_nm, T, R, A,
                             resonances=None, filename="results/spectrum.png"):
    """
    Three-panel transmission / reflection / absorption spectrum.

    Parameters
    ----------
    wavelengths_nm : array   wavelengths in nm
    T              : array   transmission (0–1)
    R              : array   reflection   (0–1)
    A              : array   absorption = 1 − T − R
    resonances     : list of dicts with keys:
                       'wavelength_nm' : float  resonance wavelength
                       'Q'             : float  quality factor (optional)
                       'label'         : str    text label    (optional)
                     Pass None to skip resonance markers.
    filename       : str    output PNG path
    """
    _ensure_dir(filename)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle("Resonant metasurface — Transmission / Reflection / Absorption")

    panels = [
        (axes[0], T, "Transmission", "tab:blue"),
        (axes[1], R, "Reflection",   "tab:orange"),
        (axes[2], A, "Absorption",   "tab:red"),
    ]

    for ax, vals, label, color in panels:
        ax.plot(wavelengths_nm, vals, color=color, lw=1.5)
        ax.set_ylabel(label)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5, ls=":")

        if resonances:
            for res in resonances:
                wl  = res["wavelength_nm"]
                q   = res.get("Q", None)
                lbl = res.get("label", f"Q={q:.0f}" if q else f"{wl:.0f} nm")
                ax.axvline(wl, color="green", ls="--", lw=1.1, alpha=0.8)
                if ax is axes[0]:
                    ax.text(wl + 3, 0.55, lbl,
                            color="green", fontsize=8, rotation=90, va="center")

    axes[2].set_xlabel("Wavelength (nm)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark results
# ──────────────────────────────────────────────────────────────────────────────

def plot_benchmark_results(results, filename="benchmarks/results/benchmark.png"):
    """
    Multi-panel benchmark summary plot.

    Parameters
    ----------
    results : dict with optional keys:

      'resolution' : dict
          'resolutions'  : list of int    px/μm values
          'times'        : list of float  wall-clock seconds per run
          'n_cells'      : list of int    total grid cells (resolution²·sx·sy)

      'symmetry' : dict
          'resolutions'  : list of int
          'times_nosym'  : list of float
          'times_sym'    : list of float

      'mpi' : dict
          'nprocs'    : list of int
          'times'     : list of float
          'speedups'  : list of float   (t[0]/t[i])
          'efficiencies' : list of float  (speedup/nprocs, percent)

    filename : str  output PNG path
    """
    _ensure_dir(filename)
    n_panels = sum(k in results for k in ("resolution", "symmetry", "mpi"))
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("Phase 2 Performance Benchmarks", fontsize=13)

    ax_idx = 0

    # ── Resolution scaling ─────────────────────────────────────────────────
    if "resolution" in results:
        ax  = axes[ax_idx]; ax_idx += 1
        res = results["resolution"]
        rs  = res["resolutions"]
        ts  = res["times"]
        ax.plot(rs, ts, "o-", color="steelblue", lw=2, ms=7)
        ax.set_xlabel("Resolution (px/μm)")
        ax.set_ylabel("Wall time (s)")
        ax.set_title("Resolution scaling\n(single unit cell)")
        ax.grid(True, alpha=0.3)

        # Annotate with O(r³) reference line
        r0, t0 = rs[0], ts[0]
        r_ref  = np.linspace(rs[0], rs[-1], 80)
        ax.plot(r_ref, t0 * (r_ref / r0) ** 3, "k--", lw=1, alpha=0.5,
                label="O(r³) theory")
        ax.legend(fontsize=8)

        # Secondary axis: speedup vs ref
        ax2 = ax.twinx()
        speedups = [ts[0] / t for t in ts]
        ax2.plot(rs, speedups, "s--", color="orange", lw=1.2, ms=5, alpha=0.6)
        ax2.set_ylabel("Speedup vs lowest res", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

    # ── Symmetry comparison ────────────────────────────────────────────────
    if "symmetry" in results:
        ax  = axes[ax_idx]; ax_idx += 1
        sym = results["symmetry"]
        rs  = sym["resolutions"]
        t_no  = sym["times_nosym"]
        t_yes = sym["times_sym"]
        x = np.arange(len(rs))
        w = 0.35
        ax.bar(x - w/2, t_no,  w, label="No symmetry", color="salmon",    alpha=0.85)
        ax.bar(x + w/2, t_yes, w, label="Mirror(X)",   color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}" for r in rs])
        ax.set_xlabel("Resolution (px/μm)")
        ax.set_ylabel("Wall time (s)")
        ax.set_title("Symmetry speedup\n(Mirror(X) vs none)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        # Annotate speedup ratios
        for xi, (tn, ts_) in enumerate(zip(t_no, t_yes)):
            ax.text(xi, max(tn, ts_) * 1.04, f"×{tn/ts_:.1f}",
                    ha="center", fontsize=8, color="navy")

    # ── MPI scaling ────────────────────────────────────────────────────────
    if "mpi" in results:
        ax  = axes[ax_idx]; ax_idx += 1
        mpi = results["mpi"]
        np_ = mpi["nprocs"]
        sp  = mpi["speedups"]
        ef  = mpi["efficiencies"]

        ax.plot(np_, sp, "o-", color="steelblue", lw=2, ms=7, label="Measured speedup")
        ax.plot(np_, np_, "k--", lw=1, alpha=0.5, label="Ideal (linear)")
        ax.set_xlabel("MPI processes")
        ax.set_ylabel("Speedup")
        ax.set_title("MPI strong scaling\n(unit cell, fixed problem size)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(np_, ef, "s--", color="darkorange", lw=1.2, ms=5, alpha=0.7,
                 label="Efficiency (%)")
        ax2.set_ylabel("Parallel efficiency (%)", color="darkorange")
        ax2.set_ylim(0, 110)
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax2.axhline(100, color="darkorange", lw=0.5, ls=":")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
#  Solver comparison overlay (Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

def plot_solver_comparison(
    widths_meep, amplitudes_meep, phases_meep,
    widths_rcwa, amplitudes_rcwa, phases_rcwa,
    period, wavelength,
    filename,
    label_meep="MEEP FDTD",
    label_rcwa="torcwa RCWA",
):
    """
    Two-panel overlay of |T| and ∠T vs pillar width for MEEP and RCWA.

    Parameters
    ----------
    widths_meep / widths_rcwa       : array  pillar widths in μm
    amplitudes_meep / amplitudes_rcwa: array  |T|
    phases_meep / phases_rcwa       : array  ∠T in radians (unwrapped)
    period      : float  unit cell period in μm (for title)
    wavelength  : float  free-space wavelength in μm (for title)
    filename    : str    output PNG path
    """
    _ensure_dir(filename)

    widths_meep_nm = np.asarray(widths_meep) * 1000
    widths_rcwa_nm = np.asarray(widths_rcwa) * 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Solver Comparison  |  λ = {wavelength*1000:.0f} nm"
        f"  |  period = {period*1000:.0f} nm",
        fontsize=13,
    )

    # ── |T| ──────────────────────────────────────────────────────────────────
    ax1.plot(widths_meep_nm, amplitudes_meep,
             "b-o", ms=4, lw=1.5, label=label_meep)
    ax1.plot(widths_rcwa_nm, amplitudes_rcwa,
             "r--s", ms=4, lw=1.5, label=label_rcwa)
    ax1.set_xlabel("Pillar width (nm)")
    ax1.set_ylabel("|T|")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Transmission amplitude")

    # ── ∠T ───────────────────────────────────────────────────────────────────
    ax2.plot(widths_meep_nm, np.degrees(phases_meep),
             "b-o", ms=4, lw=1.5, label=label_meep)
    ax2.plot(widths_rcwa_nm, np.degrees(phases_rcwa),
             "r--s", ms=4, lw=1.5, label=label_rcwa)
    ax2.set_xlabel("Pillar width (nm)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Transmission phase")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
#  Surrogate training diagnostics (Phase 4)
# ══════════════════════════════════════════════════════════════════════════════

def plot_surrogate_training(
    train_losses, val_losses,
    pred_amp, true_amp,
    pred_phase, true_phase,
    filename,
):
    """
    Three-panel diagnostic for surrogate training.

    Panel 1: train/val loss vs epoch (log scale)
    Panel 2: parity plot for |T|  (predicted vs actual)
    Panel 3: parity plot for ∠T  (predicted vs actual, degrees)

    Parameters
    ----------
    train_losses : list   MSE loss per epoch (training set)
    val_losses   : list   MSE loss per epoch (validation set)
    pred_amp     : array  surrogate-predicted |T| on training data
    true_amp     : array  ground-truth |T|
    pred_phase   : array  surrogate-predicted ∠T in radians
    true_phase   : array  ground-truth ∠T in radians
    filename     : str    output PNG path
    """
    _ensure_dir(filename)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Surrogate MLP — Training Diagnostics", fontsize=13)

    # ── Loss curves ────────────────────────────────────────────────────────
    ax = axes[0]
    epochs = np.arange(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, "b-",  lw=1.5, label="Train")
    ax.semilogy(epochs, val_losses,   "r--", lw=1.5, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── |T| parity ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(true_amp, pred_amp, s=10, alpha=0.5, color="steelblue")
    lo, hi = 0.0, 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="ideal")
    mae = float(np.mean(np.abs(pred_amp - true_amp)))
    ax.set_xlabel("True |T|")
    ax.set_ylabel("Predicted |T|")
    ax.set_title(f"|T| parity  (MAE = {mae:.4f})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── ∠T parity ──────────────────────────────────────────────────────────
    ax = axes[2]
    true_deg = np.degrees(true_phase)
    pred_deg = np.degrees(pred_phase)
    ax.scatter(true_deg, pred_deg, s=10, alpha=0.5, color="darkorange")
    lo2 = min(true_deg.min(), pred_deg.min()) - 5
    hi2 = max(true_deg.max(), pred_deg.max()) + 5
    ax.plot([lo2, hi2], [lo2, hi2], "k--", lw=1, alpha=0.6, label="ideal")
    # circular MAE
    mae_ph = float(np.degrees(np.mean(np.abs(
        np.angle(np.exp(1j * np.radians(pred_deg - true_deg)))
    ))))
    ax.set_xlabel("True ∠T (°)")
    ax.set_ylabel("Predicted ∠T (°)")
    ax.set_title(f"∠T parity  (MAE = {mae_ph:.1f}°)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")


def plot_optimised_design(
    x_um, target_phases,
    phases_opt, phases_nn,
    widths_opt, widths_nn,
    amps_opt, amps_nn,
    opt_losses,
    target_desc, filename,
):
    """
    Four-panel summary of the inverse-design optimisation result.

    Panel 1: Phase profile — target, surrogate-optimised, NN baseline
    Panel 2: Pillar widths — surrogate-optimised vs NN baseline
    Panel 3: Amplitude |T| — surrogate-optimised vs NN baseline
    Panel 4: Optimisation loss curve

    Parameters
    ----------
    x_um          : array  pillar x-positions in μm
    target_phases : array  desired phases in radians
    phases_opt    : array  surrogate-predicted phases at optimised widths
    phases_nn     : array  phases from nearest-neighbour library lookup
    widths_opt    : array  optimised pillar widths in μm
    widths_nn     : array  NN-assigned pillar widths in μm
    amps_opt      : array  surrogate-predicted amplitudes at optimised widths
    amps_nn       : array  library amplitudes at NN-assigned widths
    opt_losses    : list   loss per gradient-descent step
    target_desc   : str    description for the figure title
    filename      : str    output PNG path
    """
    _ensure_dir(filename)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Inverse Design — {target_desc}", fontsize=13)
    axes = axes.flatten()

    x_nm = np.asarray(x_um) * 1000

    # ── Phase profile ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x_nm, np.degrees(target_phases), "k-",  lw=2,   label="Target")
    ax.plot(x_nm, np.degrees(phases_opt),    "b-o",  ms=4,  lw=1.5,
            label="Surrogate opt.")
    ax.plot(x_nm, np.degrees(phases_nn),     "r--s", ms=4,  lw=1.5,
            label="NN baseline")
    ax.set_xlabel("x position (nm)")
    ax.set_ylabel("Phase (degrees)")
    ax.set_title("Phase profile")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Pillar widths ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.step(x_nm, widths_opt * 1000, "b-",  lw=1.5, where="mid",
            label="Surrogate opt.")
    ax.step(x_nm, widths_nn  * 1000, "r--", lw=1.5, where="mid",
            label="NN baseline")
    ax.set_xlabel("x position (nm)")
    ax.set_ylabel("Pillar width (nm)")
    ax.set_title("Assigned pillar widths")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Amplitudes ───────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(x_nm, amps_opt, "b-o",  ms=4, lw=1.5, label=f"Surrogate  "
            f"(mean={amps_opt.mean():.3f})")
    ax.plot(x_nm, amps_nn,  "r--s", ms=4, lw=1.5, label=f"NN baseline"
            f"(mean={amps_nn.mean():.3f})")
    ax.set_xlabel("x position (nm)")
    ax.set_ylabel("|T|")
    ax.set_ylim(0, 1.05)
    ax.set_title("Transmission amplitude")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Optimisation loss ────────────────────────────────────────────────────
    ax = axes[3]
    ax.semilogy(np.arange(1, len(opt_losses) + 1), opt_losses,
                "b-", lw=1.5)
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("Loss")
    ax.set_title("Optimisation convergence")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {filename}")
