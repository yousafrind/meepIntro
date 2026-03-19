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
