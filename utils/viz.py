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
