"""
utils/sweep.py — Universal unit-cell phase library sweep.

Runs a 2D FDTD (MEEP) parameter sweep over TiO2/SiO2/custom pillar widths,
extracts complex transmission coefficients, and builds a reusable phase library.

Public API
----------
    from utils.sweep import sweep, PhaseLibrary, run_unit_cell

    # Build a library
    lib = sweep(
        material    = "TiO2",
        wavelength  = 0.532,
        period      = 0.25,
        height      = 0.60,
        n_glass     = 1.5,
        resolution  = 64,
        n_widths    = 50,          # OR use phase_step instead of n_widths
        phase_step  = None,        # target phase spacing in degrees
    )
    lib.save("results/phase_library.npz")

    # Load a saved library
    lib = PhaseLibrary.load("results/phase_library.npz")

    # Nearest-neighbour phase → width lookup
    widths, errors = lib.assign_widths(target_phases_rad)

    # Metadata
    print(lib.phase_coverage())   # degrees
    print(lib.period, lib.wavelength, lib.material)

    # Dict-style access (backward compatible with raw np.load NpzFile)
    lib["widths"], lib["phases"], lib["period"], lib["material"]

CLI (run as a standalone script via run.py)
-------------------------------------------
    python run.py utils/sweep.py [options]

    --material    TiO2           bundled name or path to n,k file
    --wavelength  0.532          free-space wavelength in μm
    --period      0.25           unit cell period in μm
    --height      0.60           pillar height in μm
    --n-glass     1.5            substrate refractive index
    --resolution  64             MEEP pixels per μm
    --n-widths    50             number of width samples  (use OR --phase-step)
    --phase-step  None           target phase step in degrees; overrides --n-widths
    --width-min   0.05           min pillar width as fraction of period
    --width-max   0.90           max pillar width as fraction of period
    --outdir      .              output directory (phase_library.npz saved here)
    --no-plot                    skip the diagnostic plot

Phase-step mode
---------------
When --phase-step DEG is specified a brief coarse sweep (15 points) is run
first to estimate the actual phase coverage of the geometry.  The required
number of width samples to achieve ≤ DEG phase spacing is then computed
automatically before running the full sweep.  This guarantees adequate phase
sampling without manual tuning of --n-widths.

Physics notes
-------------
* 2D FDTD, TM polarisation (Ez plane wave source).
* Bloch periodic BC in x (k_point = 0 → normal incidence).
* PML at top and bottom (y direction).
* Complex transmission T = E_pillar / E_ref (spatial average of DFT Ez).
* Phases are stored wrapped (−π to π) AND unwrapped for plotting.
* The library is polarisation-specific (TE → use mp.Hz source; left for
  future extension).
"""

import argparse
import os
import sys

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = _HERE if os.path.basename(_HERE) != "utils" else os.path.dirname(_HERE)

# Ensure repo root is on the path when this module is imported from elsewhere
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.materials import load_material   # noqa: E402  (after path fixup)


# ══════════════════════════════════════════════════════════════════════════════
#  PhaseLibrary  — data container + lookup
# ══════════════════════════════════════════════════════════════════════════════

_SAVE_KEYS = (
    "widths", "phases", "phases_unwrapped", "amplitudes",
    "wavelength", "period", "pillar_height", "n_glass", "material",
)


class PhaseLibrary:
    """
    Container for a unit-cell phase library.

    Attributes
    ----------
    widths           : ndarray  pillar widths in μm
    phases           : ndarray  transmission phase, wrapped to (−π, π]
    phases_unwrapped : ndarray  unwrapped phase (for plotting / coverage)
    amplitudes       : ndarray  transmission amplitude |T|
    wavelength       : float    free-space wavelength in μm
    period           : float    unit cell period in μm
    pillar_height    : float    pillar height in μm
    n_glass          : float    substrate refractive index
    material         : str      material name / file path
    """

    def __init__(self, *, widths, phases, amplitudes,
                 wavelength, period, pillar_height, n_glass, material):
        self.widths           = np.asarray(widths,      dtype=float)
        self.phases           = np.asarray(phases,      dtype=float)
        self.phases_unwrapped = np.unwrap(self.phases)
        self.amplitudes       = np.asarray(amplitudes,  dtype=float)
        self.wavelength       = float(wavelength)
        self.period           = float(period)
        self.pillar_height    = float(pillar_height)
        self.n_glass          = float(n_glass)
        self.material         = str(material)

    # ── dict-style access (backward compat with raw NpzFile usage) ────────────

    _ALIASES = {"pillar_height": "pillar_height"}  # reserved

    def __getitem__(self, key):
        """Support lib['widths'], lib['phases'], lib['period'] etc."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    # ── phase coverage ────────────────────────────────────────────────────────

    def phase_coverage(self):
        """Total phase range in degrees (unwrapped)."""
        return abs(float(np.degrees(
            self.phases_unwrapped[-1] - self.phases_unwrapped[0]
        )))

    # ── nearest-neighbour width assignment ────────────────────────────────────

    def assign_widths(self, target_phases):
        """
        For each target phase (radians), find the nearest library entry.

        Parameters
        ----------
        target_phases : array-like  desired phases in radians

        Returns
        -------
        widths : ndarray  assigned pillar widths in μm
        errors : ndarray  |target − assigned| phase error in degrees
        """
        target_phases  = np.asarray(target_phases, dtype=float)
        ph_lib_norm    = self.phases % (2 * np.pi)

        assigned = np.empty(len(target_phases))
        errors   = np.empty(len(target_phases))

        for i, phi in enumerate(target_phases):
            phi_norm = phi % (2 * np.pi)
            diff     = np.abs(ph_lib_norm - phi_norm)
            diff     = np.minimum(diff, 2 * np.pi - diff)   # circular distance
            idx      = int(np.argmin(diff))
            assigned[i] = self.widths[idx]
            errors[i]   = np.degrees(diff[idx])

        return assigned, errors

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path):
        """Save library to .npz (adds .npz extension if missing)."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            widths           = self.widths,
            phases           = self.phases,
            phases_unwrapped = self.phases_unwrapped,
            amplitudes       = self.amplitudes,
            wavelength       = self.wavelength,
            period           = self.period,
            pillar_height    = self.pillar_height,
            n_glass          = self.n_glass,
            material         = self.material,
        )
        print(f"[sweep] Library saved → {path}")
        return path

    @classmethod
    def load(cls, path, strict=True):
        """
        Load a PhaseLibrary from a .npz file.

        Works with files saved by this module and by the old
        01_beam_steering/unit_cell_sweep.py (same key names).

        Parameters
        ----------
        path   : str   path to the .npz file
        strict : bool  if True, raise FileNotFoundError if path missing;
                       if False, return None instead

        Returns
        -------
        PhaseLibrary instance, or None if strict=False and file missing
        """
        if not os.path.exists(path):
            if not strict:
                return None
            raise FileNotFoundError(
                f"[sweep] Phase library not found: {path}\n"
                "        Generate it first:\n"
                "          python run.py utils/sweep.py --outdir <dir>"
            )
        data = np.load(path, allow_pickle=True)
        print(f"[sweep] Loaded library: {path}")
        return cls(
            widths       = data["widths"],
            phases       = data["phases"],
            amplitudes   = data["amplitudes"],
            wavelength   = float(data["wavelength"]),
            period       = float(data["period"]),
            pillar_height= float(data["pillar_height"]),
            n_glass      = float(data["n_glass"]),
            material     = str(data["material"]),
        )

    def __repr__(self):
        return (
            f"PhaseLibrary(material='{self.material}', "
            f"λ={self.wavelength*1000:.0f} nm, "
            f"period={self.period*1000:.0f} nm, "
            f"h={self.pillar_height*1000:.0f} nm, "
            f"N={len(self.widths)}, "
            f"coverage={self.phase_coverage():.0f}°)"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MEEP unit-cell simulation
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_cell(pillar_width, period, pillar_height,
                  wavelength, pillar_medium, n_glass, resolution):
    """
    2D FDTD simulation of one unit cell at normal incidence (TM, k=0).

    Cell layout (y = propagation direction, upward):

        ┌──────────────┐  ← sy/2
        │   PML (top)  │
        ├──────────────┤  ← y_mon  (DFT monitor, in air above pillar)
        │   air        │
        ├──╔════╗──────┤  ← pillar base = y_subs_top
        │  ║pill║      │
        ├──╚════╝──────┤
        │  glass subs  │
        ├──────────────┤  ← y_src  (plane-wave source)
        │   PML (bot)  │
        └──────────────┘  ← −sy/2

    Parameters
    ----------
    pillar_width  : float   pillar width in μm (0 = reference run, no pillar)
    period        : float   unit cell period in μm
    pillar_height : float   pillar height in μm
    wavelength    : float   free-space wavelength in μm
    pillar_medium : mp.Medium
    n_glass       : float   substrate refractive index
    resolution    : int     MEEP pixels per μm

    Returns
    -------
    complex : spatially-averaged DFT Ez at the monitor (0th Fourier order)
    """
    import meep as mp

    freq   = 1.0 / wavelength
    fwidth = 0.15 * freq

    dpml       = wavelength
    subs_thick = 1.5 * wavelength
    air_thick  = 1.5 * wavelength

    sy = dpml + subs_thick + pillar_height + air_thick + dpml
    sx = period

    y_bot      = -sy / 2
    y_subs_top = y_bot + dpml + subs_thick
    y_src      = y_bot + dpml + 0.4 * wavelength
    y_mon      = y_subs_top + pillar_height + 0.4 * wavelength

    glass    = mp.Medium(index=n_glass)
    geometry = [
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
        k_point=mp.Vector3(),
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

    return complex(np.mean(sim.get_dft_array(dft, mp.Ez, 0)))


# ══════════════════════════════════════════════════════════════════════════════
#  Sweep driver
# ══════════════════════════════════════════════════════════════════════════════

def _run_raw_sweep(widths, period, pillar_height,
                   wavelength, pillar_medium, n_glass, resolution):
    """
    Inner sweep loop.  Returns (phases, amplitudes) for the given widths.
    Also runs a no-pillar reference first.
    """
    import meep as mp

    n = len(widths)
    phases     = np.zeros(n)
    amplitudes = np.zeros(n)

    if mp.am_master():
        print("\n[sweep] Reference simulation (empty cell) ...")
    ez_ref = run_unit_cell(
        pillar_width=0,
        period=period, pillar_height=pillar_height,
        wavelength=wavelength, pillar_medium=pillar_medium,
        n_glass=n_glass, resolution=resolution,
    )
    if mp.am_master():
        print(f"[sweep] |E_ref| = {abs(ez_ref):.4f}")

    for i, w in enumerate(widths):
        if mp.am_master():
            print(f"\n[sweep] [{i+1:02d}/{n}] width = {w*1000:.1f} nm")
        ez = run_unit_cell(
            pillar_width=w,
            period=period, pillar_height=pillar_height,
            wavelength=wavelength, pillar_medium=pillar_medium,
            n_glass=n_glass, resolution=resolution,
        )
        T = ez / ez_ref
        phases[i]     = np.angle(T)
        amplitudes[i] = abs(T)
        if mp.am_master():
            print(f"[sweep]   |T| = {amplitudes[i]:.4f}   "
                  f"φ = {np.degrees(phases[i]):+.1f}°")

    return phases, amplitudes


def _estimate_n_widths_for_phase_step(phase_step_deg,
                                      coarse_phases, safety_factor=1.25):
    """
    From a coarse sweep, estimate how many width samples give ≤ phase_step
    phase spacing across the full coverage.
    """
    coverage_deg = abs(np.degrees(np.unwrap(coarse_phases)[-1]
                                  - np.unwrap(coarse_phases)[0]))
    n_needed = int(np.ceil(coverage_deg / phase_step_deg * safety_factor)) + 2
    return max(10, n_needed)


def sweep(material, wavelength=0.532, period=0.25, height=0.60,
          n_glass=1.5, resolution=64, n_widths=50, phase_step=None,
          width_min_frac=0.05, width_max_frac=0.90, verbose=True):
    """
    Run a unit-cell phase library sweep.

    Parameters
    ----------
    material       : str or Material  bundled name ("TiO2"), file path, or
                                      already-loaded Material object
    wavelength     : float   free-space wavelength in μm
    period         : float   unit cell period in μm
    height         : float   pillar height in μm
    n_glass        : float   substrate refractive index
    resolution     : int     MEEP pixels per μm
    n_widths       : int     number of width samples (ignored if phase_step set)
    phase_step     : float   target phase step in degrees; triggers a coarse
                             pre-sweep to estimate the required n_widths
    width_min_frac : float   min width / period  (default 0.05)
    width_max_frac : float   max width / period  (default 0.90)
    verbose        : bool    print progress

    Returns
    -------
    PhaseLibrary
    """
    try:
        import meep as mp
        master = mp.am_master()
    except ImportError:
        master = True

    if isinstance(material, str):
        mat = load_material(material)
        mat_name = material
    else:
        mat = material
        mat_name = getattr(mat, "name", str(mat))

    pillar_medium = mat(wavelength)

    w_min = width_min_frac * period
    w_max = width_max_frac * period

    if master and verbose:
        print(f"\n[sweep] ── Unit-cell sweep ─────────────────────────────")
        print(f"[sweep]   material   = {mat_name}")
        print(f"[sweep]   wavelength = {wavelength*1000:.0f} nm")
        print(f"[sweep]   period     = {period*1000:.0f} nm")
        print(f"[sweep]   height     = {height*1000:.0f} nm")
        print(f"[sweep]   n_glass    = {n_glass}")
        print(f"[sweep]   resolution = {resolution} px/μm")

    # ── phase-step mode: coarse pre-sweep → estimate n_widths ─────────────────
    if phase_step is not None:
        n_coarse = 15
        if master and verbose:
            print(f"\n[sweep] Phase-step mode ({phase_step}°): "
                  f"running {n_coarse}-point coarse sweep ...")
        w_coarse       = np.linspace(w_min, w_max, n_coarse)
        ph_coarse, _   = _run_raw_sweep(
            w_coarse, period, height, wavelength,
            pillar_medium, n_glass, resolution,
        )
        n_widths = _estimate_n_widths_for_phase_step(phase_step, ph_coarse)
        if master and verbose:
            cov = abs(np.degrees(np.unwrap(ph_coarse)[-1]
                                 - np.unwrap(ph_coarse)[0]))
            print(f"[sweep] Coarse coverage = {cov:.0f}°  "
                  f"→ using n_widths = {n_widths} for {phase_step}° step")

    # ── full sweep ─────────────────────────────────────────────────────────────
    widths = np.linspace(w_min, w_max, n_widths)
    if master and verbose:
        print(f"[sweep]   n_widths   = {n_widths}  "
              f"(width {w_min*1000:.1f}–{w_max*1000:.1f} nm)")

    phases, amplitudes = _run_raw_sweep(
        widths, period, height, wavelength,
        pillar_medium, n_glass, resolution,
    )

    lib = PhaseLibrary(
        widths       = widths,
        phases       = phases,
        amplitudes   = amplitudes,
        wavelength   = wavelength,
        period       = period,
        pillar_height= height,
        n_glass      = n_glass,
        material     = mat_name,
    )

    if master and verbose:
        cov = lib.phase_coverage()
        flag = "GOOD (≥ 300°)" if cov >= 300 else "LOW — consider increasing height"
        print(f"\n[sweep] Phase coverage = {cov:.0f}°  ({flag})")
        print(f"[sweep] {lib}")

    return lib


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build a unit-cell phase library (MEEP FDTD sweep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--material",   default="TiO2",
                   help="Pillar material: bundled name (TiO2, SiO2) or n,k file path")
    p.add_argument("--wavelength", type=float, default=0.532,
                   help="Free-space wavelength in μm")
    p.add_argument("--period",     type=float, default=0.25,
                   help="Unit cell period in μm")
    p.add_argument("--height",     type=float, default=0.60,
                   help="Pillar height in μm")
    p.add_argument("--n-glass",    type=float, default=1.5,
                   dest="n_glass", help="Substrate refractive index")
    p.add_argument("--resolution", type=int,   default=64,
                   help="MEEP pixels per μm")
    p.add_argument("--n-widths",   type=int,   default=50,
                   dest="n_widths",
                   help="Number of pillar width samples (ignored if --phase-step set)")
    p.add_argument("--phase-step", type=float, default=None,
                   dest="phase_step",
                   help="Target phase step in degrees; auto-sets --n-widths")
    p.add_argument("--width-min",  type=float, default=0.05,
                   dest="width_min_frac",
                   help="Min pillar width as fraction of period")
    p.add_argument("--width-max",  type=float, default=0.90,
                   dest="width_max_frac",
                   help="Max pillar width as fraction of period")
    p.add_argument("--outdir",     default=".",
                   help="Output directory for phase_library.npz and plot")
    p.add_argument("--no-plot",    action="store_true", dest="no_plot",
                   help="Skip the diagnostic plot")
    return p.parse_args()


def main():
    args = _parse_args()

    # Import here so the module is importable without meep installed
    try:
        import meep as mp
        master = mp.am_master()
    except ImportError:
        master = True

    from utils.device import print_platform_info   # noqa: F401
    if master:
        print_platform_info()

    lib = sweep(
        material      = args.material,
        wavelength    = args.wavelength,
        period        = args.period,
        height        = args.height,
        n_glass       = args.n_glass,
        resolution    = args.resolution,
        n_widths      = args.n_widths,
        phase_step    = args.phase_step,
        width_min_frac= args.width_min_frac,
        width_max_frac= args.width_max_frac,
    )

    if master:
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        lib_path = lib.save(os.path.join(outdir, "phase_library.npz"))

        if not args.no_plot:
            from utils.viz import plot_phase_library
            plot_phase_library(
                lib.widths, lib.phases_unwrapped, lib.amplitudes,
                period    = lib.period,
                wavelength= lib.wavelength,
                filename  = os.path.join(outdir, "phase_library.png"),
            )

        print("\n[sweep] Done.")


if __name__ == "__main__":
    main()
