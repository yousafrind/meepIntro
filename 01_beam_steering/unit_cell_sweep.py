"""
Unit cell phase library sweep — beam steering metasurface.

Thin wrapper around utils.sweep.sweep() with beam-steering defaults.
The actual MEEP simulation lives in utils/sweep.py so any sim can reuse it.

Run this ONCE to generate the phase library before running full_array_sim.py.

Usage
-----
    python run.py 01_beam_steering/unit_cell_sweep.py [options]

    --material   TiO2          bundled name or path to n,k file
    --wavelength 0.532         free-space wavelength in μm
    --period     0.25          unit cell period in μm
    --height     0.60          pillar height in μm
    --n-glass    1.5           substrate refractive index
    --resolution 64            MEEP pixels per μm
    --n-widths   50            number of width samples
    --phase-step None          target phase step in degrees (overrides --n-widths)
    --outdir     results       where to save library + plots

Output
------
    results/phase_library.npz  data (loaded by full_array_sim, metalens, hologram)
    results/phase_library.png  4-panel diagnostic plot

Alternatively, run utils/sweep.py directly (same engine, more options):
    python run.py utils/sweep.py --outdir 01_beam_steering/results
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.device import print_platform_info
from utils.sweep  import sweep
from utils.viz    import plot_phase_library


def parse_args():
    p = argparse.ArgumentParser(
        description="Unit cell sweep: build phase library for beam steering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--material",   default="TiO2")
    p.add_argument("--wavelength", type=float, default=0.532)
    p.add_argument("--period",     type=float, default=0.25)
    p.add_argument("--height",     type=float, default=0.60)
    p.add_argument("--n-glass",    type=float, default=1.5, dest="n_glass")
    p.add_argument("--resolution", type=int,   default=64)
    p.add_argument("--n-widths",   type=int,   default=50,  dest="n_widths")
    p.add_argument("--phase-step", type=float, default=None, dest="phase_step",
                   help="Target phase step in degrees; overrides --n-widths")
    p.add_argument("--outdir",     default="results")
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print_platform_info()

    lib = sweep(
        material   = args.material,
        wavelength = args.wavelength,
        period     = args.period,
        height     = args.height,
        n_glass    = args.n_glass,
        resolution = args.resolution,
        n_widths   = args.n_widths,
        phase_step = args.phase_step,
    )

    try:
        import meep as mp
        if not mp.am_master():
            return
    except ImportError:
        pass

    lib_path = lib.save(os.path.join(outdir, "phase_library.npz"))

    plot_phase_library(
        lib.widths, lib.phases_unwrapped, lib.amplitudes,
        period    = lib.period,
        wavelength= lib.wavelength,
        filename  = os.path.join(outdir, "phase_library.png"),
    )

    print(f"\n[sweep] Done.  Library → {lib_path}")


if __name__ == "__main__":
    main()
