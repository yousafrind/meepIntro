"""
Phase 2 — Performance benchmarks: resolution, symmetry, MPI scaling.

Systematically measures MEEP run time across three axes and produces
tables + plots to guide parameter choices for production runs.

Benchmarks
----------

1. Resolution scaling (single unit cell, no MPI)
   Runs run_unit_cell() at resolution = 16, 32, 64, 128 px/μm.
   Measures wall time and compares to the theoretical O(r³) scaling
   (doubling resolution → 8× more work: 4× grid points, 2× time steps).

2. Symmetry speedup (single unit cell, no MPI)
   Runs at resolution = 32 and 64 with and without Mirror(X) symmetry.
   Expected: ~2× speedup from halving the x grid.

3. MPI strong scaling (single unit cell, fixed resolution)
   Repeats a single unit-cell sim via subprocess with nprocs = 1, 2, 4 …
   Plots speedup and parallel efficiency (speedup / nprocs).
   Requires mpirun on PATH; skipped with a warning if absent.

Usage
-----
    # All three benchmarks (default)
    python run.py benchmarks/benchmark.py

    # Individual benchmarks
    python run.py benchmarks/benchmark.py --mode resolution
    python run.py benchmarks/benchmark.py --mode symmetry
    python run.py benchmarks/benchmark.py --mode mpi

    # Customise resolution list
    python run.py benchmarks/benchmark.py --mode resolution \\
        --resolutions 16 32 64

    # MPI: test up to 8 cores
    python run.py benchmarks/benchmark.py --mode mpi --max-procs 8

    # Quick smoke-test (tiny resolution, 1 width point, no MPI)
    python run.py benchmarks/benchmark.py --quick

    Options:
    --mode           all | resolution | symmetry | mpi  [default: all]
    --resolutions    space-separated list of px/μm values
    --sym-resolutions  resolutions to use for symmetry benchmark
    --max-procs      maximum MPI process count
    --material       TiO2  (or any bundled / file)
    --wavelength     0.532 μm
    --period         0.25 μm
    --height         0.60 μm
    --n-glass        1.5
    --quick          use minimal settings for a fast smoke-test
    --outdir         benchmarks/results

Output
------
    results/benchmark.png        multi-panel summary plot
    results/benchmark.json       raw timing data (machine-readable)
    results/benchmark_report.txt human-readable table
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.device    import print_platform_info
from utils.materials import load_material
from utils.sweep     import run_unit_cell
from utils.viz       import plot_benchmark_results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2 performance benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",            default="all",
                   choices=["all", "resolution", "symmetry", "mpi"],
                   help="Which benchmark(s) to run")
    p.add_argument("--resolutions",     type=int, nargs="+",
                   default=[16, 32, 64, 128],
                   help="Resolution values for the resolution benchmark")
    p.add_argument("--sym-resolutions", type=int, nargs="+",
                   default=[32, 64], dest="sym_resolutions",
                   help="Resolutions used for the symmetry benchmark")
    p.add_argument("--max-procs",       type=int, default=None,
                   dest="max_procs",
                   help="Max MPI process count (default: all logical cores)")
    p.add_argument("--material",        default="TiO2")
    p.add_argument("--wavelength",      type=float, default=0.532)
    p.add_argument("--period",          type=float, default=0.25)
    p.add_argument("--height",          type=float, default=0.60)
    p.add_argument("--n-glass",         type=float, default=1.5, dest="n_glass")
    p.add_argument("--quick",           action="store_true",
                   help="Smoke-test: smallest resolutions, skip MPI")
    p.add_argument("--outdir",          default="results")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cell_pixels(resolution, period, height, wavelength):
    """Total MEEP grid cells for the unit cell at this resolution."""
    dpml       = wavelength
    subs_thick = 1.5 * wavelength
    air_thick  = 1.5 * wavelength
    sy = dpml + subs_thick + height + air_thick + dpml
    sx = period
    return int(resolution * sx) * int(resolution * sy)


def _time_unit_cell(pillar_width, period, height, wavelength,
                    pillar_medium, n_glass, resolution, use_symmetry=False):
    """Run one unit-cell sim and return wall-clock seconds."""
    t0 = time.perf_counter()
    run_unit_cell(
        pillar_width  = pillar_width,
        period        = period,
        pillar_height = height,
        wavelength    = wavelength,
        pillar_medium = pillar_medium,
        n_glass       = n_glass,
        resolution    = resolution,
        use_symmetry  = use_symmetry,
    )
    return time.perf_counter() - t0


def _table_row(label, value_fmt, *values):
    row = f"  {label:<30}"
    for v in values:
        row += f"  {value_fmt.format(v):>12}"
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark 1 — Resolution scaling
# ══════════════════════════════════════════════════════════════════════════════

def bench_resolution(args, pillar_medium, outdir):
    """
    Time a single unit-cell sim at each resolution.
    Uses pillar_width = 0.5 * period (mid-range, representative).
    """
    print("\n" + "═" * 60)
    print("  Benchmark 1: Resolution scaling")
    print("═" * 60)
    print(f"  Resolutions : {args.resolutions}")
    print(f"  Pillar width: {0.5*args.period*1000:.0f} nm (50% fill)")

    resolutions = args.resolutions
    times       = []
    n_cells     = []

    for res in resolutions:
        nc = _cell_pixels(res, args.period, args.height, args.wavelength)
        print(f"\n  [res={res:3d}]  grid cells ≈ {nc:,}  … ", end="", flush=True)
        t = _time_unit_cell(
            pillar_width  = 0.5 * args.period,
            period        = args.period,
            height        = args.height,
            wavelength    = args.wavelength,
            pillar_medium = pillar_medium,
            n_glass       = args.n_glass,
            resolution    = res,
        )
        times.append(t)
        n_cells.append(nc)
        print(f"{t:.1f} s")

    # Print table
    print(f"\n  {'Resolution':>12}  {'Grid cells':>12}  {'Time (s)':>10}  "
          f"{'Speedup':>9}  {'vs O(r³)':>9}")
    print("  " + "-" * 60)
    for i, (res, t, nc) in enumerate(zip(resolutions, times, n_cells)):
        sp   = times[0] / t
        # Expected speedup from O(r³): (res[0]/res[i])^3
        exp  = (resolutions[0] / res) ** 3
        print(f"  {res:>12}  {nc:>12,}  {t:>10.1f}  "
              f"{sp:>9.2f}×  {exp:>9.2f}×")

    return dict(resolutions=resolutions, times=times, n_cells=n_cells)


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark 2 — Symmetry speedup
# ══════════════════════════════════════════════════════════════════════════════

def bench_symmetry(args, pillar_medium, outdir):
    """
    Compare wall time with and without Mirror(X) symmetry at each resolution.
    """
    print("\n" + "═" * 60)
    print("  Benchmark 2: Symmetry — Mirror(X) vs no symmetry")
    print("═" * 60)
    print(f"  Resolutions : {args.sym_resolutions}")

    resolutions = args.sym_resolutions
    times_nosym = []
    times_sym   = []

    for res in resolutions:
        for use_sym, label, store in [
            (False, "no-sym", times_nosym),
            (True,  "sym   ", times_sym),
        ]:
            print(f"\n  [res={res:3d}, {label}]  … ", end="", flush=True)
            t = _time_unit_cell(
                pillar_width  = 0.5 * args.period,
                period        = args.period,
                height        = args.height,
                wavelength    = args.wavelength,
                pillar_medium = pillar_medium,
                n_glass       = args.n_glass,
                resolution    = res,
                use_symmetry  = use_sym,
            )
            store.append(t)
            print(f"{t:.1f} s")

    print(f"\n  {'Resolution':>12}  {'No-sym (s)':>12}  "
          f"{'Mirror(X) (s)':>14}  {'Speedup':>9}")
    print("  " + "-" * 54)
    for res, tn, ts in zip(resolutions, times_nosym, times_sym):
        sp = tn / ts
        print(f"  {res:>12}  {tn:>12.1f}  {ts:>14.1f}  {sp:>9.2f}×")

    return dict(resolutions=resolutions,
                times_nosym=times_nosym,
                times_sym=times_sym)


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark 3 — MPI strong scaling
# ══════════════════════════════════════════════════════════════════════════════

def _available_procs(max_procs):
    """Return sorted list of proc counts to test: [1, 2, 4, …, max_procs]."""
    import multiprocessing
    n_cpu  = multiprocessing.cpu_count()
    cap    = min(max_procs, n_cpu) if max_procs else n_cpu
    counts = []
    p = 1
    while p <= cap:
        counts.append(p)
        p *= 2
    if counts[-1] < cap:
        counts.append(cap)
    return sorted(set(counts))


def _mpi_time_subprocess(nprocs, args):
    """
    Launch   MEEP_NPROCS=nprocs python run.py utils/sweep.py --n-widths 1 ...
    and return wall-clock seconds.  Returns None on failure.
    """
    env = os.environ.copy()
    env["MEEP_NPROCS"] = str(nprocs)

    cmd = [
        sys.executable,
        os.path.join(_ROOT, "run.py"),
        os.path.join(_ROOT, "utils", "sweep.py"),
        "--n-widths",   "1",
        "--material",   args.material,
        "--wavelength", str(args.wavelength),
        "--period",     str(args.period),
        "--height",     str(args.height),
        "--n-glass",    str(args.n_glass),
        "--resolution", str(max(args.sym_resolutions)),  # use highest sym res
        "--outdir",     "/tmp/_meep_bench_discard",
        "--no-plot",
    ]

    t0  = time.perf_counter()
    ret = subprocess.run(cmd, env=env,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - t0

    if ret.returncode != 0:
        return None
    return elapsed


def bench_mpi(args, outdir):
    """
    Strong scaling: same single-cell problem, increasing nprocs.
    Uses subprocess so the nprocs change takes effect via run.py / mpirun.
    """
    print("\n" + "═" * 60)
    print("  Benchmark 3: MPI strong scaling")
    print("═" * 60)

    if not shutil.which("mpirun"):
        print("  [skip] mpirun not found on PATH — skipping MPI benchmark.")
        print("         Install OpenMPI or MPICH and re-run to enable.")
        return None

    nproc_list = _available_procs(args.max_procs)
    print(f"  Testing nprocs : {nproc_list}")
    print(f"  Resolution     : {max(args.sym_resolutions)} px/μm")
    print(f"  (using subprocess → run.py → mpirun)")

    times      = []
    speedups   = []
    efficiencies = []
    successful = []

    t_serial = None
    for np_ in nproc_list:
        print(f"\n  [nprocs={np_:2d}]  … ", end="", flush=True)
        t = _mpi_time_subprocess(np_, args)
        if t is None:
            print("FAILED — skipping")
            continue
        if t_serial is None:
            t_serial = t
        sp = t_serial / t
        ef = sp / np_ * 100
        times.append(t)
        speedups.append(sp)
        efficiencies.append(ef)
        successful.append(np_)
        print(f"{t:.1f} s  speedup={sp:.2f}×  efficiency={ef:.0f}%")

    if not successful:
        print("  [mpi] No successful runs.")
        return None

    print(f"\n  {'nprocs':>8}  {'Time (s)':>10}  {'Speedup':>9}  "
          f"{'Efficiency':>12}")
    print("  " + "-" * 44)
    for np_, t, sp, ef in zip(successful, times, speedups, efficiencies):
        print(f"  {np_:>8}  {t:>10.1f}  {sp:>9.2f}×  {ef:>11.0f}%")

    return dict(nprocs=successful, times=times,
                speedups=speedups, efficiencies=efficiencies)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if args.quick:
        args.resolutions     = [16, 32]
        args.sym_resolutions = [16, 32]
        args.mode            = "resolution"   # skip MPI in quick mode
        print("[benchmark] Quick mode: reduced resolutions, no MPI.")

    print_platform_info()

    mat    = load_material(args.material)
    medium = mat(args.wavelength)

    print(f"\n[benchmark] Material   : {args.material}  n={mat.n(args.wavelength):.3f}")
    print(f"[benchmark] Wavelength : {args.wavelength*1000:.0f} nm")
    print(f"[benchmark] Period     : {args.period*1000:.0f} nm  "
          f"Height: {args.height*1000:.0f} nm")

    run_res = run_sym = run_mpi = None
    results = {}
    mode    = args.mode

    if mode in ("all", "resolution"):
        run_res = bench_resolution(args, medium, outdir)
        if run_res:
            results["resolution"] = run_res

    if mode in ("all", "symmetry"):
        run_sym = bench_symmetry(args, medium, outdir)
        if run_sym:
            results["symmetry"] = run_sym

    if mode in ("all", "mpi") and not args.quick:
        run_mpi = bench_mpi(args, outdir)
        if run_mpi:
            results["mpi"] = run_mpi

    # ── Save results ──────────────────────────────────────────────────────────
    if results:
        plot_benchmark_results(
            results,
            filename=os.path.join(outdir, "benchmark.png"),
        )

        report_path = os.path.join(outdir, "benchmark_report.txt")
        with open(report_path, "w") as fh:
            fh.write("Phase 2 Benchmark Report\n")
            fh.write("=" * 60 + "\n\n")
            fh.write(f"Material  : {args.material}\n")
            fh.write(f"Wavelength: {args.wavelength*1000:.0f} nm\n")
            fh.write(f"Period    : {args.period*1000:.0f} nm  "
                     f"Height: {args.height*1000:.0f} nm\n\n")

            if "resolution" in results:
                r = results["resolution"]
                fh.write("Resolution Scaling\n" + "-" * 40 + "\n")
                fh.write(f"{'res':>6}  {'cells':>10}  {'time(s)':>10}  "
                         f"{'speedup':>9}\n")
                for res, nc, t in zip(r["resolutions"], r["n_cells"], r["times"]):
                    sp = r["times"][0] / t
                    fh.write(f"{res:>6}  {nc:>10,}  {t:>10.1f}  {sp:>9.2f}x\n")
                fh.write("\n")

            if "symmetry" in results:
                s = results["symmetry"]
                fh.write("Symmetry Speedup\n" + "-" * 40 + "\n")
                fh.write(f"{'res':>6}  {'no-sym(s)':>10}  "
                         f"{'Mirror(X)(s)':>13}  {'speedup':>9}\n")
                for res, tn, ts in zip(s["resolutions"],
                                       s["times_nosym"], s["times_sym"]):
                    fh.write(f"{res:>6}  {tn:>10.1f}  {ts:>13.1f}  "
                             f"{tn/ts:>9.2f}x\n")
                fh.write("\n")

            if "mpi" in results:
                m = results["mpi"]
                fh.write("MPI Strong Scaling\n" + "-" * 40 + "\n")
                fh.write(f"{'nprocs':>8}  {'time(s)':>10}  "
                         f"{'speedup':>9}  {'efficiency':>12}\n")
                for np_, t, sp, ef in zip(m["nprocs"], m["times"],
                                          m["speedups"], m["efficiencies"]):
                    fh.write(f"{np_:>8}  {t:>10.1f}  "
                             f"{sp:>9.2f}x  {ef:>11.0f}%\n")

        print(f"\n[benchmark] Report  → {report_path}")

        json_path = os.path.join(outdir, "benchmark.json")
        with open(json_path, "w") as fh:
            json.dump(
                {k: {kk: list(vv) if hasattr(vv, "__iter__") else vv
                     for kk, vv in v.items()}
                 for k, v in results.items()},
                fh, indent=2,
            )
        print(f"[benchmark] JSON    → {json_path}")

    print("\n[benchmark] Done.")


if __name__ == "__main__":
    main()
