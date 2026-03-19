"""
MPI-aware launcher for all meepIntro simulations.

Auto-detects available CPU cores and launches the target script under
mpirun if available, falling back to single-process Python otherwise.
Works identically on any machine (laptop, Ryzen desktop, WSL).

Usage
-----
    python run.py <script> [script args...]

Examples
--------
    python run.py 01_beam_steering/unit_cell_sweep.py
    python run.py 01_beam_steering/unit_cell_sweep.py --n-widths 20 --resolution 32
    python run.py 01_beam_steering/full_array_sim.py --angle 45 --resolution 32
    python run.py 01_beam_steering/full_array_sim.py --help

Environment overrides
---------------------
    MEEP_NPROCS=4 python run.py ...   # force a specific core count
    MEEP_NPROCS=1 python run.py ...   # force single-process (no mpirun)
"""

import multiprocessing
import os
import shutil
import subprocess
import sys


def get_nprocs():
    """
    Determine number of MPI processes to use.

    Priority:
    1. MEEP_NPROCS environment variable (explicit override)
    2. All logical CPUs reported by the OS
    """
    env_val = os.environ.get("MEEP_NPROCS")
    if env_val is not None:
        try:
            n = int(env_val)
            if n < 1:
                raise ValueError
            return n
        except ValueError:
            print(f"[run.py] WARNING: invalid MEEP_NPROCS='{env_val}', using all cores.")

    return multiprocessing.cpu_count()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    script      = sys.argv[1]
    script_args = sys.argv[2:]

    if not os.path.isfile(script):
        sys.exit(f"[run.py] Script not found: {script}")

    nprocs = get_nprocs()
    python  = sys.executable          # same interpreter that launched run.py

    has_mpirun = shutil.which("mpirun") is not None

    if has_mpirun and nprocs > 1:
        cmd = ["mpirun", "-np", str(nprocs), python, script] + script_args
        print(f"[run.py] Launching with MPI: {nprocs} processes")
    else:
        cmd = [python, script] + script_args
        reason = "mpirun not found" if not has_mpirun else "MEEP_NPROCS=1"
        print(f"[run.py] Single-process mode ({reason})")

    print(f"[run.py] Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"\n[run.py] Simulation failed with exit code {e.returncode}.")
    except KeyboardInterrupt:
        sys.exit("\n[run.py] Interrupted.")


if __name__ == "__main__":
    main()
