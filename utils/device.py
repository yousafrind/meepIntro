"""
Auto-detect available compute resources.

MEEP parallelism is always MPI (launched by run.py).
This module handles PyTorch backend detection for Phase 3 (RCWA).
"""

import multiprocessing
import os


def get_cpu_count():
    """Return number of logical CPUs."""
    return multiprocessing.cpu_count()


def get_torch_device():
    """
    Return best available PyTorch device.
    ROCm builds expose themselves as 'cuda' — same code works for both.
    Returns None if PyTorch is not installed.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None


def print_platform_info():
    """Print available compute resources to stdout."""
    ncpu = get_cpu_count()
    # Under MPI, meep.am_master() guards output; we replicate that here
    # with a simple rank check so only rank 0 prints.
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if rank != 0:
            return
        mpi_info = f"{size} MPI processes"
    except ImportError:
        mpi_info = "1 process (mpi4py not found)"

    print(f"[platform] CPUs      : {ncpu}")
    print(f"[platform] MPI       : {mpi_info}")

    dev = get_torch_device()
    if dev is None:
        print("[platform] PyTorch   : not installed (needed for Phase 3 RCWA)")
    elif dev.type == "cuda":
        try:
            import torch
            print(f"[platform] GPU       : {torch.cuda.get_device_name(0)} (ROCm/CUDA)")
        except Exception:
            print("[platform] GPU       : available")
    else:
        print("[platform] GPU       : not available — PyTorch will use CPU")
