"""
Material loader for MEEP simulations.

Reads n,k vs wavelength from plain-text files (space/tab separated,
# comment lines allowed). Interpolates to the simulation wavelength
and returns a MEEP Medium object.

Bundled materials live in  <repo_root>/materials/
Compatible with files downloaded from refractiveindex.info
(export as "space-separated" or paste the data columns directly).

File format (two or three columns):
    # comment lines are ignored
    wavelength_um   n   [k]
    0.400           2.78  0.001
    0.500           2.58  0.0
    ...

Usage:
    from utils.materials import load_material
    mat = load_material("TiO2")          # bundled default
    mat = load_material("path/to/my.txt")  # custom file
    # Returns a callable: mat(wavelength_um) -> mp.Medium
"""

import os
import numpy as np

# Resolve repo root relative to this file
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAT_DIR   = os.path.join(_REPO_ROOT, "materials")

# Map short names → bundled filenames
BUNDLED = {
    "TiO2" : "TiO2_rutile_Siefke2016.txt",
    "SiO2" : "SiO2_Malitson1965.txt",
    "glass": "SiO2_Malitson1965.txt",
}


def _parse_nk_file(filepath):
    """
    Parse a plain-text n,k file.

    Returns
    -------
    wl  : np.ndarray  wavelengths in μm, sorted ascending
    n   : np.ndarray  real part of refractive index
    k   : np.ndarray  imaginary part (0 if not present in file)
    """
    wl_list, n_list, k_list = [], [], []

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                wl_list.append(float(parts[0]))
                n_list.append(float(parts[1]))
                k_list.append(float(parts[2]) if len(parts) >= 3 else 0.0)
            except ValueError:
                continue  # skip header rows that weren't caught by #

    if not wl_list:
        raise ValueError(f"No numeric data found in {filepath}")

    wl = np.array(wl_list)
    n  = np.array(n_list)
    k  = np.array(k_list)

    # Sort by wavelength
    idx = np.argsort(wl)
    return wl[idx], n[idx], k[idx]


class Material:
    """
    Interpolating material with n,k data.

    Call  mat(wavelength_um)  to get a mp.Medium at that wavelength.
    Access  mat.n(wl), mat.k(wl)  for raw interpolated values.
    """

    def __init__(self, filepath, name=None):
        self.filepath = filepath
        self.name     = name or os.path.basename(filepath)
        self._wl, self._n_data, self._k_data = _parse_nk_file(filepath)

        wl_min = self._wl[0]
        wl_max = self._wl[-1]
        print(f"[material] Loaded '{self.name}'  "
              f"({len(self._wl)} points, "
              f"{wl_min*1000:.0f}–{wl_max*1000:.0f} nm)")

    def n(self, wavelength_um):
        """Interpolated real refractive index at wavelength_um."""
        self._check_range(wavelength_um)
        return float(np.interp(wavelength_um, self._wl, self._n_data))

    def k(self, wavelength_um):
        """Interpolated extinction coefficient at wavelength_um."""
        self._check_range(wavelength_um)
        return float(np.interp(wavelength_um, self._wl, self._k_data))

    def _check_range(self, wl):
        if wl < self._wl[0] or wl > self._wl[-1]:
            print(f"[material] WARNING: {wl:.4f} μm is outside "
                  f"tabulated range [{self._wl[0]:.4f}, {self._wl[-1]:.4f}] μm "
                  f"for '{self.name}' — extrapolating.")

    def meep_medium(self, wavelength_um):
        """
        Return a MEEP Medium at the given wavelength.

        For lossless materials (k=0): Medium(epsilon=n²).
        For lossy materials: adds D_conductivity to represent Im(ε).

        Note: MEEP is a time-domain solver. This gives the correct
        permittivity at a single frequency. For broadband simulations
        a Lorentz-Drude fit is needed (Phase 3 work).
        """
        import meep as mp

        n_val = self.n(wavelength_um)
        k_val = self.k(wavelength_um)
        freq  = 1.0 / wavelength_um   # MEEP frequency units (c=1, length in μm)

        eps_r = n_val**2 - k_val**2   # Re(ε)
        eps_i = 2.0 * n_val * k_val   # Im(ε)

        if k_val < 1e-6:
            return mp.Medium(epsilon=eps_r)

        # MEEP: D_conductivity σ such that ε_eff = ε_r + i·σ/ω
        # → σ = eps_i · ω = eps_i · 2π·freq
        D_cond = eps_i * 2.0 * np.pi * freq
        return mp.Medium(epsilon=eps_r, D_conductivity=D_cond)

    def __call__(self, wavelength_um):
        """Shorthand: mat(wavelength_um) → mp.Medium"""
        return self.meep_medium(wavelength_um)

    def __repr__(self):
        return f"Material('{self.name}')"


def load_material(name_or_path):
    """
    Load a material by short name (bundled) or file path.

    Parameters
    ----------
    name_or_path : str
        Short name from BUNDLED dict (e.g. "TiO2", "SiO2")
        or an absolute/relative path to an n,k text file.

    Returns
    -------
    Material instance (callable: mat(wavelength_um) → mp.Medium)
    """
    # Check if it's a known short name
    if name_or_path in BUNDLED:
        filepath = os.path.join(_MAT_DIR, BUNDLED[name_or_path])
        return Material(filepath, name=name_or_path)

    # Otherwise treat as a file path
    if not os.path.isfile(name_or_path):
        raise FileNotFoundError(
            f"Material '{name_or_path}' not found as a bundled name or file.\n"
            f"Bundled options: {list(BUNDLED.keys())}"
        )
    return Material(name_or_path)
