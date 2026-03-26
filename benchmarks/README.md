# Phase 2 — Performance Benchmarks

Systematic timing of MEEP simulations across three axes: **resolution**,
**symmetry**, and **MPI scaling**.  Run these to calibrate your workflow
before committing to a long production sweep.

---

## Why benchmark?

| Parameter | Effect on runtime | Effect on accuracy |
|-----------|------------------|--------------------|
| Resolution | O(r³) — doubling r → 8× slower | Higher r → sharper field gradients resolved |
| Symmetry   | Mirror(X) → ~2× faster (halves x grid) | Identical result for valid symmetric geometries |
| MPI cores  | ~linear up to communication limit | Identical result (domain decomposition) |

Understanding the trade-offs lets you choose the fastest resolution that
still gives converged results.

---

## Usage

```bash
# Full benchmark suite (resolution + symmetry + MPI)
python run.py benchmarks/benchmark.py

# Individual benchmarks
python run.py benchmarks/benchmark.py --mode resolution
python run.py benchmarks/benchmark.py --mode symmetry
python run.py benchmarks/benchmark.py --mode mpi

# Custom resolution list
python run.py benchmarks/benchmark.py --mode resolution \
    --resolutions 16 32 64 128

# Quick smoke-test (completes in < 1 min)
python run.py benchmarks/benchmark.py --quick

python run.py benchmarks/benchmark.py --help
```

---

## Output

| File | Description |
|------|-------------|
| `results/benchmark.png` | Multi-panel summary plot |
| `results/benchmark_report.txt` | Human-readable timing tables |
| `results/benchmark.json` | Machine-readable raw data |

---

## Benchmark 1 — Resolution scaling

Runs a single unit-cell MEEP sim (one pillar, 50% fill) at resolutions
`[16, 32, 64, 128]` px/μm and records wall time.

**Theory:** Runtime scales as O(r³):
- r² grid points in 2D (x and y)
- r time steps (Courant condition: Δt ∝ 1/r)

| Resolution | Time (typical) | Notes |
|-----------|----------------|-------|
| 16 px/μm  | ~2–5 s         | Too coarse for phase accuracy |
| 32 px/μm  | ~15–50 s       | Good for design exploration |
| 64 px/μm  | ~120–400 s     | Production accuracy |
| 128 px/μm | ~1000–3200 s   | High accuracy / convergence test |

**Convergence test:** Run the same simulation at 32 and 64.  If phases agree
within ~5°, resolution 32 is sufficient.  If they differ, use 64.

---

## Benchmark 2 — Symmetry (Mirror(X))

A centred TiO2 pillar illuminated at normal incidence is symmetric about
x = 0.  Adding `mp.Mirror(mp.X)` to MEEP halves the x grid:

```
Without symmetry:  simulate full cell  [−period/2, +period/2]
With Mirror(X):    simulate half cell  [0, +period/2]  → ~2× faster
```

The result (transmission amplitude and phase) is identical.

**When is Mirror(X) valid?**
- Pillar centred at x = 0  ✓
- Normal incidence (k_x = 0)  ✓
- Plane-wave source (uniform in x)  ✓

**When is it NOT valid?**
- Off-centre pillars (full array sim)  ✗
- Oblique incidence (k_x ≠ 0)  ✗

Mirror(X) is now the default for `utils/sweep.py` via `--symmetry`:
```bash
python run.py utils/sweep.py --symmetry --outdir 01_beam_steering/results
```

---

## Benchmark 3 — MPI strong scaling

Launches the same single-cell simulation via subprocess with increasing
`MEEP_NPROCS`.  MEEP decomposes the domain along y (slab decomposition):

```
nprocs = 1  → one rank owns full domain
nprocs = 2  → each rank owns half the y-domain
nprocs = N  → each rank owns sy/N
```

**Expected behaviour:**
- Speedup ≈ N for small N (communication overhead ≪ compute)
- Speedup saturates when domain per rank is too small to amortise MPI overhead
- For the unit cell (sx × sy ≈ 0.25 × ~2.5 μm) this threshold is typically
  around 4–8 cores at resolution 32

**Parallel efficiency** = speedup / N × 100 %.
Efficiency > 70% is considered good for MEEP.

**Running the MPI benchmark requires `mpirun` on PATH:**
```bash
# Install OpenMPI via conda
conda install -c conda-forge openmpi
```

---

## Applying the results to production runs

After benchmarking:

1. **Choose resolution**: pick the lowest r where phase values converge
   (compare r and 2r, accept when Δφ < 5°).

2. **Enable symmetry** for unit-cell sweeps:
   ```bash
   python run.py utils/sweep.py --symmetry --resolution 64 \
       --outdir 01_beam_steering/results
   ```

3. **Set MPI cores** to the "knee" of the efficiency curve:
   ```bash
   MEEP_NPROCS=4 python run.py 01_beam_steering/unit_cell_sweep.py \
       --resolution 64
   ```

4. **Full-array sims** (01_beam_steering, 02_metalens, 03_holography) benefit
   from MPI but not from Mirror(X) (the phase gradient breaks x-symmetry).
   Tune cores based on the full-array cell size (sx >> period).
