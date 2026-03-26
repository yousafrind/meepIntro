"""
Multi-wavelength surrogate MLP: maps [w/period, λ] → (|T|, sin∠T, cos∠T).

Extends the Phase 4 single-wavelength surrogate by adding wavelength as a
second input feature.  Training data is collected by running the RCWA sweep
at multiple wavelengths (seconds each) and combining the libraries.

With wavelength as an input the surrogate can:
  1. Interpolate between wavelengths it has not seen during training.
  2. Support gradient-based optimisation that simultaneously satisfies
     phase constraints at multiple wavelengths — enabling achromatic design.

Workflow
--------
  # Step 1: generate per-wavelength RCWA libraries (fast)
  python run.py 07_broadband/multiwl_train.py --generate-data

  # Step 2: train (data generation + training in one call)
  python run.py 07_broadband/multiwl_train.py

  # Step 3: use the model in achromatic_design.py
  python run.py 07_broadband/achromatic_design.py

Usage
-----
    python run.py 07_broadband/multiwl_train.py [options]

    --wavelengths  0.45 0.532 0.633   wavelengths to include (μm)
    --period       0.25               unit cell period (μm)
    --height       0.60               pillar height (μm)
    --n-widths     50                 width samples per wavelength
    --fourier-order 15                RCWA Fourier order
    --hidden   64   hidden layer width
    --layers   3    number of hidden layers
    --epochs   1000 training epochs
    --lr       1e-3 initial learning rate
    --dropout  0.05 dropout probability
    --augment  20   noise copies per data point
    --val-frac 0.2  validation fraction
    --seed     42
    --outdir   results
    --skip-data     skip RCWA data generation (use existing .npz files)

Output
------
    results/multiwl_model.pt         trained model + stats
    results/multiwl_training.png     loss curves + parity plots
    results/wl_<NNN>nm_library.npz   per-wavelength RCWA PhaseLibrary files
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.device    import print_platform_info, get_torch_device
from utils.materials import load_material
from utils.sweep     import PhaseLibrary


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

WAVELENGTHS    = [0.45, 0.532, 0.633]   # μm
PERIOD         = 0.25
HEIGHT         = 0.60
N_GLASS        = 1.5
MATERIAL       = "TiO2"
N_WIDTHS       = 50
FOURIER_ORDER  = 15
GEO_RESOLUTION = 128
HIDDEN         = 64
LAYERS         = 3
EPOCHS         = 1000
LR             = 1e-3
DROPOUT        = 0.05
AUGMENT        = 20
VAL_FRAC       = 0.2
SEED           = 42
OUT_DIR        = "results"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-wavelength surrogate: [w/period, λ] → (|T|, sin∠T, cos∠T).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wavelengths",    nargs="+", type=float,
                   default=WAVELENGTHS,
                   help="Wavelengths to include in training data (μm)")
    p.add_argument("--material",       default=MATERIAL)
    p.add_argument("--period",         type=float, default=PERIOD)
    p.add_argument("--height",         type=float, default=HEIGHT)
    p.add_argument("--n-glass",        type=float, default=N_GLASS, dest="n_glass")
    p.add_argument("--n-widths",       type=int,   default=N_WIDTHS, dest="n_widths")
    p.add_argument("--fourier-order",  type=int,   default=FOURIER_ORDER,
                   dest="fourier_order")
    p.add_argument("--geo-resolution", type=int,   default=GEO_RESOLUTION,
                   dest="geo_resolution")
    p.add_argument("--hidden",         type=int,   default=HIDDEN)
    p.add_argument("--layers",         type=int,   default=LAYERS)
    p.add_argument("--epochs",         type=int,   default=EPOCHS)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--dropout",        type=float, default=DROPOUT)
    p.add_argument("--augment",        type=int,   default=AUGMENT)
    p.add_argument("--val-frac",       type=float, default=VAL_FRAC, dest="val_frac")
    p.add_argument("--seed",           type=int,   default=SEED)
    p.add_argument("--outdir",         default=OUT_DIR)
    p.add_argument("--skip-data",      action="store_true", dest="skip_data",
                   help="Skip RCWA data generation, use existing .npz files")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _lib_path(outdir, wavelength_um):
    nm = int(round(wavelength_um * 1000))
    return os.path.join(outdir, f"wl_{nm:04d}nm_library.npz")


def generate_libraries(args, outdir, device):
    """
    Run RCWA sweep at each wavelength and save PhaseLibrary files.
    Skips wavelengths whose output file already exists.
    """
    # Import here so CPU-only installs without torcwa still parse the module
    from rcwa_sim import rcwa_sweep   # 05_solver_comparison/rcwa_sim.py
    sys.path.insert(0, os.path.join(_ROOT, "05_solver_comparison"))

    # Re-import after path update
    import importlib
    rcwa_mod = importlib.import_module("rcwa_sim")

    for wl in args.wavelengths:
        path = _lib_path(outdir, wl)
        if os.path.exists(path):
            print(f"[data] λ={wl*1000:.0f} nm — existing library found, skipping.")
            continue
        print(f"\n[data] λ={wl*1000:.0f} nm — running RCWA sweep ...")
        lib, elapsed = rcwa_mod.rcwa_sweep(
            material       = args.material,
            wavelength     = wl,
            period         = args.period,
            height         = args.height,
            n_glass        = args.n_glass,
            n_widths       = args.n_widths,
            fourier_order  = args.fourier_order,
            geo_resolution = args.geo_resolution,
            device         = device,
            verbose        = False,
        )
        lib.save(path)
        # Append timing
        data = dict(np.load(path, allow_pickle=True))
        data["sweep_time_s"] = np.float64(elapsed)
        np.savez(path, **data)
        print(f"[data]   done in {elapsed:.1f} s  →  {path}")


def load_multiwl_dataset(args, outdir):
    """
    Load all per-wavelength libraries and stack into a combined dataset.

    Returns
    -------
    X      : float32 array [N, 2]  columns: (w_norm, wl_norm)
    Y      : float32 array [N, 3]  columns: (|T|, sin∠T, cos∠T)
    stats  : dict with normalisation parameters
    raw    : dict with per-wavelength arrays for parity evaluation
    """
    all_X, all_Y = [], []
    raw = {"wavelengths": [], "widths_frac": [], "amplitudes": [], "phases": []}

    wl_all = np.array(args.wavelengths, dtype=float)
    wl_mean = float(wl_all.mean())
    wl_std  = float(wl_all.std()) + 1e-8

    w_frac_all = []

    for wl in args.wavelengths:
        lib = PhaseLibrary.load(_lib_path(outdir, wl))
        w_frac = lib.widths / lib.period

        w_frac_all.append(w_frac)
        raw["wavelengths"].extend([wl] * len(w_frac))
        raw["widths_frac"].extend(w_frac.tolist())
        raw["amplitudes"].extend(lib.amplitudes.tolist())
        raw["phases"].extend(lib.phases.tolist())

    w_frac_concat = np.concatenate(w_frac_all)
    w_mean = float(w_frac_concat.mean())
    w_std  = float(w_frac_concat.std()) + 1e-8

    for wl in args.wavelengths:
        lib = PhaseLibrary.load(_lib_path(outdir, wl))
        w_frac = lib.widths / lib.period
        w_norm = (w_frac - w_mean) / w_std
        wl_norm = (wl - wl_mean) / wl_std

        X = np.column_stack([
            w_norm,
            np.full_like(w_norm, wl_norm),
        ]).astype(np.float32)

        Y = np.column_stack([
            lib.amplitudes,
            np.sin(lib.phases),
            np.cos(lib.phases),
        ]).astype(np.float32)

        all_X.append(X)
        all_Y.append(Y)

    stats = dict(w_mean=w_mean, w_std=w_std,
                 wl_mean=wl_mean, wl_std=wl_std)
    for k in raw:
        raw[k] = np.array(raw[k])

    return np.vstack(all_X), np.vstack(all_Y), stats, raw


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(n_hidden, n_layers, dropout, device):
    """
    MLP: 2 inputs [w_norm, wl_norm] → 3 outputs [|T|, sin∠T, cos∠T].
    Same architecture as Phase 4 surrogate but n_in=2.
    """
    import torch.nn as nn

    layers = []
    in_dim = 2
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, n_hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = n_hidden
    layers.append(nn.Linear(in_dim, 3))

    class MWLNet(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk

        def forward(self, x):
            import torch
            out = self.trunk(x)
            amp = torch.sigmoid(out[:, :1])
            sc  = out[:, 1:]
            return torch.cat([amp, sc], dim=1)

    import torch
    return MWLNet(nn.Sequential(*layers)).to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(args, X_all, Y_all, device):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Augment width inputs with noise (wavelength inputs kept fixed)
    if args.augment > 0:
        noise = rng.normal(0, 0.05, size=(args.augment * len(X_all),)).astype(np.float32)
        X_aug = np.tile(X_all, (args.augment, 1))
        X_aug[:, 0] += noise                        # only w_norm column
        Y_aug = np.tile(Y_all, (args.augment, 1))
        X_all = np.vstack([X_all, X_aug])
        Y_all = np.vstack([Y_all, Y_aug])

    idx   = rng.permutation(len(X_all))
    n_val = max(1, int(len(idx) * args.val_frac))
    idx_val, idx_tr = idx[:n_val], idx[n_val:]

    X_tr  = torch.tensor(X_all[idx_tr],  device=device)
    Y_tr  = torch.tensor(Y_all[idx_tr],  device=device)
    X_val = torch.tensor(X_all[idx_val], device=device)
    Y_val = torch.tensor(Y_all[idx_val], device=device)

    print(f"[train] After augmentation: {len(X_all)} pts  "
          f"(train={len(idx_tr)}, val={len(idx_val)})")

    loader  = DataLoader(TensorDataset(X_tr, Y_tr),
                         batch_size=min(256, len(idx_tr)), shuffle=True)
    net     = build_model(args.hidden, args.layers, args.dropout, device)
    n_params= sum(p.numel() for p in net.parameters())
    print(f"[train] Parameters: {n_params}  "
          f"(hidden={args.hidden}, layers={args.layers})")

    opt   = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val, best_state = float("inf"), None

    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        net.train()
        batch_loss = 0.0
        for xb, yb in loader:
            pred = net(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            batch_loss += loss.item() * len(xb)
        sched.step()
        tl = batch_loss / len(idx_tr)
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(X_val), Y_val).item()
        train_losses.append(tl); val_losses.append(vl)
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"[train]   epoch {epoch:5d}/{args.epochs}  "
                  f"train={tl:.6f}  val={vl:.6f}")

    print(f"[train] Done in {time.perf_counter()-t_start:.1f} s  "
          f"| best val MSE = {best_val:.6f}")
    net.load_state_dict(best_state)
    return net, train_losses, val_losses


# ══════════════════════════════════════════════════════════════════════════════
#  PARITY EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_parity(net, X_raw, Y_raw, device):
    import torch
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(X_raw, device=device)).cpu().numpy()
    pred_amp   = pred[:, 0]
    pred_phase = np.arctan2(pred[:, 1], pred[:, 2])
    true_amp   = Y_raw[:, 0]
    true_phase = np.arctan2(Y_raw[:, 1], Y_raw[:, 2])
    mae_amp    = float(np.mean(np.abs(pred_amp - true_amp)))
    mae_phase  = float(np.degrees(np.mean(np.abs(
        np.angle(np.exp(1j * (pred_phase - true_phase)))
    ))))
    print(f"[train] Parity  MAE |T| = {mae_amp:.4f}   MAE ∠T = {mae_phase:.2f}°")
    return pred_amp, pred_phase, true_amp, true_phase


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print_platform_info()

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required. pip install torch")

    device = get_torch_device() or torch.device("cpu")
    print(f"\n[multiwl] device = {device}")
    print(f"[multiwl] wavelengths: "
          f"{[f'{w*1000:.0f} nm' for w in args.wavelengths]}")
    print(f"[multiwl] period={args.period*1000:.0f} nm  "
          f"height={args.height*1000:.0f} nm  "
          f"material={args.material}")

    # ── Data ─────────────────────────────────────────────────────────────────
    if not args.skip_data:
        sys.path.insert(0, os.path.join(_ROOT, "05_solver_comparison"))
        generate_libraries(args, outdir, device)

    X_all, Y_all, stats, raw = load_multiwl_dataset(args, outdir)
    print(f"[multiwl] Total data points: {len(X_all)}")

    # ── Train ────────────────────────────────────────────────────────────────
    net, train_losses, val_losses = train(args, X_all, Y_all, device)

    # Parity on raw (un-augmented) data
    X_raw = X_all[:len(raw["wavelengths"])]  # first N rows are original
    Y_raw = Y_all[:len(raw["wavelengths"])]
    pred_amp, pred_phase, true_amp, true_phase = \
        evaluate_parity(net, X_raw, Y_raw, device)

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = os.path.join(outdir, "multiwl_model.pt")
    torch.save({
        "state_dict"  : net.state_dict(),
        "stats"       : stats,
        "hidden"      : args.hidden,
        "layers"      : args.layers,
        "dropout"     : args.dropout,
        "wavelengths" : args.wavelengths,
        "period"      : args.period,
        "height"      : args.height,
        "material"    : args.material,
    }, model_path)
    print(f"[multiwl] Model saved → {model_path}")

    # ── Training plot ─────────────────────────────────────────────────────────
    from utils.viz import plot_surrogate_training
    plot_surrogate_training(
        train_losses = train_losses,
        val_losses   = val_losses,
        pred_amp     = pred_amp,
        true_amp     = true_amp,
        pred_phase   = pred_phase,
        true_phase   = true_phase,
        filename     = os.path.join(outdir, "multiwl_training.png"),
    )

    print(f"\n[multiwl] Done.  Outputs in: {outdir}")


if __name__ == "__main__":
    main()
