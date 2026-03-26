"""
Train a surrogate MLP that maps pillar geometry → transmission amplitude/phase.

The surrogate replaces expensive FDTD/RCWA calls during optimisation.
Training data comes from one or more PhaseLibrary .npz files (MEEP or RCWA).

Network
-------
  Input  (3): [w/period,  sin(∠T_target),  cos(∠T_target)]
  → actually the forward model is:
  Input  (1 or 3 depending on --multi-wavelength): normalised pillar width
  Output (3): [|T|,  sin(∠T),  cos(∠T)]

  sin/cos encoding avoids the phase-wrapping discontinuity at ±π.
  |T| ∈ [0,1] is bounded by a sigmoid output activation.

  Architecture: Linear(n_in) → [Linear(H) → ReLU → Dropout] × n_layers → Linear(3)

Training
--------
  - Adam optimiser, cosine annealing LR schedule
  - MSE loss on all three outputs
  - 80/20 train/val split (augmented with small Gaussian noise on widths)
  - Early stopping on validation loss

Usage
-----
    # Train on the default MEEP library
    python run.py 06_surrogate/train.py

    # Train on RCWA library (faster to generate)
    python run.py 06_surrogate/train.py \\
        --libs 05_solver_comparison/results/rcwa_phase_library.npz

    # Train on multiple libraries (MEEP + RCWA combined)
    python run.py 06_surrogate/train.py \\
        --libs 01_beam_steering/results/phase_library.npz \\
               05_solver_comparison/results/rcwa_phase_library.npz

    # Custom architecture
    python run.py 06_surrogate/train.py --hidden 128 --layers 4 --epochs 2000

    Options
    -------
    --libs          one or more paths to PhaseLibrary .npz files
    --hidden   64   hidden layer width
    --layers   3    number of hidden layers
    --epochs   1000 training epochs
    --lr       1e-3 initial learning rate
    --dropout  0.05 dropout probability
    --augment  20   noise augmentation copies per data point (σ = 0.5% of period)
    --val-frac 0.2  fraction of data held out for validation
    --seed     42   random seed
    --outdir   results output directory

Output
------
    results/surrogate_model.pt       saved model + normalisation stats
    results/training_curves.png      train/val loss vs epoch
    results/parity_plot.png          predicted vs actual |T| and ∠T
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.device import print_platform_info, get_torch_device
from utils.sweep  import PhaseLibrary
from utils.viz    import plot_surrogate_training


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_LIB = os.path.join(
    _ROOT, "01_beam_steering", "results", "phase_library.npz"
)
HIDDEN    = 64
LAYERS    = 3
EPOCHS    = 1000
LR        = 1e-3
DROPOUT   = 0.05
AUGMENT   = 20
VAL_FRAC  = 0.2
SEED      = 42
OUT_DIR   = "results"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train surrogate MLP on PhaseLibrary data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--libs",     nargs="+", default=[_DEFAULT_LIB],
                   help="Path(s) to PhaseLibrary .npz files")
    p.add_argument("--hidden",   type=int,   default=HIDDEN)
    p.add_argument("--layers",   type=int,   default=LAYERS)
    p.add_argument("--epochs",   type=int,   default=EPOCHS)
    p.add_argument("--lr",       type=float, default=LR)
    p.add_argument("--dropout",  type=float, default=DROPOUT)
    p.add_argument("--augment",  type=int,   default=AUGMENT,
                   help="Noise augmentation copies per data point")
    p.add_argument("--val-frac", type=float, default=VAL_FRAC, dest="val_frac")
    p.add_argument("--seed",     type=int,   default=SEED)
    p.add_argument("--outdir",   default=OUT_DIR)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(n_hidden, n_layers, dropout, device):
    """
    MLP surrogate: 1 input (normalised width) → 3 outputs (|T|, sin∠T, cos∠T).

    Input:  w_norm = (w - w_mean) / w_std   (scalar, shape [B, 1])
    Output: [|T|, sin(∠T), cos(∠T)]         (shape [B, 3])
      |T| is bounded to [0,1] by sigmoid.
      sin/cos outputs are unbounded (MSE loss keeps them near unit circle).
    """
    import torch
    import torch.nn as nn

    layers = []
    in_dim = 1
    for i in range(n_layers):
        layers.append(nn.Linear(in_dim, n_hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = n_hidden

    # Output head: raw linear outputs, then custom activation applied outside
    layers.append(nn.Linear(in_dim, 3))

    class SurrogateNet(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk

        def forward(self, x):
            out = self.trunk(x)
            amp = torch.sigmoid(out[:, :1])          # |T| ∈ (0, 1)
            sc  = out[:, 1:]                          # sin, cos (free)
            return torch.cat([amp, sc], dim=1)

    net = SurrogateNet(nn.Sequential(*layers)).to(device)
    return net


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(lib_paths):
    """
    Load one or more PhaseLibrary files, return (widths_norm, targets, stats).

    Targets: numpy array [N, 3] = [|T|, sin(∠T), cos(∠T)]
    stats: dict with w_mean, w_std for denormalisation.
    """
    all_w, all_amp, all_phase = [], [], []

    for path in lib_paths:
        lib = PhaseLibrary.load(path)
        all_w.append(lib.widths / lib.period)     # normalise by period → [0,1]
        all_amp.append(lib.amplitudes)
        all_phase.append(lib.phases)

    widths_frac = np.concatenate(all_w)
    amplitudes  = np.concatenate(all_amp)
    phases      = np.concatenate(all_phase)

    # Normalise width fraction to zero mean / unit std
    w_mean = float(widths_frac.mean())
    w_std  = float(widths_frac.std()) + 1e-8

    w_norm = (widths_frac - w_mean) / w_std

    targets = np.stack([
        amplitudes,
        np.sin(phases),
        np.cos(phases),
    ], axis=1).astype(np.float32)

    stats = dict(w_mean=w_mean, w_std=w_std)
    return w_norm.astype(np.float32), targets, stats, widths_frac


def augment_data(w_norm, targets, n_copies, sigma_frac, rng):
    """
    Add Gaussian noise to w_norm inputs, keeping targets fixed.
    sigma_frac: noise std as fraction of w_norm std (already=1 after normalisation).
    """
    if n_copies <= 0:
        return w_norm, targets
    noise = rng.normal(0, sigma_frac, size=(n_copies * len(w_norm),)).astype(np.float32)
    w_aug = np.tile(w_norm, n_copies) + noise
    t_aug = np.tile(targets, (n_copies, 1))
    w_all = np.concatenate([w_norm, w_aug])
    t_all = np.concatenate([targets, t_aug])
    return w_all, t_all


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(args, device):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ── Load & prepare data ──────────────────────────────────────────────────
    print(f"\n[train] Loading {len(args.libs)} library file(s) ...")
    w_norm, targets, stats, widths_frac = load_dataset(args.libs)
    print(f"[train] Raw data points : {len(w_norm)}")

    w_aug, t_aug = augment_data(w_norm, targets,
                                n_copies=args.augment,
                                sigma_frac=0.05,
                                rng=rng)

    # Shuffle then split
    idx = rng.permutation(len(w_aug))
    n_val = max(1, int(len(idx) * args.val_frac))
    idx_val, idx_tr = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(w_aug[idx_tr, None], device=device)
    Y_tr = torch.tensor(t_aug[idx_tr],       device=device)
    X_val= torch.tensor(w_aug[idx_val, None],device=device)
    Y_val= torch.tensor(t_aug[idx_val],      device=device)

    print(f"[train] After augmentation: {len(w_aug)} points  "
          f"(train={len(idx_tr)}, val={len(idx_val)})")

    loader = DataLoader(TensorDataset(X_tr, Y_tr),
                        batch_size=min(256, len(idx_tr)),
                        shuffle=True)

    # ── Model, optimiser, scheduler ─────────────────────────────────────────
    net = build_model(args.hidden, args.layers, args.dropout, device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[train] Model parameters: {n_params}")

    opt   = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val  = float("inf")
    best_state= None

    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        net.train()
        batch_loss = 0.0
        for xb, yb in loader:
            pred = net(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_loss += loss.item() * len(xb)
        sched.step()

        train_loss = batch_loss / len(idx_tr)
        train_losses.append(train_loss)

        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(X_val), Y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"[train]   epoch {epoch:5d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    elapsed = time.perf_counter() - t_start
    print(f"\n[train] Training done in {elapsed:.1f} s  "
          f"| best val MSE = {best_val:.6f}")

    # Restore best weights
    net.load_state_dict(best_state)
    return net, stats, train_losses, val_losses, widths_frac, targets[:len(w_norm)]


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def save_model(net, stats, outdir, args):
    """Save model state dict + normalisation stats + hyperparams to .pt file."""
    import torch
    path = os.path.join(outdir, "surrogate_model.pt")
    torch.save({
        "state_dict" : net.state_dict(),
        "stats"      : stats,
        "hidden"     : args.hidden,
        "layers"     : args.layers,
        "dropout"    : args.dropout,
        "libs"       : args.libs,
    }, path)
    print(f"[train] Model saved → {path}")
    return path


def evaluate_parity(net, w_norm_raw, targets_raw, stats, device):
    """
    Run the trained model on the original (un-augmented) data points.
    Returns predicted amplitudes and phases for parity plot.
    """
    import torch
    net.eval()
    X = torch.tensor(w_norm_raw[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = net(X).cpu().numpy()

    pred_amp   = pred[:, 0]
    pred_phase = np.arctan2(pred[:, 1], pred[:, 2])  # recover ∠T from sin/cos

    true_amp   = targets_raw[:, 0]
    true_phase = np.arctan2(targets_raw[:, 1], targets_raw[:, 2])

    # MAE summary
    mae_amp   = float(np.mean(np.abs(pred_amp   - true_amp)))
    mae_phase = float(np.mean(np.abs(
        np.angle(np.exp(1j * (pred_phase - true_phase)))  # circular diff
    )))
    print(f"[train] Parity  MAE |T| = {mae_amp:.4f}   "
          f"MAE ∠T = {np.degrees(mae_phase):.2f}°")

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
        raise ImportError(
            "PyTorch is required for Phase 4.  "
            "Install with: pip install torch  "
            "or (ROCm): pip install torch --index-url "
            "https://download.pytorch.org/whl/rocm6.0"
        )

    device = get_torch_device() or torch.device("cpu")
    print(f"\n[train] device = {device}")
    print(f"[train] hidden={args.hidden}  layers={args.layers}  "
          f"epochs={args.epochs}  lr={args.lr}")

    net, stats, train_losses, val_losses, widths_frac, targets_raw = \
        train(args, device)

    # Normalise raw widths the same way as training data for parity eval
    w_frac_norm = ((widths_frac / widths_frac.max()
                    - stats["w_mean"]) / stats["w_std"]).astype(np.float32)
    pred_amp, pred_phase, true_amp, true_phase = \
        evaluate_parity(net, w_frac_norm, targets_raw, stats, device)

    save_model(net, stats, outdir, args)

    # Plots
    plot_surrogate_training(
        train_losses = train_losses,
        val_losses   = val_losses,
        pred_amp     = pred_amp,
        true_amp     = true_amp,
        pred_phase   = pred_phase,
        true_phase   = true_phase,
        filename     = os.path.join(outdir, "training_curves.png"),
    )

    print(f"\n[train] Done.  Outputs in: {outdir}")


if __name__ == "__main__":
    main()
