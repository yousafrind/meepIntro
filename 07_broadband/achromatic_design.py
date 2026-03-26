"""
Achromatic metasurface inverse design using the multi-wavelength surrogate.

Jointly optimises pillar widths to satisfy a metalens phase profile
simultaneously at multiple wavelengths.  The loss is the sum of circular phase
MSE across all wavelengths — achievable only through gradient descent, not
nearest-neighbour lookup.

Compares three designs:
  1. Surrogate-optimised (achromatic) — jointly optimised across all λ
  2. Single-λ NN baseline            — nearest-neighbour from the central λ library
  3. Per-λ NN baselines              — best possible NN result per wavelength

Usage
-----
    # Default: metalens at 450 / 532 / 633 nm, focal length 10 μm
    python run.py 07_broadband/achromatic_design.py

    # Custom wavelengths and focal length
    python run.py 07_broadband/achromatic_design.py \\
        --wavelengths 0.48 0.55 0.65 --focal-len 15

    # Beam-steering target
    python run.py 07_broadband/achromatic_design.py \\
        --target beam --angle 30

    Options
    -------
    --model        07_broadband/results/multiwl_model.pt
    --lib          01_beam_steering/results/phase_library.npz  (NN baseline)
    --target       metalens | beam
    --wavelengths  0.45 0.532 0.633   wavelengths to optimise over (μm)
    --focal-len    10     focal length in μm (metalens)
    --angle        30     beam steering angle in degrees (beam)
    --n-pillars    40     number of pillars
    --period       0.25   unit cell period in μm
    --steps        800    gradient descent steps
    --lr           0.05   Adam learning rate
    --outdir       results

Output
------
    results/achromatic_design.npz   optimised widths + per-λ phase profiles
    results/achromatic_design.png   phase profile grid + convergence
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.device import print_platform_info, get_torch_device
from utils.sweep  import PhaseLibrary
from utils.viz    import plot_achromatic_design


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_MODEL = os.path.join(_HERE, "results", "multiwl_model.pt")
_DEFAULT_LIB   = os.path.join(
    _ROOT, "01_beam_steering", "results", "phase_library.npz"
)

WAVELENGTHS = [0.45, 0.532, 0.633]
FOCAL_LEN   = 10.0
ANGLE       = 30.0
N_PILLARS   = 40
PERIOD      = 0.25
OPT_STEPS   = 800
OPT_LR      = 0.05
OUT_DIR     = "results"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Achromatic metasurface design via multi-wavelength surrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",       default=_DEFAULT_MODEL)
    p.add_argument("--lib",         default=_DEFAULT_LIB,
                   help="PhaseLibrary for single-λ NN baseline")
    p.add_argument("--target",      default="metalens",
                   choices=["metalens", "beam"])
    p.add_argument("--wavelengths", nargs="+", type=float,
                   default=WAVELENGTHS)
    p.add_argument("--focal-len",   type=float, default=FOCAL_LEN,
                   dest="focal_len")
    p.add_argument("--angle",       type=float, default=ANGLE)
    p.add_argument("--n-pillars",   type=int,   default=N_PILLARS,
                   dest="n_pillars")
    p.add_argument("--period",      type=float, default=PERIOD)
    p.add_argument("--steps",       type=int,   default=OPT_STEPS)
    p.add_argument("--lr",          type=float, default=OPT_LR)
    p.add_argument("--outdir",      default=OUT_DIR)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET PHASE PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def _target_phases(target, n_pillars, period, wavelength, focal_len, angle_deg):
    x = (np.arange(n_pillars) - (n_pillars - 1) / 2) * period
    k0 = 2 * np.pi / wavelength
    if target == "metalens":
        phi = k0 * (focal_len - np.sqrt(x**2 + focal_len**2))
    else:
        phi = k0 * np.sin(np.radians(angle_deg)) * x
    return phi, x


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD SURROGATE
# ══════════════════════════════════════════════════════════════════════════════

def load_surrogate(model_path, device):
    import torch
    import torch.nn as nn

    ckpt = torch.load(model_path, map_location=device)
    stats = ckpt["stats"]

    # Rebuild model (same architecture as multiwl_train.py)
    hidden  = ckpt["hidden"]
    n_layers= ckpt["layers"]
    dropout = ckpt["dropout"]

    layers = []
    in_dim = 2
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 3))

    class MWLNet(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk

        def forward(self, x):
            out = self.trunk(x)
            amp = torch.sigmoid(out[:, :1])
            return torch.cat([amp, out[:, 1:]], dim=1)

    net = MWLNet(nn.Sequential(*layers)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, stats, ckpt


# ══════════════════════════════════════════════════════════════════════════════
#  ACHROMATIC OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def optimise_achromatic(net, stats, wavelengths, target_phases_all,
                        period, n_steps, lr, device,
                        width_min_frac=0.05, width_max_frac=0.90):
    """
    Find one set of pillar widths that simultaneously minimises phase error
    across all wavelengths.

    Parameters
    ----------
    net               : multi-wavelength SurrogateNet
    stats             : normalisation dict (w_mean, w_std, wl_mean, wl_std)
    wavelengths       : list of floats  (μm)
    target_phases_all : list of ndarray, one per wavelength  [N_pillars]
    period            : float  unit cell period in μm
    n_steps, lr       : optimiser settings
    device            : torch.device

    Returns
    -------
    widths_opt     : ndarray [N]
    phases_per_wl  : list of ndarray  predicted phases at each wavelength
    amps_per_wl    : list of ndarray  predicted amplitudes at each wavelength
    losses         : list of floats   total loss per step
    """
    import torch
    import torch.nn as nn

    N = len(target_phases_all[0])
    w_mean  = stats["w_mean"];   w_std  = stats["w_std"]
    wl_mean = stats["wl_mean"];  wl_std = stats["wl_std"]

    w_min_norm = (width_min_frac - w_mean) / w_std
    w_max_norm = (width_max_frac - w_mean) / w_std

    def to_norm(logit_w):
        return w_min_norm + (w_max_norm - w_min_norm) * torch.sigmoid(logit_w)

    logit_w = nn.Parameter(torch.zeros(N, 1, device=device))
    opt     = torch.optim.Adam([logit_w], lr=lr)

    # Pre-compute wavelength tensors (constant across steps)
    wl_tensors = []
    phi_targets = []
    for wl, phi in zip(wavelengths, target_phases_all):
        wl_norm = (wl - wl_mean) / wl_std
        wl_tensors.append(
            torch.full((N, 1), wl_norm, dtype=torch.float32, device=device)
        )
        phi_targets.append(
            torch.tensor(phi, dtype=torch.float32, device=device)
        )

    losses = []
    for step in range(n_steps):
        w_norm = to_norm(logit_w)              # [N, 1]
        total_loss = torch.tensor(0.0, device=device)

        for wl_t, phi_t in zip(wl_tensors, phi_targets):
            X    = torch.cat([w_norm, wl_t], dim=1)  # [N, 2]
            pred = net(X)                              # [N, 3]
            sin_pred = pred[:, 1]; cos_pred = pred[:, 2]
            sin_t    = torch.sin(phi_t)
            cos_t    = torch.cos(phi_t)
            phase_loss = ((sin_pred - sin_t)**2 + (cos_pred - cos_t)**2).mean()
            amp_reg    = (1.0 - pred[:, 0]).pow(2).mean()
            total_loss = total_loss + phase_loss + 0.1 * amp_reg

        opt.zero_grad(); total_loss.backward(); opt.step()
        losses.append(total_loss.item())

    # Extract final results per wavelength
    net.eval()
    phases_per_wl = []
    amps_per_wl   = []
    with torch.no_grad():
        w_norm_final = to_norm(logit_w)
        for wl_t in wl_tensors:
            X    = torch.cat([w_norm_final, wl_t], dim=1)
            pred = net(X).cpu().numpy()
            amps_per_wl.append(pred[:, 0])
            phases_per_wl.append(np.arctan2(pred[:, 1], pred[:, 2]))

    w_frac_final = w_norm_final.detach().cpu().numpy().squeeze() * w_std + w_mean
    widths_opt   = np.clip(w_frac_final, width_min_frac, width_max_frac) * period
    return widths_opt, phases_per_wl, amps_per_wl, losses


# ══════════════════════════════════════════════════════════════════════════════
#  NN BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def nn_baselines(lib_path, target_phases_all, wavelengths):
    """
    For each wavelength, compute nearest-neighbour assignment from the
    single-λ library (lib_path is the library whose wavelength matches best).
    Returns list of (widths, phases, amps, errors) per wavelength.
    """
    lib     = PhaseLibrary.load(lib_path)
    results = []
    for phi_t in target_phases_all:
        widths_nn, errors = lib.assign_widths(phi_t)
        idx = np.array([
            int(np.argmin(np.abs(lib.widths - w))) for w in widths_nn
        ])
        results.append((widths_nn, lib.phases[idx], lib.amplitudes[idx], errors))
    return results


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
    print(f"\n[achromatic] device = {device}")

    # Load model
    print(f"[achromatic] Loading model: {args.model}")
    net, stats, ckpt = load_surrogate(args.model, device)
    print(f"[achromatic] Trained on λ = "
          f"{[f'{w*1000:.0f} nm' for w in ckpt['wavelengths']]}")

    # Build target phase profiles for each wavelength
    target_phases_all = []
    for wl in args.wavelengths:
        phi, x_um = _target_phases(
            args.target, args.n_pillars, args.period, wl,
            args.focal_len, args.angle
        )
        target_phases_all.append(phi)

    target_desc = (
        f"metalens  f={args.focal_len:.0f} μm" if args.target == "metalens"
        else f"beam steering  θ={args.angle:.0f}°"
    )
    wl_str = " / ".join(f"{w*1000:.0f}" for w in args.wavelengths)
    print(f"[achromatic] Target  : {target_desc}")
    print(f"[achromatic] λ values: {wl_str} nm")
    print(f"[achromatic] Pillars : {args.n_pillars}  period={args.period*1000:.0f} nm")

    # ── Achromatic optimisation ───────────────────────────────────────────────
    print(f"\n[achromatic] Gradient optimisation ({args.steps} steps) ...")
    widths_opt, phases_per_wl, amps_per_wl, losses = optimise_achromatic(
        net, stats, args.wavelengths, target_phases_all,
        args.period, args.steps, args.lr, device,
    )

    print(f"[achromatic] Final loss: {losses[-1]:.5f}")
    for wl, phi_t, phi_p, amp in zip(
            args.wavelengths, target_phases_all, phases_per_wl, amps_per_wl):
        mae = float(np.degrees(np.mean(np.abs(
            np.angle(np.exp(1j * (phi_p - phi_t)))
        ))))
        print(f"[achromatic]   λ={wl*1000:.0f} nm  "
              f"MAE phase={mae:.1f}°  mean|T|={amp.mean():.3f}")

    # ── NN baselines ──────────────────────────────────────────────────────────
    nn_results = nn_baselines(args.lib, target_phases_all, args.wavelengths)
    print(f"\n[achromatic] NN baseline (central λ library):")
    for wl, (_, phi_nn, amp_nn, err_nn) in zip(args.wavelengths, nn_results):
        print(f"[achromatic]   λ={wl*1000:.0f} nm  "
              f"MAE phase={err_nn.mean():.1f}°  mean|T|={amp_nn.mean():.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    npz_path = os.path.join(outdir, "achromatic_design.npz")
    np.savez(
        npz_path,
        x_um             = x_um,
        wavelengths      = np.array(args.wavelengths),
        widths_opt       = widths_opt,
        target_phases    = np.array(target_phases_all),
        phases_per_wl    = np.array(phases_per_wl),
        amps_per_wl      = np.array(amps_per_wl),
        widths_nn        = np.array([r[0] for r in nn_results]),
        phases_nn        = np.array([r[1] for r in nn_results]),
        amps_nn          = np.array([r[2] for r in nn_results]),
        opt_losses       = np.array(losses),
        period           = args.period,
        target_desc      = target_desc,
    )
    print(f"\n[achromatic] Saved → {npz_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_achromatic_design(
        x_um             = x_um,
        wavelengths      = args.wavelengths,
        target_phases    = target_phases_all,
        phases_per_wl    = phases_per_wl,
        phases_nn        = [r[1] for r in nn_results],
        widths_opt       = widths_opt,
        widths_nn        = nn_results[0][0],
        amps_per_wl      = amps_per_wl,
        opt_losses       = losses,
        target_desc      = target_desc,
        filename         = os.path.join(outdir, "achromatic_design.png"),
    )

    print(f"\n[achromatic] Done.  Outputs in: {outdir}")


if __name__ == "__main__":
    main()
