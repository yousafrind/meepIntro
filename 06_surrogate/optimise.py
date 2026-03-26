"""
Gradient-based metasurface inverse design using the trained surrogate MLP.

Given a target phase profile φ(x) (beam steering, metalens, or custom),
this script finds the pillar width at each position by minimising
the phase error through the differentiable surrogate — no FDTD calls needed.

The result is compared against the nearest-neighbour baseline from PhaseLibrary
to quantify the benefit of gradient-based optimisation.

Usage
-----
    # Beam-steering target (default, 30°)
    python run.py 06_surrogate/optimise.py

    # Metalens target
    python run.py 06_surrogate/optimise.py --target metalens --focal-len 10

    # Custom steering angle
    python run.py 06_surrogate/optimise.py --target beam --angle 45

    # Use a specific trained model
    python run.py 06_surrogate/optimise.py \\
        --model 06_surrogate/results/surrogate_model.pt \\
        --lib   01_beam_steering/results/phase_library.npz

    Options
    -------
    --model    06_surrogate/results/surrogate_model.pt
    --lib      01_beam_steering/results/phase_library.npz  (for NN baseline)
    --target   beam | metalens   target phase profile type
    --angle    30     beam steering angle in degrees
    --focal-len 10    focal length in μm (metalens only)
    --n-pillars 40    number of pillars in the metasurface
    --period   0.25   unit cell period in μm
    --steps    500    gradient descent steps
    --lr       0.05   optimiser learning rate
    --outdir   results

Output
------
    results/optimised_design.npz   pillar widths, phases, target comparison
    results/optimised_design.png   phase profile: surrogate vs NN baseline vs target
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
from utils.viz    import plot_optimised_design


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_MODEL = os.path.join(_HERE, "results", "surrogate_model.pt")
_DEFAULT_LIB   = os.path.join(
    _ROOT, "01_beam_steering", "results", "phase_library.npz"
)

N_PILLARS  = 40
PERIOD     = 0.25    # μm
ANGLE      = 30.0    # degrees
FOCAL_LEN  = 10.0    # μm
OPT_STEPS  = 500
OPT_LR     = 0.05
OUT_DIR    = "results"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Surrogate-based gradient optimisation of pillar widths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    default=_DEFAULT_MODEL,
                   help="Path to surrogate_model.pt")
    p.add_argument("--lib",      default=_DEFAULT_LIB,
                   help="Path to PhaseLibrary .npz (for NN-baseline comparison)")
    p.add_argument("--target",   default="beam", choices=["beam", "metalens"],
                   help="Target phase profile type")
    p.add_argument("--angle",    type=float, default=ANGLE,
                   help="Beam steering angle in degrees")
    p.add_argument("--focal-len",type=float, default=FOCAL_LEN, dest="focal_len",
                   help="Focal length in μm (metalens only)")
    p.add_argument("--n-pillars",type=int,   default=N_PILLARS, dest="n_pillars")
    p.add_argument("--period",   type=float, default=PERIOD)
    p.add_argument("--steps",    type=int,   default=OPT_STEPS)
    p.add_argument("--lr",       type=float, default=OPT_LR)
    p.add_argument("--outdir",   default=OUT_DIR)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET PHASE PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def beam_steering_phase(n_pillars, period, wavelength, angle_deg):
    """
    Linear phase gradient for beam steering to angle_deg from normal.
    φ(x) = k₀ · sin(θ) · x,  wrapped to (−π, π].
    """
    k0      = 2 * np.pi / wavelength
    x       = (np.arange(n_pillars) - (n_pillars - 1) / 2) * period
    phi     = k0 * np.sin(np.radians(angle_deg)) * x
    return phi, x


def metalens_phase(n_pillars, period, wavelength, focal_len):
    """
    Hyperbolic (aberration-free) metalens phase:
    φ(x) = k₀ · (f − √(x² + f²))
    """
    k0 = 2 * np.pi / wavelength
    x  = (np.arange(n_pillars) - (n_pillars - 1) / 2) * period
    phi = k0 * (focal_len - np.sqrt(x**2 + focal_len**2))
    return phi, x


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD SURROGATE
# ══════════════════════════════════════════════════════════════════════════════

def load_surrogate(model_path, device):
    """
    Load trained surrogate model from .pt file.
    Returns (net, stats, hidden, layers, dropout, libs).
    """
    import torch
    ckpt = torch.load(model_path, map_location=device)

    from train import build_model
    net = build_model(ckpt["hidden"], ckpt["layers"], ckpt["dropout"], device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    return net, ckpt["stats"], ckpt["libs"]


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIENT-BASED OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def optimise_widths(net, stats, target_phases, period,
                    n_steps, lr, device,
                    width_min_frac=0.05, width_max_frac=0.90):
    """
    Minimise phase MSE through the differentiable surrogate.

    Parameters
    ----------
    net            : trained SurrogateNet
    stats          : dict with w_mean, w_std (from training normalisation)
    target_phases  : ndarray [N]  desired phases in radians
    period         : float  unit cell period in μm
    n_steps        : int    gradient descent steps
    lr             : float  Adam learning rate
    device         : torch.device

    Returns
    -------
    widths_opt : ndarray [N]  optimised pillar widths in μm
    phases_opt : ndarray [N]  surrogate-predicted phases at optimised widths
    amps_opt   : ndarray [N]  surrogate-predicted amplitudes
    losses     : list         loss per step
    """
    import torch
    import torch.nn as nn

    N      = len(target_phases)
    w_mean = stats["w_mean"]
    w_std  = stats["w_std"]

    # Initialise: use centre of valid width range
    w_init_frac = (width_min_frac + width_max_frac) / 2.0
    w_norm_init = (w_init_frac - w_mean) / w_std

    # Free parameter: logit of fractional width (enforces w ∈ [w_min, w_max])
    w_min_norm = (width_min_frac - w_mean) / w_std
    w_max_norm = (width_max_frac - w_mean) / w_std

    # Parameterise as sigmoid in normalised space so bounds are always respected
    def to_norm(logit_w):
        return w_min_norm + (w_max_norm - w_min_norm) * torch.sigmoid(logit_w)

    logit_init = torch.zeros(N, 1, device=device)
    logit_w    = nn.Parameter(logit_init)

    phi_target = torch.tensor(target_phases, dtype=torch.float32, device=device)

    opt = torch.optim.Adam([logit_w], lr=lr)

    losses = []

    for step in range(n_steps):
        w_norm = to_norm(logit_w)              # [N, 1], in normalised space
        pred   = net(w_norm)                   # [N, 3]: amp, sin∠T, cos∠T
        sin_pred = pred[:, 1]
        cos_pred = pred[:, 2]

        # Phase error via sin/cos — avoids wrapping issues
        sin_tgt = torch.sin(phi_target)
        cos_tgt = torch.cos(phi_target)

        loss = ((sin_pred - sin_tgt)**2 + (cos_pred - cos_tgt)**2).mean()

        # Regularise: penalise low amplitude
        amp_penalty = (1.0 - pred[:, 0]).pow(2).mean()
        loss = loss + 0.1 * amp_penalty

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    # Extract results
    net.eval()
    with torch.no_grad():
        w_norm_final = to_norm(logit_w)
        pred_final   = net(w_norm_final).cpu().numpy()

    w_frac_final  = (w_norm_final.detach().cpu().numpy().squeeze()
                     * w_std + w_mean)
    widths_opt    = np.clip(w_frac_final, width_min_frac, width_max_frac) * period
    amps_opt      = pred_final[:, 0]
    phases_opt    = np.arctan2(pred_final[:, 1], pred_final[:, 2])

    return widths_opt, phases_opt, amps_opt, losses


# ══════════════════════════════════════════════════════════════════════════════
#  NEAREST-NEIGHBOUR BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def nn_baseline(lib, target_phases):
    """Nearest-neighbour width assignment from PhaseLibrary."""
    widths_nn, errors_nn = lib.assign_widths(target_phases)
    # Look up predicted amplitudes/phases at assigned widths
    idx = np.array([
        int(np.argmin(np.abs(lib.widths - w))) for w in widths_nn
    ])
    amps_nn   = lib.amplitudes[idx]
    phases_nn = lib.phases[idx]
    return widths_nn, phases_nn, amps_nn, errors_nn


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
            "Install with: pip install torch"
        )

    device = get_torch_device() or torch.device("cpu")
    print(f"\n[optimise] device = {device}")

    # Load surrogate
    print(f"[optimise] Loading surrogate: {args.model}")
    net, stats, train_libs = load_surrogate(args.model, device)
    print(f"[optimise] Trained on: {train_libs}")

    # Load library for NN baseline (infer wavelength/period from it)
    print(f"[optimise] Loading phase library: {args.lib}")
    lib = PhaseLibrary.load(args.lib)
    wavelength = lib.wavelength
    period     = args.period

    # Build target phase profile
    if args.target == "beam":
        target_phases, x_um = beam_steering_phase(
            args.n_pillars, period, wavelength, args.angle
        )
        target_desc = f"beam steering  θ = {args.angle:.0f}°"
    else:
        target_phases, x_um = metalens_phase(
            args.n_pillars, period, wavelength, args.focal_len
        )
        target_desc = f"metalens  f = {args.focal_len:.0f} μm"

    print(f"\n[optimise] Target : {target_desc}")
    print(f"[optimise] Pillars: {args.n_pillars}  period={period*1000:.0f} nm  "
          f"λ={wavelength*1000:.0f} nm")

    # ── Gradient-based optimisation ──────────────────────────────────────────
    print(f"\n[optimise] Running gradient descent ({args.steps} steps) ...")
    widths_opt, phases_opt, amps_opt, losses = optimise_widths(
        net, stats, target_phases, period,
        n_steps=args.steps, lr=args.lr, device=device,
    )

    phase_err_opt = np.degrees(np.abs(np.angle(
        np.exp(1j * (phases_opt - target_phases))
    )))
    print(f"[optimise] Surrogate result  "
          f"MAE phase = {phase_err_opt.mean():.1f}°  "
          f"mean |T| = {amps_opt.mean():.3f}")

    # ── Nearest-neighbour baseline ────────────────────────────────────────────
    widths_nn, phases_nn, amps_nn, errors_nn = nn_baseline(lib, target_phases)
    print(f"[optimise] NN baseline       "
          f"MAE phase = {errors_nn.mean():.1f}°  "
          f"mean |T| = {amps_nn.mean():.3f}")

    # ── Save results ─────────────────────────────────────────────────────────
    npz_path = os.path.join(outdir, "optimised_design.npz")
    np.savez(
        npz_path,
        x_um          = x_um,
        target_phases = target_phases,
        widths_opt    = widths_opt,
        phases_opt    = phases_opt,
        amps_opt      = amps_opt,
        widths_nn     = widths_nn,
        phases_nn     = phases_nn,
        amps_nn       = amps_nn,
        opt_losses    = np.array(losses),
        period        = period,
        wavelength    = wavelength,
        target_desc   = target_desc,
    )
    print(f"[optimise] Saved → {npz_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_optimised_design(
        x_um          = x_um,
        target_phases = target_phases,
        phases_opt    = phases_opt,
        phases_nn     = phases_nn,
        widths_opt    = widths_opt,
        widths_nn     = widths_nn,
        amps_opt      = amps_opt,
        amps_nn       = amps_nn,
        opt_losses    = losses,
        target_desc   = target_desc,
        filename      = os.path.join(outdir, "optimised_design.png"),
    )

    print(f"\n[optimise] Done.  Outputs in: {outdir}")


if __name__ == "__main__":
    main()
