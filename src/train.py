"""
Step 5 — train the Thermal Neural Network on MEASURED data.

Non-negotiables from the brief, and why each one matters:

  MULTI-STEP ROLLOUT LOSS. The model is fed its own output, never the
  measurement, for N steps. Single-step (teacher-forced) loss on 2 Hz thermal
  data is trivially easy - temperatures barely move in 0.5 s, so a model that
  predicts "same as last time" scores well and drifts catastrophically over an
  hour. N is scheduled from 8 up to 360 steps (30 min at dt=5 s).

  DRIFT PENALTY. An explicit term on how fast error GROWS with horizon, not just
  its average. Two models with the same mean error are not equally useful if one
  is flat and the other is a ramp.

  PHYSICS RESIDUAL. Energy balance is enforced structurally by the architecture
  (see tnn.py), so the residual term here is a CHECK, not a crutch: it should
  stay at machine precision, and if it ever does not, the architecture broke.

  EARLY STOPPING on validation ROLLOUT error, not training loss, checkpointing
  the best.

  PER-NODE ERROR IN KELVIN. Normalised loss is not reportable.

Validation profiles are carved out of the FIT split, never from the held-out
test profiles - those are touched exactly once, at the very end, by the reality
gate.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (OBSERVED_NODES, Window, load_raw, load_split, make_windows,
                  select_informative, summarise)
from lptn import IDX
from tnn import N, ThermalNN, build_features

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "params" / "tnn.pt"
OUT = ROOT / "params" / "tnn_result.json"

OBS_IDX = [IDX[n] for n in OBSERVED_NODES]
DEV = torch.device("cpu")


# ---------------------------------------------------------------------------
# Tensorising
# ---------------------------------------------------------------------------

def to_tensors(windows: Sequence[Window]) -> Dict[str, torch.Tensor]:
    """Stack equal-length windows into batched tensors."""
    n = len(windows[0])
    assert all(len(w) == n for w in windows), "windows must be equal length"
    f = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=DEV)
    i_d = f([w.i_d for w in windows])
    i_q = f([w.i_q for w in windows])
    spd = f([w.speed_rpm for w in windows])
    cool = f([w.coolant_C for w in windows])
    amb = f([w.ambient_C for w in windows])
    return {
        "feats": build_features(i_d, i_q, spd, cool, amb),   # (B, n, 7)
        "i_sq": i_d ** 2 + i_q ** 2,                          # (B, n) in A^2
        "coolant": cool, "ambient": amb,
        "T_meas": f([w.T_meas for w in windows]),             # (B, n, 3)
        "T0": f([w.initial_state() for w in windows]),        # (B, 4)
    }


def slice_batch(D: Dict[str, torch.Tensor], idx: torch.Tensor,
                start: int, horizon: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Sub-batch starting at `start` for `horizon` steps.

    The initial state for a mid-window start comes from the MEASUREMENT for the
    three observed nodes; housing is latent, so it is seeded from the coolant.
    That is the same assumption the LPTN baseline makes, so the comparison is
    like for like.
    """
    seq = {k: D[k][idx, start:start + horizon] for k in ("feats", "i_sq", "coolant", "ambient")}
    meas = D["T_meas"][idx, start:start + horizon + 1]
    T0 = torch.empty(len(idx), N, device=DEV)
    T0[:, OBS_IDX] = meas[:, 0]
    T0[:, IDX["housing"]] = D["coolant"][idx, start]
    return seq, T0, meas


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def rollout_loss(model: ThermalNN, seq, T0, meas, dt: float,
                 drift_weight: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    horizon = seq["feats"].shape[1]
    pred = model.rollout(T0, seq, dt, horizon)          # (B, H+1, 4)
    err = pred[:, :, OBS_IDX] - meas                    # (B, H+1, 3)
    abs_err = err.abs()

    # Huber in Kelvin: quadratic near zero, linear beyond 3 K, so a handful of
    # bad windows cannot dominate the gradient.
    delta = 3.0
    huber = torch.where(abs_err < delta, 0.5 * err ** 2,
                        delta * (abs_err - 0.5 * delta))
    main = huber.mean()

    # Drift: penalise the GROWTH of error across the horizon, not its level.
    # Compares the last fifth of the rollout against the first fifth.
    q = max(1, (horizon + 1) // 5)
    early = abs_err[:, :q].mean()
    late = abs_err[:, -q:].mean()
    drift = torch.relu(late - early)

    loss = main + drift_weight * drift
    return loss, {"huber": float(main), "drift_K": float(drift),
                  "mae_K": float(abs_err.mean())}


@torch.no_grad()
def evaluate(model: ThermalNN, D: Dict[str, torch.Tensor], dt: float,
             horizon: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Free-running rollout over whole windows; per-node error in Kelvin."""
    B, n = D["T_meas"].shape[0], D["T_meas"].shape[1]
    H = (n - 1) if horizon is None else min(horizon, n - 1)
    idx = torch.arange(B)
    seq, T0, meas = slice_batch(D, idx, 0, H)
    pred = model.rollout(T0, seq, dt, H)[:, :, OBS_IDX]
    err = (pred - meas).cpu().numpy()
    out = {}
    for j, nm in enumerate(OBSERVED_NODES):
        e = err[:, :, j]
        out[nm] = {"mae_K": float(np.mean(np.abs(e))),
                   "rmse_K": float(np.sqrt(np.mean(e ** 2))),
                   "max_K": float(np.max(np.abs(e))),
                   "bias_K": float(np.mean(e))}
    out["ALL"] = {"mae_K": float(np.mean(np.abs(err))),
                  "rmse_K": float(np.sqrt(np.mean(err ** 2))),
                  "max_K": float(np.max(np.abs(err))),
                  "bias_K": float(np.mean(err))}
    return out


@torch.no_grad()
def horizon_errors(model: ThermalNN, D: Dict[str, torch.Tensor], dt: float,
                   horizons: Sequence[int]) -> Dict[int, Dict[str, float]]:
    B, n = D["T_meas"].shape[0], D["T_meas"].shape[1]
    H = min(max(horizons), n - 1)
    idx = torch.arange(B)
    seq, T0, meas = slice_batch(D, idx, 0, H)
    pred = model.rollout(T0, seq, dt, H)[:, :, OBS_IDX]
    ae = (pred - meas).abs().cpu().numpy()
    out = {}
    for h in horizons:
        if h > H:
            out[h] = {nm: float("nan") for nm in OBSERVED_NODES + ["ALL"]}
            continue
        per = {nm: float(np.mean(ae[:, h, j])) for j, nm in enumerate(OBSERVED_NODES)}
        per["ALL"] = float(np.mean(ae[:, h, :]))
        out[h] = per
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def schedule_horizon(epoch: int, n_epochs: int, lo: int, hi: int) -> int:
    """Scheduled sampling: short rollouts first, then progressively longer.

    Starting long is unstable - an untrained model diverges within a few steps
    and the gradient is meaningless. Starting short and never growing is the
    failure the brief warns about, because drift never enters the loss.
    """
    frac = min(1.0, epoch / max(1, int(0.7 * n_epochs)))
    return int(round(lo * (hi / lo) ** frac))


def train(train_D, val_D, dt: float, priors: Dict[str, float],
          n_epochs: int = 60, batch: int = 24, lr: float = 3e-3,
          patience: int = 12, seed: int = 0, verbose: bool = True) -> ThermalNN:
    torch.manual_seed(seed)

    model = ThermalNN(
        cap_init=priors["cap"], cond_init=priors["cond"],
        bnd_init=priors["bnd"], r_phase_init=priors["r_phase"],
    ).to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    B, n = train_D["T_meas"].shape[0], train_D["T_meas"].shape[1]
    best = math.inf
    best_state = None
    since = 0
    t0 = time.time()

    for ep in range(n_epochs):
        H = schedule_horizon(ep, n_epochs, 8, min(360, n - 1))
        model.train()
        perm = torch.randperm(B)
        ep_stats = []
        for s in range(0, B, batch):
            idx = perm[s:s + batch]
            max_start = max(0, n - 1 - H)
            start = int(torch.randint(0, max_start + 1, (1,)))
            seq, T0, meas = slice_batch(train_D, idx, start, H)
            loss, stats = rollout_loss(model, seq, T0, meas, dt)
            opt.zero_grad()
            loss.backward()
            # Rollouts of a few hundred steps can produce very large gradients
            # early on; clipping keeps the first epochs from destroying the
            # physically-initialised parameters.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_stats.append(stats)
        sched.step()

        model.eval()
        # Early stopping on VALIDATION ROLLOUT error, not training loss.
        val = evaluate(model, val_D, dt, horizon=min(360, n - 1))
        v = val["ALL"]["mae_K"]
        if v < best - 1e-4:
            best, since = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since += 1

        if verbose and (ep % 5 == 0 or ep == n_epochs - 1):
            tr_mae = float(np.mean([s["mae_K"] for s in ep_stats]))
            dr = float(np.mean([s["drift_K"] for s in ep_stats]))
            print(f"  ep {ep:3d}  H={H:3d}  train MAE {tr_mae:6.2f} K  "
                  f"drift {dr:5.2f} K  val MAE {v:6.2f} K  "
                  f"{'*' if since == 0 else ' '}  ({time.time()-t0:5.0f}s)")

        if since >= patience:
            if verbose:
                print(f"  early stop at epoch {ep} (no val improvement for {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"  best validation rollout MAE: {best:.3f} K")
    return model


def priors_from_lptn() -> Dict[str, float]:
    """Initialise the TNN at the identified LPTN parameters.

    Not a shortcut: it means training starts from a physically calibrated point
    and any improvement is attributable to the learned operating-point
    dependence, which is exactly the question the comparison is asking.
    """
    from lptn import ThermalNetwork
    from params import load as load_params
    ps = load_params()
    ident = json.loads((ROOT / "params" / "identified.json").read_text())["fitted"]
    net = ThermalNetwork(ps, ident)
    from lptn import Inputs
    u = Inputs(i_d=-70, i_q=60, speed_rpm=2500, coolant_C=35, ambient_C=25,
               flow_kg_s=net.nominal_flow)
    r = net.resistances(u)
    return {
        "cap": [float(c) for c in net.C],
        "cond": [1.0 / r["winding_yoke"], 1.0 / r["yoke_housing"],
                 1.0 / r["magnet_winding_gap"], 1.0 / r["magnet_shaft_housing"]],
        "bnd": [1.0 / r["housing_coolant"], 1.0 / r["housing_ambient"]],
        "r_phase": float(ident.get("losses.copper.phase_resistance_20C", 0.02)),
    }


def main() -> None:
    fit_ids, test_ids = load_split()
    rng = np.random.default_rng(11)

    # Validation profiles come out of the FIT split. The test profiles are not
    # touched until the reality gate at the end.
    fit_ids = list(fit_ids)
    rng.shuffle(fit_ids)
    val_ids = sorted(fit_ids[:10])
    tr_ids = sorted(fit_ids[10:])
    print(f"train profiles {len(tr_ids)}   val profiles {len(val_ids)}   "
          f"held-out {len(test_ids)}")

    WIN_S, STRIDE_S = 3600.0, 1800.0     # 1 h windows -> 720 steps at dt=5 s,
                                         # long enough for a 1000-step horizon
    W_tr = select_informative(make_windows(load_raw(tr_ids), WIN_S, STRIDE_S), 140)
    W_va = select_informative(make_windows(load_raw(val_ids), WIN_S, STRIDE_S), 40)
    print(summarise(W_tr, "train windows"))
    print(summarise(W_va, "val windows"))

    dt = W_tr[0].dt
    D_tr, D_va = to_tensors(W_tr), to_tensors(W_va)

    priors = priors_from_lptn()
    print(f"\ninitialising at the identified LPTN parameters: "
          f"C={np.round(priors['cap'],0)}, R_phase={priors['r_phase']:.4f} ohm")

    print("\n--- training ---")
    model = train(D_tr, D_va, dt, priors)

    # Architectural guarantee, verified rather than assumed.
    idx = torch.arange(min(16, D_va["T_meas"].shape[0]))
    seq, T0, _ = slice_batch(D_va, idx, 0, 1)
    res = model.energy_residual(T0, seq["feats"][:, 0], seq["i_sq"][:, 0],
                                seq["coolant"][:, 0], seq["ambient"][:, 0])
    print(f"\nenergy-balance residual (structural, should be ~0): "
          f"max |r| = {float(res.abs().max()):.3e} W")

    torch.save(model.state_dict(), CKPT)

    print("\n" + "=" * 74)
    print("REALITY GATE - held-out profiles, never seen in training or validation")
    print("=" * 74)
    W_te = make_windows(load_raw(test_ids), WIN_S, STRIDE_S)
    print(summarise(W_te, "held-out windows"))
    D_te = to_tensors(W_te)

    err = evaluate(model, D_te, dt)
    print("\n  per-node error, free-running over the whole 1 h window:")
    for nm in OBSERVED_NODES + ["ALL"]:
        e = err[nm]
        print(f"  {nm:8s} MAE {e['mae_K']:6.2f} K   RMSE {e['rmse_K']:6.2f} K"
              f"   max {e['max_K']:7.2f} K   bias {e['bias_K']:+6.2f} K")

    hz = [1, 10, 100, 700]
    he = horizon_errors(model, D_te, dt, hz)
    print("\n  error vs rollout horizon (free-running, held-out):")
    print("  " + "steps".ljust(8) + "".join(f"{n:>10s}" for n in OBSERVED_NODES + ["ALL"]))
    for h in hz:
        print(f"  {h:<8d}" + "".join(f"{he[h][n]:10.2f}" for n in OBSERVED_NODES + ["ALL"])
              + f"   ({h*dt/60:.0f} min)")

    OUT.write_text(json.dumps({
        "heldout_error": err,
        "heldout_horizon_error": {str(k): v for k, v in he.items()},
        "learned_capacitances_J_per_K": [float(c) for c in model.capacitances()],
        "learned_R_phase_ohm": float(torch.exp(model.log_r_phase)),
        "learned_resistances_at_ref": model.resistances_at(seq["feats"][:, 0]),
        "n_train_windows": len(W_tr), "n_val_windows": len(W_va),
        "n_heldout_windows": len(W_te),
        "train_profiles": tr_ids, "val_profiles": val_ids,
        "test_profiles": test_ids,
    }, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
