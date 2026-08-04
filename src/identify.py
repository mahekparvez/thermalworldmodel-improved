"""
Step 6a — identify the LPTN parameters against measured hardware data.

This is the step that turns assumed constants into estimated ones, and the first
point at which any number in this project is answerable to a motor rather than
to my own assumptions.

WHAT IS FITTED AND WHAT IS NOT
------------------------------
Only parameters marked `fit: true` in thermal_params.yaml. Capacitances (from
m*c_p), the copper temperature coefficient, and every published material
property are FIXED. Fitting a knowable constant to improve a residual would
destroy the only ground truth the model has, and would let the optimiser hide a
structural error inside a physically meaningless value.

Bounds are the `plausible_range` from the same file, and they are hard. In
particular every resistance is bounded strictly positive: a negative thermal
resistance transports heat up a gradient, and an optimiser will happily propose
one if it lowers the residual.

WHY TRANSIENT WINDOWS
---------------------
Steady-state data fixes only the PRODUCT R20 * R_eff, not the split. The
capacitances - and therefore the individual resistances - are only visible in
how fast the temperature moves. So fitting uses the most transient windows
available, which are also the hardest to fit.

INTEGRATOR
----------
The inner loop uses RK4 at dt = 5 s rather than the matrix exponential. The
fastest node time constant is ~50 s, so dt/tau ~ 0.1 - comfortably inside RK4's
stability region and far cheaper than an expm per step. `verify_integrator()`
checks RK4 against the exact propagator before any fitting happens, so this is a
measured claim rather than an assumption.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (NODE_COLUMN, OBSERVED_NODES, Window, load_raw, load_split,
                  make_windows, select_informative, summarise)
from lptn import IDX, N, NODES, Inputs, ThermalNetwork
from params import load as load_params

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "params" / "identified.json"


# ---------------------------------------------------------------------------
# Fast batched rollout
# ---------------------------------------------------------------------------

def _system_arrays(net: ThermalNetwork, w: Window) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute (A_t, b_t) for every step of a window.

    Building these once per window per objective evaluation is the whole cost of
    the fit; doing it inside the RK4 stages would quadruple it for no gain,
    since A is constant across a step by construction (zero-order hold).
    """
    return net.system_batch(w.i_d, w.i_q, w.speed_rpm, w.coolant_C, w.ambient_C)


def rollout_rk4(net: ThermalNetwork, w: Window) -> np.ndarray:
    """Free-running rollout over a window. Returns (n, 4) temperatures.

    Free-running means the model is fed its OWN previous output, never the
    measurement. Teacher-forced one-step error would be trivially small here and
    would say nothing about whether the model drifts.
    """
    A, b = _system_arrays(net, w)
    dt = w.dt
    T = w.initial_state()
    out = np.empty((len(w), N))
    out[0] = T
    for k in range(len(w) - 1):
        Ak, bk = A[k], b[k]
        k1 = Ak @ T + bk
        k2 = Ak @ (T + 0.5 * dt * k1) + bk
        k3 = Ak @ (T + 0.5 * dt * k2) + bk
        k4 = Ak @ (T + dt * k3) + bk
        T = T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = T
    return out


def verify_integrator(net: ThermalNetwork, w: Window) -> float:
    """Max |RK4 - exact matrix exponential| over a window, in Kelvin."""
    A, b = _system_arrays(net, w)
    dt = w.dt
    T_rk = w.initial_state()
    T_ex = w.initial_state()
    worst = 0.0
    for k in range(min(len(w) - 1, 400)):
        Ak, bk = A[k], b[k]
        k1 = Ak @ T_rk + bk
        k2 = Ak @ (T_rk + 0.5 * dt * k1) + bk
        k3 = Ak @ (T_rk + 0.5 * dt * k2) + bk
        k4 = Ak @ (T_rk + dt * k3) + bk
        T_rk = T_rk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        u = Inputs(i_d=w.i_d[k], i_q=w.i_q[k], speed_rpm=w.speed_rpm[k],
                   coolant_C=w.coolant_C[k], ambient_C=w.ambient_C[k],
                   flow_kg_s=net.nominal_flow)
        T_ex = net.step_exact(T_ex, u, dt)
        worst = max(worst, float(np.max(np.abs(T_rk - T_ex))))
    return worst


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

OBS_IDX = [IDX[n] for n in OBSERVED_NODES]


def residuals(net: ThermalNetwork, windows: Sequence[Window]) -> np.ndarray:
    parts = []
    for w in windows:
        pred = rollout_rk4(net, w)[:, OBS_IDX]
        parts.append((pred - w.T_meas).ravel())
    return np.concatenate(parts)


def per_node_errors(net: ThermalNetwork, windows: Sequence[Window]) -> Dict[str, Dict[str, float]]:
    err = {n: [] for n in OBSERVED_NODES}
    for w in windows:
        pred = rollout_rk4(net, w)[:, OBS_IDX]
        d = pred - w.T_meas
        for j, n in enumerate(OBSERVED_NODES):
            err[n].append(d[:, j])
    out = {}
    for n, chunks in err.items():
        e = np.concatenate(chunks)
        out[n] = {"mae_K": float(np.mean(np.abs(e))),
                  "rmse_K": float(np.sqrt(np.mean(e ** 2))),
                  "max_K": float(np.max(np.abs(e))),
                  "bias_K": float(np.mean(e))}
    allc = np.concatenate([np.concatenate(v) for v in err.values()])
    out["ALL"] = {"mae_K": float(np.mean(np.abs(allc))),
                  "rmse_K": float(np.sqrt(np.mean(allc ** 2))),
                  "max_K": float(np.max(np.abs(allc))),
                  "bias_K": float(np.mean(allc))}
    return out


def horizon_errors(net: ThermalNetwork, windows: Sequence[Window],
                   horizons: Sequence[int]) -> Dict[int, Dict[str, float]]:
    """Error as a function of how far the model has been left to run alone."""
    acc = {h: {n: [] for n in OBSERVED_NODES} for h in horizons}
    for w in windows:
        pred = rollout_rk4(net, w)[:, OBS_IDX]
        d = np.abs(pred - w.T_meas)
        for h in horizons:
            if h < len(w):
                for j, n in enumerate(OBSERVED_NODES):
                    acc[h][n].append(d[h, j])
    out = {}
    for h in horizons:
        per = {n: float(np.mean(v)) if v else float("nan")
               for n, v in acc[h].items()}
        flat = [x for v in acc[h].values() for x in v]
        per["ALL"] = float(np.mean(flat)) if flat else float("nan")
        out[h] = per
    return out


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

@dataclass
class FitSpec:
    keys: List[str]
    lo: np.ndarray
    hi: np.ndarray
    x0: np.ndarray
    logmask: np.ndarray      # True where the parameter is fitted in log space


def build_spec(ps) -> FitSpec:
    keys, lo, hi, x0, lg = [], [], [], [], []
    for p in ps.fittable():
        if p.plausible_range is None:
            continue
        keys.append(p.name)
        lo.append(p.plausible_range[0])
        hi.append(p.plausible_range[1])
        x0.append(p.value)
        lg.append(p.sampling == "loguniform")
    return FitSpec(keys, np.array(lo), np.array(hi), np.array(x0), np.array(lg))


def _to_free(spec: FitSpec, v: np.ndarray) -> np.ndarray:
    out = v.copy().astype(float)
    out[spec.logmask] = np.log(out[spec.logmask])
    return out


def _from_free(spec: FitSpec, z: np.ndarray) -> np.ndarray:
    out = z.copy().astype(float)
    out[spec.logmask] = np.exp(out[spec.logmask])
    return np.clip(out, spec.lo, spec.hi)


def identify(windows: Sequence[Window], max_nfev: int = 300,
             verbose: bool = True) -> Tuple[Dict[str, float], Dict]:
    ps = load_params()
    spec = build_spec(ps)
    base = ThermalNetwork(ps)

    if verbose:
        print(f"fitting {len(spec.keys)} parameters against {len(windows)} windows")
        drift = verify_integrator(base, windows[0])
        print(f"integrator check: max |RK4 - exact expm| = {drift:.3e} K "
              f"over {min(len(windows[0])-1,400)} steps at dt={windows[0].dt:.0f}s")

    n_eval = [0]
    t0 = time.time()

    def fun(z: np.ndarray) -> np.ndarray:
        vals = _from_free(spec, z)
        ov = dict(zip(spec.keys, vals))
        try:
            net = base.with_overrides(ov)
            r = residuals(net, windows)
        except Exception:
            # An infeasible proposal (e.g. a non-positive resistance) must be
            # pushed away from, not crashed on.
            r = np.full(sum(len(w) * len(OBSERVED_NODES) for w in windows), 1e3)
        n_eval[0] += 1
        if verbose and n_eval[0] % 25 == 0:
            print(f"  eval {n_eval[0]:4d}  rmse = {np.sqrt(np.mean(r**2)):7.3f} K"
                  f"  ({time.time()-t0:5.0f}s)")
        return r

    lo_f = _to_free(spec, spec.lo)
    hi_f = _to_free(spec, spec.hi)
    z0 = _to_free(spec, spec.x0)

    res = least_squares(fun, z0, bounds=(lo_f, hi_f), method="trf",
                        x_scale="jac", max_nfev=max_nfev, verbose=0,
                        diff_step=1e-3)

    fitted = dict(zip(spec.keys, _from_free(spec, res.x)))
    info = {
        "n_eval": int(n_eval[0]),
        "seconds": round(time.time() - t0, 1),
        "final_rmse_K": float(np.sqrt(np.mean(res.fun ** 2))),
        "status": int(res.status),
        "message": str(res.message),
    }
    return fitted, info


def flag_out_of_range(ps, fitted: Dict[str, float], tol: float = 1e-6) -> List[str]:
    """A parameter pinned at a bound means the model structure is suspect."""
    notes = []
    for k, v in fitted.items():
        p = ps[k]
        if p.plausible_range is None:
            continue
        lo, hi = p.plausible_range
        span = hi - lo
        if v <= lo + tol * span:
            notes.append(f"{k}: pinned at LOWER bound {lo:g} (fitted {v:g})")
        elif v >= hi - tol * span:
            notes.append(f"{k}: pinned at UPPER bound {hi:g} (fitted {v:g})")
    return notes


def main() -> None:
    fit_ids, test_ids = load_split()

    print("=" * 74)
    print("LPTN identification against measured hardware")
    print("=" * 74)

    df_fit = load_raw(fit_ids)
    # 1 h windows (720 steps at dt=5 s) so the 1000-step horizon in the report
    # is actually reachable; 40 min windows made it NaN.
    W_all = make_windows(df_fit, window_s=3600.0, stride_s=1800.0)
    W = select_informative(W_all, 48)
    print(summarise(W, "fit windows"))

    ps = load_params()
    base = ThermalNetwork(ps)

    print("\n--- before fitting (priors) ---")
    e0 = per_node_errors(base, W)
    for n in OBSERVED_NODES + ["ALL"]:
        print(f"  {n:8s} MAE {e0[n]['mae_K']:6.2f} K   RMSE {e0[n]['rmse_K']:6.2f} K"
              f"   max {e0[n]['max_K']:7.2f} K   bias {e0[n]['bias_K']:+6.2f} K")

    print("\n--- identifying ---")
    fitted, info = identify(W)
    print(f"  {info['n_eval']} evaluations in {info['seconds']}s -> "
          f"rmse {info['final_rmse_K']:.3f} K   ({info['message']})")

    net = base.with_overrides(fitted)

    print("\n--- fitted vs prior ---")
    print(f"  {'parameter':46s} {'prior':>10s} {'fitted':>10s} {'ratio':>7s}")
    for k, v in fitted.items():
        p = ps[k]
        ratio = f"{v/p.value:6.2f}x" if p.value else "  n/a "
        print(f"  {k:46s} {p.value:10.4g} {v:10.4g} {ratio:>7s}")

    notes = flag_out_of_range(ps, fitted)
    print("\n--- parameters pinned at a plausible-range bound ---")
    if notes:
        for nline in notes:
            print(f"  ! {nline}")
        print("  A parameter that had to reach its bound to fit the data means")
        print("  the model STRUCTURE is wrong, not the parameter.")
    else:
        print("  none - every fitted value landed strictly inside its prior range")

    print("\n--- after fitting, on the FIT windows ---")
    e1 = per_node_errors(net, W)
    for n in OBSERVED_NODES + ["ALL"]:
        print(f"  {n:8s} MAE {e1[n]['mae_K']:6.2f} K   RMSE {e1[n]['rmse_K']:6.2f} K"
              f"   max {e1[n]['max_K']:7.2f} K   bias {e1[n]['bias_K']:+6.2f} K")

    # ---- the reality gate: entire profiles never seen during fitting --------
    print("\n" + "=" * 74)
    print("REALITY GATE - held-out profiles, never seen during fitting")
    print("=" * 74)
    df_test = load_raw(test_ids)
    WT = make_windows(df_test, window_s=3600.0, stride_s=1800.0)
    print(summarise(WT, "held-out windows"))

    e2 = per_node_errors(net, WT)
    print("\n  per-node error on held-out data:")
    for n in OBSERVED_NODES + ["ALL"]:
        print(f"  {n:8s} MAE {e2[n]['mae_K']:6.2f} K   RMSE {e2[n]['rmse_K']:6.2f} K"
              f"   max {e2[n]['max_K']:7.2f} K   bias {e2[n]['bias_K']:+6.2f} K")

    print("\n  error vs rollout horizon (free-running, held-out):")
    hz = [1, 10, 100, 700]
    he = horizon_errors(net, WT, hz)
    hdr = "  " + "steps".ljust(8) + "".join(f"{n:>10s}" for n in OBSERVED_NODES + ["ALL"])
    print(hdr)
    for h in hz:
        row = f"  {h:<8d}" + "".join(f"{he[h][n]:10.2f}" for n in OBSERVED_NODES + ["ALL"])
        print(row + f"   ({h * WT[0].dt / 60:.0f} min)")

    OUT.write_text(json.dumps({
        "fitted": fitted, "info": info,
        "pinned_at_bound": notes,
        "fit_error": e1, "heldout_error": e2,
        "heldout_horizon_error": {str(k): v for k, v in he.items()},
        "n_fit_windows": len(W), "n_heldout_windows": len(WT),
        "fit_profiles": fit_ids, "test_profiles": test_ids,
    }, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
