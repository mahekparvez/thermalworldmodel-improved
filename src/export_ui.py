"""
Export everything the web page needs: both models run free over ENTIRE held-out
profiles, plus the parameter table and gate results.

Whole profiles, not the 1 h windows used for scoring. A profile runs up to 6 h,
and the model is given the measured temperatures once, at t = 0, and then left
alone. That is a harder test than the windowed metric in REPORT.md and it is the
one worth looking at, because drift over hours is what a thermal model is for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import NODE_COLUMN, OBSERVED_NODES, load_raw, load_split
from identify import OBS_IDX
from lptn import IDX, N, NODES, ThermalNetwork
from params import load as load_params
from tnn import ThermalNN, build_features
from train import priors_from_lptn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data.json"
DECIMATE = 10          # 2 Hz -> 0.2 Hz, dt = 5 s
PLOT_POINTS = 420      # per profile, after a second decimation for transport


def lptn_rollout(net: ThermalNetwork, d: pd.DataFrame, dt: float) -> np.ndarray:
    A, b = net.system_batch(d.i_d.to_numpy(), d.i_q.to_numpy(),
                            d.motor_speed.to_numpy(), d.coolant.to_numpy(),
                            d.ambient.to_numpy())
    T = np.array([d.stator_winding.iloc[0], d.stator_yoke.iloc[0],
                  d.pm.iloc[0], d.coolant.iloc[0]])
    out = np.empty((len(d), N))
    out[0] = T
    for k in range(len(d) - 1):
        Ak, bk = A[k], b[k]
        k1 = Ak @ T + bk
        k2 = Ak @ (T + 0.5 * dt * k1) + bk
        k3 = Ak @ (T + 0.5 * dt * k2) + bk
        k4 = Ak @ (T + dt * k3) + bk
        T = T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = T
    return out


@torch.no_grad()
def tnn_rollout(model: ThermalNN, d: pd.DataFrame, dt: float) -> np.ndarray:
    f = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(0)
    i_d, i_q = f(d.i_d.to_numpy()), f(d.i_q.to_numpy())
    spd, cool = f(d.motor_speed.to_numpy()), f(d.coolant.to_numpy())
    amb = f(d.ambient.to_numpy())
    seq = {"feats": build_features(i_d, i_q, spd, cool, amb),
           "i_sq": i_d ** 2 + i_q ** 2, "coolant": cool, "ambient": amb}
    T0 = torch.tensor([[d.stator_winding.iloc[0], d.stator_yoke.iloc[0],
                        d.pm.iloc[0], d.coolant.iloc[0]]], dtype=torch.float32)
    return model.rollout(T0, seq, dt, len(d) - 1)[0].numpy()


def r(x, n=2):
    return round(float(x), n)


def main() -> None:
    fit_ids, test_ids = load_split()
    ps = load_params()
    ident = json.loads((ROOT / "params" / "identified.json").read_text())
    tnn_res = json.loads((ROOT / "params" / "tnn_result.json").read_text())

    net = ThermalNetwork(ps, ident["fitted"])
    priors = priors_from_lptn()
    model = ThermalNN(cap_init=priors["cap"], cond_init=priors["cond"],
                      bnd_init=priors["bnd"], r_phase_init=priors["r_phase"])
    model.load_state_dict(torch.load(ROOT / "params" / "tnn.pt"))
    model.eval()

    df = load_raw(test_ids)
    dt = 0.5 * DECIMATE

    profiles = []
    for pid, g in df.groupby("profile_id"):
        g = g.iloc[::DECIMATE].reset_index(drop=True)
        if len(g) < 200:
            continue
        P_l = lptn_rollout(net, g, dt)
        P_t = tnn_rollout(model, g, dt)
        meas = np.column_stack([g[NODE_COLUMN[n]].to_numpy() for n in OBSERVED_NODES])

        step = max(1, len(g) // PLOT_POINTS)
        s = slice(None, None, step)
        t_min = (np.arange(len(g)) * dt / 60.0)[s]

        err_l = P_l[:, OBS_IDX] - meas
        err_t = P_t[:, OBS_IDX] - meas

        profiles.append({
            "id": int(pid),
            "minutes": r(len(g) * dt / 60.0, 1),
            "t": [r(v, 1) for v in t_min],
            "meas": {n: [r(v, 1) for v in meas[s, j]] for j, n in enumerate(OBSERVED_NODES)},
            "lptn": {n: [r(v, 1) for v in P_l[s, IDX[n]]] for n in OBSERVED_NODES},
            "tnn": {n: [r(v, 1) for v in P_t[s, IDX[n]]] for n in OBSERVED_NODES},
            "housing_lptn": [r(v, 1) for v in P_l[s, IDX["housing"]]],
            "tooth": [r(v, 1) for v in g.stator_tooth.to_numpy()[s]],
            "inputs": {
                "i_sq": [int(v) for v in (g.i_d.to_numpy()**2 + g.i_q.to_numpy()**2)[s]],
                "speed": [int(v) for v in g.motor_speed.to_numpy()[s]],
                "coolant": [r(v, 1) for v in g.coolant.to_numpy()[s]],
                "ambient": [r(v, 1) for v in g.ambient.to_numpy()[s]],
            },
            "mae": {
                "lptn": {n: r(np.mean(np.abs(err_l[:, j])), 3) for j, n in enumerate(OBSERVED_NODES)},
                "tnn": {n: r(np.mean(np.abs(err_t[:, j])), 3) for j, n in enumerate(OBSERVED_NODES)},
            },
            "max": {
                "lptn": {n: r(np.max(np.abs(err_l[:, j])), 2) for j, n in enumerate(OBSERVED_NODES)},
                "tnn": {n: r(np.max(np.abs(err_t[:, j])), 2) for j, n in enumerate(OBSERVED_NODES)},
            },
        })

    profiles.sort(key=lambda p: -p["minutes"])

    # whole-profile aggregate: the honest headline for a free 6 h run
    agg = {}
    for m in ("lptn", "tnn"):
        per = {}
        for n in OBSERVED_NODES:
            e = np.concatenate([np.abs(np.array(p["meas"][n]) - np.array(p[m][n]))
                                for p in profiles])
            per[n] = {"mae": r(e.mean(), 3), "max": r(e.max(), 2)}
        allc = np.concatenate([np.concatenate(
            [np.abs(np.array(p["meas"][n]) - np.array(p[m][n])) for n in OBSERVED_NODES])
            for p in profiles])
        per["ALL"] = {"mae": r(allc.mean(), 3), "max": r(allc.max(), 2)}
        agg[m] = per

    params = []
    for k, v in ident["fitted"].items():
        p = ps[k]
        lo, hi = p.plausible_range if p.plausible_range else (None, None)
        pinned = any(k in note for note in ident["pinned_at_bound"])
        params.append({
            "name": k, "prior": p.value, "fitted": r(v, 6),
            "lo": lo, "hi": hi, "pinned": pinned,
            "source": p.source, "units": p.units,
            "ratio": (r(v / p.value, 2) if p.value else None),
            "note": p.note[:240],
        })

    payload = {
        "meta": {
            "dt_s": dt,
            "n_profiles_total": 69,
            "n_fit": len(fit_ids), "n_test": len(test_ids),
            "test_ids": test_ids,
            "hours_total": 184.8,
            "dataset": "Paderborn LEA electric motor temperature (PMSM test bench)",
        },
        "aggregate_whole_profile": agg,
        "windowed": {
            "lptn": ident["heldout_error"],
            "tnn": tnn_res["heldout_error"],
        },
        "horizon": {
            "lptn": ident["heldout_horizon_error"],
            "tnn": tnn_res["heldout_horizon_error"],
        },
        "params": params,
        "pinned": ident["pinned_at_bound"],
        "tnn_learned": {
            "capacitances": [r(c, 0) for c in tnn_res["learned_capacitances_J_per_K"]],
            "r_phase": r(tnn_res["learned_R_phase_ohm"], 5),
            "resistances": {k: r(v, 5) for k, v in tnn_res["learned_resistances_at_ref"].items()},
            "n_params": 2631,
        },
        "lptn_capacitances": [r(c, 0) for c in net.C],
        "profiles": profiles,
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({kb:.0f} KB)")
    print(f"  {len(profiles)} held-out profiles, "
          f"{sum(p['minutes'] for p in profiles)/60:.1f} h")
    print("\nwhole-profile free-running MAE (K), model given t=0 only:")
    for n in OBSERVED_NODES + ["ALL"]:
        print(f"  {n:8s} LPTN {agg['lptn'][n]['mae']:6.2f}  "
              f"TNN {agg['tnn'][n]['mae']:6.2f}   "
              f"(max {agg['lptn'][n]['max']:6.1f} / {agg['tnn'][n]['max']:6.1f})")


if __name__ == "__main__":
    main()
