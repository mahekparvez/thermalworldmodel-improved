"""
Paderborn LEA dataset loading, node mapping, and window extraction.

WHAT THE DATASET GIVES AND WHAT THE MODEL HAS
---------------------------------------------
Measured temperatures:  stator_winding, stator_tooth, stator_yoke, pm, coolant, ambient
Model nodes:            winding,        --           yoke,         magnet,  (boundary), (boundary)

Two mismatches, both stated rather than papered over:

  * `stator_tooth` is measured but NOT modelled. A four-node network lumps the
    tooth into the yoke. The tooth runs hotter than the yoke (it is closer to
    the winding), so the model's single "yoke" node should land between the two
    measurements. That is a structural limitation, not a fitting failure.

  * `housing` is modelled but NOT measured. It is a latent state. It is tightly
    coupled to the coolant (housing->coolant is the smallest resistance in the
    network), so initialising it at the measured coolant temperature is a good
    approximation, and any error in it decays with the housing time constant.
    But it is weakly observable, and its identified capacitance should be
    treated with more suspicion than the others.

WINDOWING
---------
Fitting uses windows drawn from whole profiles. Two rules:

  1. Never split a profile across fit/test. Adjacent 2 Hz samples are nearly
     identical, so a timestep-level split leaks almost perfectly.
  2. Prefer TRANSIENT windows. Steady-state data cannot separate R20 from
     R_eff - only their product is identifiable (see thermal_params.yaml). The
     capacitances, and hence the split, only show up in how fast temperature
     moves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "measures_v2.csv"
SPLIT = ROOT / "params" / "data_split.json"

SAMPLE_HZ = 2.0
DT_RAW = 1.0 / SAMPLE_HZ

# model node -> measured column. None = latent (not measured).
NODE_COLUMN: Dict[str, Optional[str]] = {
    "winding": "stator_winding",
    "yoke": "stator_yoke",
    "magnet": "pm",
    "housing": None,
}
OBSERVED_NODES = [n for n, c in NODE_COLUMN.items() if c is not None]


@dataclass
class Window:
    """One contiguous slice of a profile, resampled to a uniform step."""
    profile_id: int
    dt: float
    # inputs, one row per step
    i_d: np.ndarray
    i_q: np.ndarray
    speed_rpm: np.ndarray
    coolant_C: np.ndarray
    ambient_C: np.ndarray
    # measured temperatures, one row per step, columns in OBSERVED_NODES order
    T_meas: np.ndarray
    # measured tooth, carried for diagnostics only (not modelled)
    tooth_C: np.ndarray

    def __len__(self) -> int:
        return len(self.i_d)

    @property
    def duration_s(self) -> float:
        return len(self) * self.dt

    def initial_state(self) -> np.ndarray:
        """T0 for all four nodes. Housing is latent -> start it at coolant."""
        from lptn import NODES
        T0 = np.empty(len(NODES))
        for k, n in enumerate(NODES):
            col = NODE_COLUMN[n]
            if col is None:
                T0[k] = self.coolant_C[0]
            else:
                T0[k] = self.T_meas[0, OBSERVED_NODES.index(n)]
        return T0

    def transient_score(self) -> float:
        """Total absolute temperature movement, K. Higher = more informative."""
        return float(np.sum(np.abs(np.diff(self.T_meas, axis=0))))


def load_split() -> Tuple[List[int], List[int]]:
    s = json.loads(SPLIT.read_text())
    return s["fit_profiles"], s["test_profiles"]


def load_raw(profiles: Optional[Sequence[int]] = None) -> pd.DataFrame:
    df = pd.read_csv(CSV)
    if profiles is not None:
        df = df[df.profile_id.isin(list(profiles))]
    return df.reset_index(drop=True)


def make_windows(df: pd.DataFrame, window_s: float = 2400.0,
                 stride_s: float = 1200.0, decimate: int = 10,
                 min_transient_K: float = 0.0) -> List[Window]:
    """Cut profiles into overlapping windows and decimate.

    `decimate=10` takes the 2 Hz data to 0.2 Hz (5 s steps). The fastest node
    time constant is ~50 s, so that is still ~10 samples per time constant -
    ample - while cutting the rollout cost by 10x. Decimation takes every Nth
    sample rather than averaging, so the inputs remain instantaneous values
    consistent with a zero-order hold.
    """
    dt = DT_RAW * decimate
    n_win = int(round(window_s / dt))
    n_str = max(1, int(round(stride_s / dt)))
    out: List[Window] = []

    for pid, g in df.groupby("profile_id"):
        g = g.iloc[::decimate].reset_index(drop=True)
        if len(g) < n_win:
            continue
        for start in range(0, len(g) - n_win + 1, n_str):
            sl = g.iloc[start:start + n_win]
            T = np.column_stack([sl[NODE_COLUMN[n]].to_numpy() for n in OBSERVED_NODES])
            w = Window(
                profile_id=int(pid), dt=dt,
                i_d=sl.i_d.to_numpy(), i_q=sl.i_q.to_numpy(),
                speed_rpm=sl.motor_speed.to_numpy(),
                coolant_C=sl.coolant.to_numpy(), ambient_C=sl.ambient.to_numpy(),
                T_meas=T, tooth_C=sl.stator_tooth.to_numpy(),
            )
            if w.transient_score() >= min_transient_K:
                out.append(w)
    return out


def select_informative(windows: Sequence[Window], k: int,
                       seed: int = 0) -> List[Window]:
    """Keep the k most transient windows, spread across profiles.

    Steady windows carry almost no information about the capacitances, and the
    R20/R_eff degeneracy is only broken by transients. Picking the most active
    windows is not cherry-picking the easy cases - it is the opposite, these are
    the hardest to fit and the only ones that constrain the dynamics.
    """
    rng = np.random.default_rng(seed)
    by_pid: Dict[int, List[Window]] = {}
    for w in windows:
        by_pid.setdefault(w.profile_id, []).append(w)
    for pid in by_pid:
        by_pid[pid].sort(key=lambda w: -w.transient_score())

    chosen: List[Window] = []
    pids = sorted(by_pid)
    idx = {p: 0 for p in pids}
    while len(chosen) < k:
        progressed = False
        for p in pids:
            if idx[p] < len(by_pid[p]) and len(chosen) < k:
                chosen.append(by_pid[p][idx[p]])
                idx[p] += 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(chosen)
    return chosen


def summarise(windows: Sequence[Window], label: str) -> str:
    if not windows:
        return f"{label}: none"
    tr = np.array([w.transient_score() for w in windows])
    hrs = sum(w.duration_s for w in windows) / 3600.0
    return (f"{label}: {len(windows)} windows across "
            f"{len({w.profile_id for w in windows})} profiles, {hrs:.1f} h, "
            f"dt={windows[0].dt:.1f}s, transient score "
            f"min/med/max = {tr.min():.0f}/{np.median(tr):.0f}/{tr.max():.0f} K")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    fit_ids, test_ids = load_split()
    print(f"fit profiles:  {len(fit_ids)}")
    print(f"test profiles: {len(test_ids)}  {test_ids}")

    df = load_raw(fit_ids)
    W = make_windows(df)
    print("\n" + summarise(W, "all fit windows"))
    sel = select_informative(W, 60)
    print(summarise(sel, "selected (most transient)"))

    w = sel[0]
    print(f"\nexample window: profile {w.profile_id}, {len(w)} steps, "
          f"{w.duration_s/60:.0f} min")
    print(f"  T0 (winding,yoke,magnet,housing) = "
          f"{np.round(w.initial_state(), 2)}")
    print(f"  winding range {w.T_meas[:,0].min():.1f} - {w.T_meas[:,0].max():.1f} C")
    print(f"  tooth  range {w.tooth_C.min():.1f} - {w.tooth_C.max():.1f} C  "
          "(measured, NOT modelled - lumped into yoke)")
    print(f"  yoke   range {w.T_meas[:,1].min():.1f} - {w.T_meas[:,1].max():.1f} C")
    print("\nnote: model 'yoke' should land between measured yoke and tooth, "
          "since a 4-node network lumps them.")
