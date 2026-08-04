"""
Thermal Neural Network — a learned world model with the LPTN's structure baked in.

After Kirchgassner, Wallscheid & Bocker, "Thermal Neural Networks:
Lumped-Parameter Thermal Modeling with State-Space Machine Learning"
(arXiv:2103.16323).

WHY THIS RATHER THAN AN MLP ON SYNTHETIC ROLLOUTS
-------------------------------------------------
Training a network to predict the next state of a hand-written RC model teaches
it to reproduce that model. Domain randomisation broadens it to a FAMILY of RC
models, which is better, but the network still never sees a motor. The resulting
metrics measure whether one piece of code can imitate another.

A TNN instead keeps the network structure - nodes, conductances, capacitances,
the energy balance - and learns the PARAMETERS as functions of the operating
point, directly from measured data. Three consequences:

  1. It is trained on measurements, so it can be wrong about a real motor and
     the error will show. There is no circle.

  2. It CANNOT violate the first or second law. The state update is
        dT/dt = C^-1 [ P(u) + sum_j g_ij(u) (T_j - T_i) ]
     with g_ij >= 0 and C > 0 enforced by construction (softplus). Conservation
     is a property of the architecture, not something the loss has to police.
     The physics gate becomes a structural guarantee rather than a test.

  3. Everything it learns is in physical units - K/W, J/K, W - and can be
     compared against the priors in thermal_params.yaml. A black-box MLP offers
     no such check.

WHAT IS LEARNED
---------------
  g_ij(u)  four inter-node conductances, each a small MLP of the operating point.
           This is what lets the air gap open up with speed and the coolant path
           stiffen as flow drops, without either being hand-coded.
  g_i,bnd  two boundary conductances (housing->coolant, housing->ambient).
  P_i(u)   loss injected at each node. The copper term keeps its EXPLICIT
           I^2 R(T) temperature feedback rather than learning it, because that
           feedback is known exactly and learning it would waste capacity and
           discard a guarantee.
  C_i      four capacitances, learned as scalars (they do not depend on the
           operating point - mass and specific heat are constants).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

NODES = ("winding", "yoke", "magnet", "housing")
N = len(NODES)
IDX = {n: i for i, n in enumerate(NODES)}

# Which node pairs are physically connected. Same topology as the LPTN: a
# learned model is not licensed to invent a conduction path that does not exist.
LINKS: Tuple[Tuple[int, int], ...] = (
    (IDX["winding"], IDX["yoke"]),
    (IDX["yoke"], IDX["housing"]),
    (IDX["magnet"], IDX["winding"]),      # across the air gap
    (IDX["magnet"], IDX["housing"]),      # via shaft and bearings
)
# Boundary couplings: (node, boundary index) where boundary 0=coolant, 1=ambient
BOUNDARIES: Tuple[Tuple[int, int], ...] = (
    (IDX["housing"], 0),
    (IDX["housing"], 1),
)

ALPHA_CU = 3.93e-3
T_REF_CU = 20.0

# Input feature layout, all normalised before use:
#   0 i_d, 1 i_q, 2 i^2, 3 speed, 4 |speed|, 5 coolant, 6 ambient
N_FEAT = 7


def build_features(i_d, i_q, speed_rpm, coolant_C, ambient_C) -> torch.Tensor:
    """Physically meaningful features, scaled to O(1)."""
    i_d = i_d / 300.0
    i_q = i_q / 300.0
    i_sq = (i_d ** 2 + i_q ** 2)
    spd = speed_rpm / 6000.0
    return torch.stack([i_d, i_q, i_sq, spd, spd.abs(),
                        coolant_C / 100.0, ambient_C / 100.0], dim=-1)


class PositiveMLP(nn.Module):
    """Small MLP whose output is guaranteed positive, with a sensible scale.

    softplus, not exp: exp saturates and produces enormous gradients early in
    training, and a conductance that overflows takes the whole rollout with it.
    The learned bias is initialised so the output starts near `init`, which
    means training begins from the physical prior rather than from noise.
    """

    def __init__(self, n_out: int, init: Sequence[float], hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEAT, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_out),
        )
        # start the network near zero so the bias dominates initially
        nn.init.zeros_(self.net[-1].weight)
        inv = torch.tensor([math.log(math.expm1(max(v, 1e-9))) for v in init],
                           dtype=torch.float32)
        self.net[-1].bias.data = inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.net(x))


class ThermalNN(nn.Module):
    """State-space thermal model with learned, operating-point-dependent parameters."""

    def __init__(self, cap_init: Sequence[float],
                 cond_init: Sequence[float],
                 bnd_init: Sequence[float],
                 r_phase_init: float = 0.02,
                 hidden: int = 24):
        super().__init__()
        # Capacitances: scalars, strictly positive. Learned in log space.
        self.log_cap = nn.Parameter(torch.log(torch.tensor(cap_init, dtype=torch.float32)))

        # Conductances as functions of the operating point.
        self.g_link = PositiveMLP(len(LINKS), cond_init, hidden)
        self.g_bnd = PositiveMLP(len(BOUNDARIES), bnd_init, hidden)

        # Losses. Copper is explicit; the rest are learned functions of speed
        # and current, constrained non-negative (a loss cannot be a heat sink).
        self.log_r_phase = nn.Parameter(torch.log(torch.tensor(float(r_phase_init))))
        self.p_other = PositiveMLP(N, [1.0, 20.0, 1.0, 10.0], hidden)

    # -- physical read-outs, for comparison against the priors -------------

    def capacitances(self) -> torch.Tensor:
        return torch.exp(self.log_cap)

    def resistances_at(self, feats: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            g = self.g_link(feats)
            gb = self.g_bnd(feats)
        names = ["winding_to_yoke", "yoke_to_housing",
                 "magnet_to_winding_airgap", "magnet_to_housing_shaft"]
        out = {n: float(1.0 / g[..., k].mean()) for k, n in enumerate(names)}
        out["housing_to_coolant"] = float(1.0 / gb[..., 0].mean())
        out["housing_to_ambient"] = float(1.0 / gb[..., 1].mean())
        return out

    # -- dynamics ----------------------------------------------------------

    def derivative(self, T: torch.Tensor, feats: torch.Tensor,
                   i_sq_A2: torch.Tensor, coolant: torch.Tensor,
                   ambient: torch.Tensor) -> torch.Tensor:
        """dT/dt for a batch. T: (B, 4). Returns (B, 4)."""
        g = self.g_link(feats)              # (B, n_links)  >= 0
        gb = self.g_bnd(feats)              # (B, n_bnd)    >= 0
        C = self.capacitances()             # (4,)          >  0

        q = torch.zeros_like(T)

        # Internal conduction. Written as an antisymmetric exchange so that
        # whatever leaves one node arrives at the other EXACTLY - conservation
        # holds to floating point, by construction rather than by penalty.
        for k, (a, b) in enumerate(LINKS):
            flow = g[:, k] * (T[:, b] - T[:, a])
            q[:, a] = q[:, a] + flow
            q[:, b] = q[:, b] - flow

        # Boundaries
        bnd_T = torch.stack([coolant, ambient], dim=-1)     # (B, 2)
        for k, (node, bi) in enumerate(BOUNDARIES):
            q[:, node] = q[:, node] + gb[:, k] * (bnd_T[:, bi] - T[:, node])

        # Copper loss, kept explicit with its exact temperature feedback.
        r_phase = torch.exp(self.log_r_phase)
        r_T = r_phase * (1.0 + ALPHA_CU * (T[:, IDX["winding"]] - T_REF_CU))
        p_cu = 1.5 * r_T * i_sq_A2
        q[:, IDX["winding"]] = q[:, IDX["winding"]] + p_cu

        # Remaining losses: learned, non-negative.
        q = q + self.p_other(feats)

        return q / C

    def step(self, T: torch.Tensor, feats: torch.Tensor, i_sq: torch.Tensor,
             coolant: torch.Tensor, ambient: torch.Tensor,
             dt: float) -> torch.Tensor:
        """One RK4 step. Inputs are held constant across the step."""
        k1 = self.derivative(T, feats, i_sq, coolant, ambient)
        k2 = self.derivative(T + 0.5 * dt * k1, feats, i_sq, coolant, ambient)
        k3 = self.derivative(T + 0.5 * dt * k2, feats, i_sq, coolant, ambient)
        k4 = self.derivative(T + dt * k3, feats, i_sq, coolant, ambient)
        return T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, T0: torch.Tensor, seq: Dict[str, torch.Tensor],
                dt: float, horizon: int) -> torch.Tensor:
        """Free-running rollout. Returns (B, horizon+1, 4).

        The model is fed its OWN output at every step. It never sees a
        measurement after t=0, which is the only regime in which drift shows up.
        """
        B = T0.shape[0]
        out = torch.empty(B, horizon + 1, N, device=T0.device, dtype=T0.dtype)
        out[:, 0] = T0
        T = T0
        for k in range(horizon):
            f = seq["feats"][:, k]
            T = self.step(T, f, seq["i_sq"][:, k], seq["coolant"][:, k],
                          seq["ambient"][:, k], dt)
            out[:, k + 1] = T
        return out

    # -- conservation check (should hold identically) -----------------------

    def energy_residual(self, T: torch.Tensor, feats: torch.Tensor,
                        i_sq: torch.Tensor, coolant: torch.Tensor,
                        ambient: torch.Tensor) -> torch.Tensor:
        """generated - exported - stored. Zero by construction; verified anyway."""
        g = self.g_link(feats)
        gb = self.g_bnd(feats)
        C = self.capacitances()

        r_phase = torch.exp(self.log_r_phase)
        r_T = r_phase * (1.0 + ALPHA_CU * (T[:, IDX["winding"]] - T_REF_CU))
        generated = 1.5 * r_T * i_sq + self.p_other(feats).sum(dim=-1)

        bnd_T = torch.stack([coolant, ambient], dim=-1)
        exported = torch.zeros_like(generated)
        for k, (node, bi) in enumerate(BOUNDARIES):
            exported = exported + gb[:, k] * (T[:, node] - bnd_T[:, bi])

        dT = self.derivative(T, feats, i_sq, coolant, ambient)
        stored = (C * dT).sum(dim=-1)
        return generated - exported - stored
