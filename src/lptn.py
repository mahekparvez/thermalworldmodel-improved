"""
Four-node lumped-parameter thermal network for a PMSM.

Nodes:      winding, yoke, magnet(+rotor), housing
Boundaries: coolant, ambient   (prescribed temperatures, not integrated)

THE KEY STRUCTURAL FACT
-----------------------
Copper resistance is LINEAR in temperature, R(T) = R20[1 + a(T - 20)], so the
copper loss injected at the winding node is also linear in the winding
temperature. That means the whole network is linear time-invariant for fixed
inputs:

    C dT/dt = A T + b

and two useful things follow:

  * the exact solution over an interval of constant input is a matrix
    exponential, so no integration error at all where inputs are piecewise
    constant (which is how 2 Hz sampled data is naturally treated);
  * the copper feedback appears as a POSITIVE contribution to the winding
    diagonal of A. If it exceeds the cooling conductance, A has a non-negative
    eigenvalue and the system has no stable equilibrium. That is thermal
    runaway, and it is detected by inspecting eigenvalues rather than by waiting
    for a simulation to blow up.

STIFFNESS
---------
Node time constants span roughly 10-100x (winding tens of seconds, housing tens
of minutes). Explicit fixed-step integration either goes unstable or, worse,
stays stable while accumulating silent error. `solve_ivp` with Radau/BDF is used
where a general solver is wanted; `step_exact` is used where inputs are
piecewise constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from params import ParamSet, load as load_params

NODES: Tuple[str, ...] = ("winding", "yoke", "magnet", "housing")
IDX = {n: i for i, n in enumerate(NODES)}
N = len(NODES)


# ---------------------------------------------------------------------------
# Operating inputs
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    """One operating point. Currents are PEAK dq amplitudes, as in the dataset."""
    i_d: float = 0.0            # A
    i_q: float = 0.0            # A
    speed_rpm: float = 0.0
    coolant_C: float = 35.0
    ambient_C: float = 25.0
    flow_kg_s: float = 0.12

    @property
    def omega(self) -> float:
        """Mechanical angular velocity, rad/s."""
        return self.speed_rpm * 2.0 * math.pi / 60.0

    def elec_freq(self, pole_pairs: float) -> float:
        """Electrical frequency in Hz. f = p * n / 60."""
        return abs(pole_pairs * self.speed_rpm / 60.0)


# ---------------------------------------------------------------------------
# Speed- and flow-dependent resistances
# ---------------------------------------------------------------------------

def airgap_conductivity_ratio(taylor: float) -> float:
    """k_eff / k_air across a rotating annulus, from the Taylor number.

    Below the critical Taylor number the gap is a pure conduction layer. Above
    it, counter-rotating Taylor vortices form and effective conductivity rises.
    Becker & Kaye piecewise correlation, as used throughout the LPTN literature.
    """
    if taylor < 1700.0:
        return 1.0
    if taylor < 1.0e4:
        return 0.128 * taylor ** 0.367
    return 0.409 * taylor ** 0.241


def taylor_number(omega: float, r_mean: float, gap: float, nu: float) -> float:
    """Ta = omega^2 * r_m * delta^3 / nu^2."""
    return (omega ** 2) * r_mean * (gap ** 3) / (nu ** 2)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class ThermalNetwork:
    def __init__(self, ps: Optional[ParamSet] = None,
                 overrides: Optional[Dict[str, float]] = None):
        self.ps = ps if ps is not None else load_params()
        self._ov = dict(overrides or {})

        g = self.ps.raw["geometry"]
        self.r_mean = 0.5 * (g["stator_inner_radius_m"]["value"]
                             + g["rotor_outer_radius_m"]["value"])
        self.gap = g["air_gap_m"]["value"]
        self.nu_air = self.ps.value("materials.air.kinematic_viscosity")
        self.pole_pairs = float(self.ps.raw["meta"]["pole_pairs"])
        self.steel_mass = g["steel_mass_kg"]["value"]
        self.magnet_mass = g["magnet_mass_kg"]["value"]

        self.nominal_flow = self.ps.value("operating.nominal_flow_kg_s")
        self.alpha = self.ps.value("constants.alpha_copper")
        self.T_ref = self.ps.value("constants.T_ref_copper")

        self.C = np.array([self.p(f"capacitances.{n}") for n in NODES], dtype=float)
        if np.any(self.C <= 0):
            raise ValueError("thermal capacitances must be positive")

    # -- parameter access with override support (used by the identifier) ----

    def p(self, key: str) -> float:
        if key in self._ov:
            return self._ov[key]
        return self.ps.value(key)

    def with_overrides(self, ov: Dict[str, float]) -> "ThermalNetwork":
        merged = dict(self._ov)
        merged.update(ov)
        return ThermalNetwork(self.ps, merged)

    # -- resistances -------------------------------------------------------

    def resistances(self, u: Inputs) -> Dict[str, float]:
        """All six resistances at this operating point. Two are functions."""
        r = {
            "winding_yoke": self.p("resistances.winding_to_yoke"),
            "yoke_housing": self.p("resistances.yoke_to_housing"),
            "magnet_shaft_housing": self.p("resistances.magnet_to_housing_shaft"),
            "housing_ambient": self.p("resistances.housing_to_ambient"),
        }

        # Air gap: falls as the rotor spins and vortices set in.
        ta = taylor_number(abs(u.omega), self.r_mean, self.gap, self.nu_air)
        r["magnet_winding_gap"] = (self.p("resistances.magnet_to_winding_airgap")
                                   / airgap_conductivity_ratio(ta))

        # Coolant: h ~ Re^0.8 ~ mdot^0.8, so R ~ mdot^-0.8.
        flow = max(u.flow_kg_s, 1e-6)
        r["housing_coolant"] = (self.p("resistances.housing_to_coolant")
                                * (self.nominal_flow / flow) ** 0.8)

        for k, v in r.items():
            if not (v > 0):
                # A non-positive resistance transports heat up a gradient.
                raise ValueError(f"resistance '{k}' is {v}; must be > 0 (2nd law)")
        return r

    # -- losses ------------------------------------------------------------

    def copper_loss_terms(self, u: Inputs) -> Tuple[float, float]:
        """Copper loss split into (constant part, coefficient on T_winding).

        P_cu(T) = 1.5 * R20 * [1 + a(T - 20)] * (i_d^2 + i_q^2)
                = p_const + k_T * T
        Returned separately so the caller can place k_T on the A-matrix diagonal
        instead of iterating.
        """
        r20 = self.p("losses.copper.phase_resistance_20C")
        fac = self.p("losses.copper.dq_to_three_phase_factor")
        i_sq = u.i_d ** 2 + u.i_q ** 2

        # AC excess, driven by electrical frequency. Defaults to zero.
        k_ac = self.p("losses.copper.ac_excess_coefficient")
        f_e = u.elec_freq(self.pole_pairs)
        f_ref = 100.0
        r_eff = r20 * (1.0 + k_ac * (f_e / f_ref) ** 2)

        base = fac * r_eff * i_sq
        p_const = base * (1.0 - self.alpha * self.T_ref)
        k_T = base * self.alpha
        return p_const, k_T

    def static_losses(self, u: Inputs) -> np.ndarray:
        """Temperature-independent losses, W, injected per node."""
        P = np.zeros(N)
        f_e = u.elec_freq(self.pole_pairs)
        omega = abs(u.omega)
        B = self.p("losses.iron.flux_density_T")
        n_st = self.p("losses.iron.steinmetz_exponent")

        # Iron loss: Steinmetz. Scales with SPEED, not torque.
        k_h = self.p("losses.iron.hysteresis_coefficient")
        k_e = self.p("losses.iron.eddy_coefficient")
        p_iron = self.steel_mass * (k_h * f_e * B ** n_st + k_e * f_e ** 2 * B ** 2)
        P[IDX["yoke"]] += p_iron

        # Magnet eddy loss, driven by harmonics rather than the fundamental.
        k_pm = self.p("losses.magnet_eddy.coefficient")
        P[IDX["magnet"]] += self.magnet_mass * k_pm * f_e ** 2 * B ** 2

        # Mechanical. Bearing friction ~ omega, windage ~ omega^3.
        tau_b = self.p("losses.mechanical.bearing_friction_torque")
        k_w = self.p("losses.mechanical.windage_coefficient")
        P[IDX["housing"]] += tau_b * omega
        P[IDX["magnet"]] += k_w * omega ** 3

        return P

    # -- assembly ----------------------------------------------------------

    def system(self, u: Inputs) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A, b) for  dT/dt = A T + b."""
        r = self.resistances(u)
        G = np.zeros((N, N))

        def link(a: str, b: str, res: float) -> None:
            i, j = IDX[a], IDX[b]
            g = 1.0 / res
            G[i, j] += g
            G[j, i] += g
            G[i, i] -= g
            G[j, j] -= g

        link("winding", "yoke", r["winding_yoke"])
        link("yoke", "housing", r["yoke_housing"])
        link("magnet", "winding", r["magnet_winding_gap"])
        link("magnet", "housing", r["magnet_shaft_housing"])

        P = self.static_losses(u)
        p_cu, k_cu = self.copper_loss_terms(u)
        P[IDX["winding"]] += p_cu

        # Boundaries: conductance to a prescribed temperature.
        g_cool = 1.0 / r["housing_coolant"]
        g_amb = 1.0 / r["housing_ambient"]
        G[IDX["housing"], IDX["housing"]] -= (g_cool + g_amb)
        P[IDX["housing"]] += g_cool * u.coolant_C + g_amb * u.ambient_C

        # Copper feedback: a POSITIVE term on the winding diagonal. If it
        # outweighs the conduction out of the winding, the eigenvalue turns
        # non-negative and there is no equilibrium.
        G[IDX["winding"], IDX["winding"]] += k_cu

        A = G / self.C[:, None]
        b = P / self.C
        return A, b

    def system_batch(self, i_d, i_q, speed_rpm, coolant_C, ambient_C,
                     flow_kg_s=None):
        """Vectorised (A, b) for a whole trajectory at once.

        Same maths as `system()`, but over arrays. Built because the
        identification inner loop calls this tens of thousands of times per
        objective evaluation, and a Python loop over timesteps made the fit
        ~100x slower than it needed to be.
        """
        i_d = np.asarray(i_d, float); i_q = np.asarray(i_q, float)
        spd = np.asarray(speed_rpm, float)
        cool = np.asarray(coolant_C, float); amb = np.asarray(ambient_C, float)
        n = i_d.shape[0]
        flow = np.full(n, self.nominal_flow) if flow_kg_s is None \
            else np.asarray(flow_kg_s, float)

        omega = np.abs(spd) * 2.0 * math.pi / 60.0
        f_e = np.abs(self.pole_pairs * spd / 60.0)

        # --- resistances (two are functions of the operating point) ---
        r_wy = self.p("resistances.winding_to_yoke")
        r_yh = self.p("resistances.yoke_to_housing")
        r_ms = self.p("resistances.magnet_to_housing_shaft")
        r_ha = self.p("resistances.housing_to_ambient")
        for nm, v in (("winding_to_yoke", r_wy), ("yoke_to_housing", r_yh),
                      ("magnet_to_housing_shaft", r_ms),
                      ("housing_to_ambient", r_ha)):
            if not (v > 0):
                raise ValueError(f"resistance '{nm}' is {v}; must be > 0 (2nd law)")

        ta = (omega ** 2) * self.r_mean * (self.gap ** 3) / (self.nu_air ** 2)
        ratio = np.where(ta < 1700.0, 1.0,
                 np.where(ta < 1.0e4, 0.128 * np.maximum(ta, 1e-30) ** 0.367,
                                      0.409 * np.maximum(ta, 1e-30) ** 0.241))
        r_gap0 = self.p("resistances.magnet_to_winding_airgap")
        if not (r_gap0 > 0):
            raise ValueError("air-gap resistance must be > 0 (2nd law)")
        r_gap = r_gap0 / ratio

        r_hc0 = self.p("resistances.housing_to_coolant")
        if not (r_hc0 > 0):
            raise ValueError("coolant resistance must be > 0 (2nd law)")
        r_hc = r_hc0 * (self.nominal_flow / np.maximum(flow, 1e-6)) ** 0.8

        # --- losses ---
        B = self.p("losses.iron.flux_density_T")
        n_st = self.p("losses.iron.steinmetz_exponent")
        k_h = self.p("losses.iron.hysteresis_coefficient")
        k_e = self.p("losses.iron.eddy_coefficient")
        p_iron = self.steel_mass * (k_h * f_e * B ** n_st + k_e * f_e ** 2 * B ** 2)
        k_pm = self.p("losses.magnet_eddy.coefficient")
        p_pm = self.magnet_mass * k_pm * f_e ** 2 * B ** 2
        tau_b = self.p("losses.mechanical.bearing_friction_torque")
        k_w = self.p("losses.mechanical.windage_coefficient")
        p_bear = tau_b * omega
        p_wind = k_w * omega ** 3

        r20 = self.p("losses.copper.phase_resistance_20C")
        fac = self.p("losses.copper.dq_to_three_phase_factor")
        k_ac = self.p("losses.copper.ac_excess_coefficient")
        r_eff_cu = r20 * (1.0 + k_ac * (f_e / 100.0) ** 2)
        i_sq = i_d ** 2 + i_q ** 2
        base = fac * r_eff_cu * i_sq
        p_cu_const = base * (1.0 - self.alpha * self.T_ref)
        k_cu = base * self.alpha

        # --- assemble ---
        A = np.zeros((n, N, N))
        P = np.zeros((n, N))
        iw, iy, im, ih = IDX["winding"], IDX["yoke"], IDX["magnet"], IDX["housing"]

        def link(a, b, res):
            g = 1.0 / res
            A[:, a, b] += g; A[:, b, a] += g
            A[:, a, a] -= g; A[:, b, b] -= g

        link(iw, iy, r_wy)
        link(iy, ih, r_yh)
        link(im, iw, r_gap)
        link(im, ih, r_ms)

        P[:, iy] += p_iron
        P[:, im] += p_pm + p_wind
        P[:, ih] += p_bear
        P[:, iw] += p_cu_const

        g_cool = 1.0 / r_hc
        g_amb = 1.0 / r_ha
        A[:, ih, ih] -= (g_cool + g_amb)
        P[:, ih] += g_cool * cool + g_amb * amb
        A[:, iw, iw] += k_cu

        A /= self.C[None, :, None]
        b = P / self.C[None, :]
        return A, b

    def conductance_matrix(self, u: Inputs) -> np.ndarray:
        """Passive conductance only (no copper feedback). Must be symmetric."""
        r = self.resistances(u)
        G = np.zeros((N, N))
        for a, bnode, key in (("winding", "yoke", "winding_yoke"),
                              ("yoke", "housing", "yoke_housing"),
                              ("magnet", "winding", "magnet_winding_gap"),
                              ("magnet", "housing", "magnet_shaft_housing")):
            i, j = IDX[a], IDX[bnode]
            g = 1.0 / r[key]
            G[i, j] += g
            G[j, i] += g
            G[i, i] -= g
            G[j, j] -= g
        return G

    # -- solving -----------------------------------------------------------

    def derivative(self, T: np.ndarray, u: Inputs) -> np.ndarray:
        A, b = self.system(u)
        return A @ T + b

    def is_runaway(self, u: Inputs) -> bool:
        """No stable equilibrium: some eigenvalue of A has non-negative real part."""
        A, _ = self.system(u)
        return bool(np.max(np.real(np.linalg.eigvals(A))) >= -1e-12)

    def steady_state(self, u: Inputs) -> np.ndarray:
        """Equilibrium temperatures. NaN where none exists."""
        A, b = self.system(u)
        if self.is_runaway(u):
            return np.full(N, np.nan)
        return np.linalg.solve(A, -b)

    def step_exact(self, T: np.ndarray, u: Inputs, dt: float) -> np.ndarray:
        """Exact propagation over dt, for CONSTANT inputs.

        T(t+dt) = e^{A dt} T + A^-1 (e^{A dt} - I) b

        Built by exponentiating the augmented matrix, which avoids inverting A
        and stays correct when A is singular or near-singular.
        """
        A, b = self.system(u)
        M = np.zeros((N + 1, N + 1))
        M[:N, :N] = A
        M[:N, N] = b
        E = expm(M * dt)
        return E[:N, :N] @ T + E[:N, N]

    def simulate(self, T0: Sequence[float], u: Inputs, t_end: float,
                 n_out: int = 200, method: str = "Radau") -> Tuple[np.ndarray, np.ndarray]:
        """General stiff solve at a constant operating point."""
        t_eval = np.linspace(0.0, t_end, n_out)
        sol = solve_ivp(lambda t, T: self.derivative(T, u),
                        (0.0, t_end), np.asarray(T0, dtype=float),
                        t_eval=t_eval, method=method, rtol=1e-8, atol=1e-10)
        if not sol.success:
            raise RuntimeError(f"solver failed: {sol.message}")
        return sol.t, sol.y.T

    def rollout(self, T0: Sequence[float], inputs: Sequence[Inputs],
                dt: float) -> np.ndarray:
        """Propagate through a sequence of piecewise-constant operating points.

        Uses the exact propagator, so the result carries no integration error -
        only the (real) modelling error of treating the input as constant across
        each sample interval.
        """
        T = np.asarray(T0, dtype=float).copy()
        out = np.empty((len(inputs) + 1, N))
        out[0] = T
        for k, u in enumerate(inputs):
            T = self.step_exact(T, u, dt)
            out[k + 1] = T
        return out

    # -- energy accounting -------------------------------------------------

    def power_balance(self, T: np.ndarray, u: Inputs) -> Dict[str, float]:
        """Instantaneous generated / exported / stored power, W."""
        r = self.resistances(u)
        P = self.static_losses(u)
        p_cu, k_cu = self.copper_loss_terms(u)
        gen = float(P.sum() + p_cu + k_cu * T[IDX["winding"]])

        out = ((T[IDX["housing"]] - u.coolant_C) / r["housing_coolant"]
               + (T[IDX["housing"]] - u.ambient_C) / r["housing_ambient"])
        stored = float(self.C @ self.derivative(T, u))
        return {"generated": gen, "exported": float(out), "stored": stored,
                "residual": gen - float(out) - stored}

    def internal_fluxes(self, T: np.ndarray, u: Inputs) -> List[Tuple[str, str, float]]:
        """Heat flow on each internal link, positive from first to second."""
        r = self.resistances(u)
        links = (("winding", "yoke", "winding_yoke"),
                 ("yoke", "housing", "yoke_housing"),
                 ("magnet", "winding", "magnet_winding_gap"),
                 ("magnet", "housing", "magnet_shaft_housing"))
        return [(a, b, (T[IDX[a]] - T[IDX[b]]) / r[key]) for a, b, key in links]


if __name__ == "__main__":
    net = ThermalNetwork()
    u = Inputs(i_d=-70.0, i_q=60.0, speed_rpm=2500, coolant_C=35.0,
               ambient_C=25.0, flow_kg_s=0.12)
    print("operating point: i_d=-70 i_q=60 A, 2500 rpm, coolant 35 C\n")

    P = net.static_losses(u)
    p_cu, k_cu = net.copper_loss_terms(u)
    print(f"copper loss at 25 C   {p_cu + k_cu*25:8.1f} W")
    print(f"copper loss at 120 C  {p_cu + k_cu*120:8.1f} W   "
          f"(+{100*(k_cu*95)/(p_cu+k_cu*25):.1f}% from temperature alone)")
    for n, v in zip(NODES, P):
        if v > 0:
            print(f"static loss @ {n:8s} {v:8.1f} W")

    ss = net.steady_state(u)
    print("\nsteady state:")
    for n, v in zip(NODES, ss):
        print(f"  {n:8s} {v:7.2f} C")
    print(f"  runaway: {net.is_runaway(u)}")

    t, Y = net.simulate([35.0] * N, u, t_end=7200.0)
    print(f"\nafter 2 h (Radau): " +
          "  ".join(f"{n}={Y[-1, i]:.2f}" for i, n in enumerate(NODES)))
    print(f"max |Radau - analytic steady state| = {np.max(np.abs(Y[-1] - ss)):.4f} K")

    bal = net.power_balance(Y[-1], u)
    print(f"\npower balance at that point: generated {bal['generated']:.2f} W, "
          f"exported {bal['exported']:.2f} W, stored {bal['stored']:.3f} W, "
          f"residual {bal['residual']:.2e} W")
