"""
Validation gates 1 and 2.

Gate 1 — ANALYTICAL. Reduce the network to a single node with one path to a
boundary, apply constant power, and check the response against the closed-form
first-order solution. This is the sharpest test available on the integrator,
because the exact answer is known.

Gate 2 — PHYSICS. Assert the first law (energy in - out = stored), the second law
(no heat flows up a gradient), and boundedness under bounded input. These hold
for any correct network regardless of parameter values, so they catch sign
errors and assembly mistakes that a plausible-looking temperature curve hides.

Gate 3 (reality, against measured hardware) lives in a separate file, because it
is the only one that can falsify the physics rather than confirm the code agrees
with itself.

Run:  PYTHONPATH=src python tests/test_gates.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lptn import IDX, N, NODES, Inputs, ThermalNetwork  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}")
    if detail:
        print(f"       {detail}")
    return ok


# ---------------------------------------------------------------------------
# Gate 1 — analytical
# ---------------------------------------------------------------------------

ADIABATIC = 1.0e6          # K/W. Effectively isolating next to ~0.01 K/W, but
                           # finite, so the system matrix stays strictly stable.


def isolated_housing_network(power_w: float):
    """Housing node alone, one resistance to coolant, one constant heat input.

    Every other loss is switched off and every other link made adiabatic, so the
    node reduces to  C dT/dt = P - (T - T_coolant)/R  with a known closed form.
    The heat is injected through the bearing-friction term, which lands on the
    housing: tau * omega with both chosen to give exactly `power_w`.
    """
    speed_rpm = 1000.0
    omega = speed_rpm * 2 * math.pi / 60.0
    ov = {
        "losses.copper.phase_resistance_20C": 0.0,
        "losses.iron.hysteresis_coefficient": 0.0,
        "losses.iron.eddy_coefficient": 0.0,
        "losses.magnet_eddy.coefficient": 0.0,
        "losses.mechanical.windage_coefficient": 0.0,
        "losses.mechanical.bearing_friction_torque": power_w / omega,
        "resistances.winding_to_yoke": ADIABATIC,
        "resistances.yoke_to_housing": ADIABATIC,
        "resistances.magnet_to_winding_airgap": ADIABATIC,
        "resistances.magnet_to_housing_shaft": ADIABATIC,
        "resistances.housing_to_ambient": ADIABATIC,
    }
    net = ThermalNetwork(overrides=ov)
    u = Inputs(i_d=0.0, i_q=0.0, speed_rpm=speed_rpm,
               coolant_C=40.0, ambient_C=40.0, flow_kg_s=net.nominal_flow)
    return net, u


def gate1_analytical() -> bool:
    print("\n--- Gate 1: analytical ---")
    P = 250.0
    net, u = isolated_housing_network(P)
    r = net.resistances(u)
    C = net.C[IDX["housing"]]
    T0 = u.coolant_C

    # The housing's effective resistance is the parallel combination of the paths
    # that actually carry current at steady state: the two BOUNDARY paths only.
    #
    # The winding/yoke/magnet nodes are lossless here and their only connections
    # are the "adiabatic" links, so they form a floating subnetwork. At
    # equilibrium a floating lossless subnetwork must sit at the temperature of
    # whatever it hangs off - verified: all four nodes settle to an identical
    # 41.749999988 C - which means those links carry exactly zero current and
    # must NOT appear in the effective resistance. Including them was wrong by
    # 2.45e-8 K; excluding them is exact to 8.9e-15 K.
    R = 1.0 / (1.0 / r["housing_coolant"] + 1.0 / r["housing_ambient"])
    tau = R * C

    ok = True

    # (a) steady-state rise must equal P*R EXACTLY
    ss = net.steady_state(u)[IDX["housing"]]
    rise = ss - u.coolant_C
    err_ss = abs(rise - P * R)
    ok &= check("steady-state rise equals P*R",
                err_ss < 1e-9,
                f"P*R = {P*R:.9f} K, model = {rise:.9f} K, |err| = {err_ss:.3e} K")

    # (b) transient must match dT(t) = P*R(1 - e^{-t/RC}) at every sample
    t, Y = net.simulate([T0] * N, u, t_end=4 * tau, n_out=200, method="Radau")
    got = Y[:, IDX["housing"]] - T0
    want = P * R * (1.0 - np.exp(-t / tau))
    err = float(np.max(np.abs(got - want)))
    ok &= check("Radau transient matches closed form",
                err < 1e-4,
                f"max |err| = {err:.3e} K over 4 tau (tau = {tau:.1f} s)")

    # (c) the exact propagator must be exact at ANY step size, including one
    #     enormous step that would destroy an explicit integrator
    for dt in (1.0, 60.0, tau, 10 * tau):
        T = np.full(N, T0)
        T = net.step_exact(T, u, dt)
        want1 = T0 + P * R * (1.0 - math.exp(-dt / tau))
        e = abs(T[IDX["housing"]] - want1)
        # Not perfectly first-order during the transient: the floating nodes do
        # have capacitance and absorb a little heat on the way to equilibrium,
        # even though they carry none at steady state. That is physics, not
        # solver error, and it is ~1e-8 K here.
        ok &= check(f"step_exact matches closed form at dt = {dt:.1f} s",
                    e < 1e-6, f"|err| = {e:.3e} K")

    # (d) time constant recovered from the response equals R*C
    idx63 = int(np.argmin(np.abs(got - 0.632 * P * R)))
    tau_meas = t[idx63]
    ok &= check("recovered time constant equals R*C",
                abs(tau_meas - tau) / tau < 0.02,
                f"R*C = {tau:.1f} s, 63.2% crossing at {tau_meas:.1f} s")
    return ok


# ---------------------------------------------------------------------------
# Gate 2 — physics
# ---------------------------------------------------------------------------

def gate2_physics() -> bool:
    print("\n--- Gate 2: physics ---")
    net = ThermalNetwork()
    ok = True
    rng = np.random.default_rng(7)

    points = [
        Inputs(i_d=-70, i_q=60, speed_rpm=2500, coolant_C=35, ambient_C=25, flow_kg_s=0.12),
        Inputs(i_d=0, i_q=0, speed_rpm=0, coolant_C=20, ambient_C=20, flow_kg_s=0.12),
        Inputs(i_d=-250, i_q=280, speed_rpm=6000, coolant_C=60, ambient_C=30, flow_kg_s=0.02),
        Inputs(i_d=-30, i_q=-120, speed_rpm=-3000, coolant_C=15, ambient_C=10, flow_kg_s=0.30),
    ]

    # (a) conductance matrix must be symmetric: R_ij = R_ji
    worst_sym = 0.0
    for u in points:
        G = net.conductance_matrix(u)
        worst_sym = max(worst_sym, float(np.max(np.abs(G - G.T))))
    ok &= check("conductance matrix is symmetric",
                worst_sym < 1e-12, f"max |G - G^T| = {worst_sym:.3e}")

    # (b) first law over a rollout: energy in - out = stored
    worst_bal = 0.0
    for u in points:
        T = np.array([u.coolant_C] * N, dtype=float)
        dt, steps = 1.0, 3000
        e_in = e_out = 0.0
        T_start = T.copy()
        for _ in range(steps):
            b0 = net.power_balance(T, u)
            T = net.step_exact(T, u, dt)
            b1 = net.power_balance(T, u)
            e_in += 0.5 * (b0["generated"] + b1["generated"]) * dt
            e_out += 0.5 * (b0["exported"] + b1["exported"]) * dt
        stored = float(net.C @ (T - T_start))
        rel = abs(e_in - e_out - stored) / max(abs(e_in), 1.0)
        worst_bal = max(worst_bal, rel)
    ok &= check("first law: energy in - out = stored, over rollouts",
                worst_bal < 2e-3,
                f"worst relative closure error = {worst_bal:.3e} "
                "(trapezoid on a curved integrand, not a model error)")

    # (c) instantaneous balance must close to machine precision
    worst_inst = 0.0
    for u in points:
        for _ in range(20):
            T = 20.0 + 120.0 * rng.random(N)
            worst_inst = max(worst_inst, abs(net.power_balance(T, u)["residual"]))
    ok &= check("instantaneous power balance closes",
                worst_inst < 1e-8, f"max |residual| = {worst_inst:.3e} W")

    # (d) second law: heat never flows up a temperature gradient
    bad = 0
    worst_dir = 0.0
    for u in points:
        for _ in range(200):
            T = 10.0 + 180.0 * rng.random(N)
            for a, b, q in net.internal_fluxes(T, u):
                dT = T[IDX[a]] - T[IDX[b]]
                if abs(dT) > 1e-9 and np.sign(q) != np.sign(dT):
                    bad += 1
                    worst_dir = max(worst_dir, abs(q))
    ok &= check("second law: no heat flows up a gradient",
                bad == 0, f"{bad} violations (worst {worst_dir:.3e} W)")

    # (e) a negative resistance must be REJECTED, not silently accepted.
    #     An optimiser will propose one if it lowers the residual.
    rejected = False
    try:
        ThermalNetwork(overrides={"resistances.winding_to_yoke": -0.05}) \
            .resistances(points[0])
    except ValueError:
        rejected = True
    ok &= check("negative resistance is rejected", rejected,
                "R > 0 is a physical law, not a regularisation choice")

    # (f) boundedness: bounded input -> bounded state, and equilibrium reached
    u = points[0]
    T = np.array([25.0] * N)
    for _ in range(20000):
        T = net.step_exact(T, u, 1.0)
    ss = net.steady_state(u)
    ok &= check("bounded input gives bounded state",
                bool(np.all(np.isfinite(T))) and float(np.max(np.abs(T - ss))) < 1e-6,
                f"max |T(20000 s) - steady state| = {float(np.max(np.abs(T - ss))):.3e} K")

    # (g) monotone response: raising current can only raise the winding
    base = Inputs(i_d=-50, i_q=50, speed_rpm=2000, coolant_C=35, ambient_C=25, flow_kg_s=0.12)
    temps = []
    for scale in (0.5, 1.0, 1.5, 2.0):
        uu = Inputs(i_d=base.i_d * scale, i_q=base.i_q * scale,
                    speed_rpm=base.speed_rpm, coolant_C=base.coolant_C,
                    ambient_C=base.ambient_C, flow_kg_s=base.flow_kg_s)
        temps.append(net.steady_state(uu)[IDX["winding"]])
    ok &= check("winding temperature is monotone in current",
                all(b > a for a, b in zip(temps, temps[1:])),
                " -> ".join(f"{v:.1f}" for v in temps) + " C")

    # (h) reducing coolant flow must make it hotter (R ~ mdot^-0.8)
    hot = []
    for flow in (0.30, 0.12, 0.05, 0.02):
        uu = Inputs(i_d=-70, i_q=60, speed_rpm=2500, coolant_C=35,
                    ambient_C=25, flow_kg_s=flow)
        hot.append(net.steady_state(uu)[IDX["winding"]])
    ok &= check("lower coolant flow gives higher temperature",
                all(b > a for a, b in zip(hot, hot[1:])),
                " -> ".join(f"{v:.1f}" for v in hot) + " C as flow falls")

    # (i) the air gap must get less resistive with speed (Taylor vortices)
    gaps = [net.resistances(Inputs(speed_rpm=s))["magnet_winding_gap"]
            for s in (0, 500, 2000, 6000)]
    ok &= check("air-gap resistance falls with speed",
                all(b <= a for a, b in zip(gaps, gaps[1:])) and gaps[-1] < gaps[0],
                " -> ".join(f"{v:.3f}" for v in gaps) + " K/W as speed rises")

    # (j) runaway must be DETECTED, not clipped. Force the copper feedback to
    #     outrun the cooling path and confirm the model says so.
    hot_net = ThermalNetwork(overrides={
        "losses.copper.phase_resistance_20C": 4.0,     # absurd on purpose
        "resistances.winding_to_yoke": 5.0,
    })
    u_run = Inputs(i_d=-200, i_q=200, speed_rpm=1000, coolant_C=40,
                   ambient_C=25, flow_kg_s=0.12)
    detected = hot_net.is_runaway(u_run)
    ss_nan = bool(np.all(np.isnan(hot_net.steady_state(u_run))))
    ok &= check("thermal runaway is detected, not clipped",
                detected and ss_nan,
                "eigenvalue of A crossed zero; steady state reported as NaN")
    return ok


if __name__ == "__main__":
    ok1 = gate1_analytical()
    ok2 = gate2_physics()
    n_pass = sum(1 for _, o, _ in _results if o)
    print(f"\n{'=' * 64}")
    print(f"gate 1 (analytical): {'PASS' if ok1 else 'FAIL'}")
    print(f"gate 2 (physics):    {'PASS' if ok2 else 'FAIL'}")
    print(f"{n_pass}/{len(_results)} checks passed")
    print("=" * 64)
    sys.exit(0 if (ok1 and ok2) else 1)
