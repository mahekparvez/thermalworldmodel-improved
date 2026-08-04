# Why a Motor Gets Hot

First-principles derivation of every loss mechanism, every heat-transfer path,
and the governing balance that the simulator must obey. Written before any code,
because the structure of the model is a physics decision, not a software one.

**Nothing in this document is fitted.** Numerical values quoted here are material
properties and standard correlations, cited to source. Values that could not be
sourced are marked `UNSOURCED` and must not be silently replaced by a guess.

---

## Contents

1. [The two questions](#1-the-two-questions)
2. [Loss mechanisms — where heat comes from](#2-loss-mechanisms)
   - 2.1 [Joule / copper loss](#21-joule--copper-loss)
   - 2.2 [Core / iron loss](#22-core--iron-loss)
   - 2.3 [Permanent-magnet eddy-current loss](#23-permanent-magnet-eddy-current-loss)
   - 2.4 [Mechanical loss](#24-mechanical-loss)
   - 2.5 [Summary: what drives what](#25-summary-what-drives-what)
3. [Heat transfer — where heat goes](#3-heat-transfer)
   - 3.1 [Conduction](#31-conduction)
   - 3.2 [Convection](#32-convection)
   - 3.3 [Radiation](#33-radiation)
   - 3.4 [The air gap](#34-the-air-gap)
4. [The governing balance](#4-the-governing-balance)
5. [Node network](#5-node-network)
6. [Stiffness, and why the solver matters](#6-stiffness)
7. [What a lumped model cannot capture](#7-what-a-lumped-model-cannot-capture)
8. [Constants and properties](#8-constants-and-properties)
9. [References](#9-references)

---

## 1. The two questions

Every thermal model answers two separate questions, and conflating them is the
most common source of error:

1. **How much heat is generated, and where?** — §2. Depends on current, speed,
   flux and temperature. Governed by electromagnetics and mechanics.
2. **How readily does that heat escape?** — §3. Depends on geometry, materials,
   contact quality and coolant flow. Governed by heat transfer.

Generation is comparatively well-constrained: currents and speeds are measured,
and the loss laws are standard. Escape is poorly constrained, because it depends
on interface resistances that are geometry- and assembly-dependent and are
rarely published. This asymmetry is why §6 of the build plan fits the *thermal
resistances* against measured data rather than the loss coefficients.

---

## 2. Loss mechanisms

### 2.1 Joule / copper loss

Resistive dissipation in the stator windings. For a three-phase machine with
per-phase RMS current `I_ph` and per-phase resistance `R_ph`:

```
P_cu = 3 · I_ph² · R_ph
```

In the `dq` frame used by the Paderborn dataset, with `i_d` and `i_q` as peak
amplitudes:

```
P_cu = (3/2) · R_ph · (i_d² + i_q²)
```

The factor 3/2 converts peak `dq` amplitudes to the equivalent three-phase RMS
sum. Getting this factor wrong is a silent 33% error.

#### The temperature coupling — mandatory

Copper resistivity rises with temperature. Over the range a motor operates in,
the linear form is accurate:

```
R(T) = R₂₀ · [1 + α · (T − 20 °C)]
α = 3.93 × 10⁻³ K⁻¹   for annealed copper at 20 °C
```

This is a **positive feedback loop**:

```
hotter winding → higher resistance → more I²R → hotter winding
```

It has three consequences the model must represent:

**(a) Loss is not constant during a transient.** A winding that settles 100 K
above ambient dissipates roughly `1 + 0.00393 × 100 = 1.39`, i.e. ~40% more than
the same winding cold. Reporting a single cold-start loss figure understates the
steady-state cooling duty by that margin.

**(b) The node equation stays first-order.** Because `R(T)` is *linear* in `T`,
substituting it into the balance keeps the equation linear:

```
C dT/dt = I²R₂₀[1 + α(T − 20)] − (T − T_amb)/R_th
        = p₀ + k·(T − T_amb)
```

with

```
p₀ = I²R₂₀[1 + α(T_amb − 20)]
k  = I²R₂₀α − 1/R_th
```

so a single node has the closed-form solution

```
T(t) = T_amb + (p₀/k)·(e^{k·t/C} − 1)
```

No numerical integration is needed for the single-node case, which is exactly
what makes the analytical gate (§Part 5.1 of the build plan) a sharp test.

**(c) Runaway is possible.** If `k ≥ 0` — that is, if `I²R₂₀α ≥ 1/R_th` — the
feedback outruns the cooling path and **no stable equilibrium exists**.
Resistance rises, loss rises, temperature rises, without bound. A real drive hits
current derating long before this; a model without derating must *report*
runaway rather than clip it, because clipping hides a design failure.

The stability criterion is worth stating explicitly, since it is a design
constraint on the cooler:

```
stable  ⟺  R_th < 1 / (I² R₂₀ α)
```

#### AC effects

At electrical frequency the current is not uniformly distributed in the
conductor. Skin effect (self-field) and proximity effect (neighbouring
conductors) raise the effective resistance:

```
R_ac(f) = R_dc · [1 + k_ac · (f/f_ref)²]
```

Second-order for the fundamental in a typical PMSM at moderate speed, but it
grows quadratically with frequency and is not negligible at high speed or in
form-wound machines with large conductors. Include with a small coefficient;
flag it as `UNSOURCED` unless a conductor geometry is available to compute it.

---

### 2.2 Core / iron loss

Dissipation in the laminated stator steel, arising from two distinct physical
mechanisms that happen to be summed by convention.

**Hysteresis loss** — energy consumed traversing the B–H loop once per electrical
cycle. The area enclosed by the loop is energy per unit volume per cycle, so
power is proportional to frequency:

```
P_h = k_h · f · B_max^n          n ≈ 1.6 – 2.2 (Steinmetz exponent)
```

**Eddy-current loss** — circulating currents induced in the steel by the changing
flux. The induced EMF is proportional to `dB/dt ∝ f·B`, and the resulting loss is
`EMF²/R`, hence:

```
P_e = k_e · f² · B_max²
```

The quadratic frequency dependence is why laminations exist: subdividing the core
into thin sheets of thickness `d` insulated from one another reduces eddy loss as
`d²`, since `k_e ∝ d²·σ/ρ` for lamination thickness `d`, conductivity `σ` and
density `ρ`.

Combined, this is the **Steinmetz equation**:

```
P_fe = k_h · f · B_max^n  +  k_e · f² · B_max²
```

Often written with a third "excess" or "anomalous" term `k_a · f^1.5 · B^1.5`
that accounts for domain-wall dynamics not captured by the other two [1].

#### Why this scales with speed, not torque

Electrical frequency is set by mechanical speed and pole-pair count:

```
f = p · n / 60      [Hz]     n in rpm, p = pole pairs
  = p · ω / (2π)             ω in rad/s
```

In a PMSM operating below base speed the flux `B_max` is approximately fixed by
the permanent magnets. Torque is set by current, which does **not** appear in the
Steinmetz equation at all. Therefore:

> **Copper loss scales with torque². Iron loss scales with speed.**

This is the single most important structural fact for a thermal model, and the
reason a purely torque-driven model is wrong. Two operating points with identical
torque and different speed have different heat. A high-speed, low-torque hold —
which looks benign by current — can be iron-loss dominated.

#### Reference-frame note

`k_h` and `k_e` are properties of the specific steel grade and lamination
thickness (M-19, M-235-35A, etc.), usually published as a loss curve in W/kg at
50 Hz / 1.5 T. Converting a published W/kg figure to the coefficients requires
knowing core mass and operating flux. If core mass is unavailable, express iron
loss as an **equivalent drag torque at the shaft** — `P = τ_drag·ω` — which
folds the unknown geometry into one coefficient rather than carrying `k_h`,
`k_e`, `B_max`, `n` and core mass as five separate unknowns that only ever
appear as a product. Fewer free parameters is the same thing as more certainty.

---

### 2.3 Permanent-magnet eddy-current loss

The rotor magnets sit in a field that, in the rotor frame, is *not* constant.
Slotting harmonics, MMF space harmonics from the winding distribution, and PWM
carrier harmonics all produce field ripple that the magnets see as an alternating
component. NdFeB is electrically conductive (ρ ≈ 1.4 × 10⁻⁶ Ω·m), so that ripple
induces eddy currents inside the magnet blocks:

```
P_pm ≈ k_pm · f_harm² · B_ripple² · V_magnet
```

Small in absolute terms — typically a few percent of total loss — but it matters
disproportionately for three reasons:

1. **The rotor has almost no cooling path.** It is surrounded by the air gap
   (§3.4), which is a poor conductor. Heat generated in the rotor mostly has to
   leave through the shaft and bearings, or radiate/convect across the gap.

2. **NdFeB demagnetises irreversibly above its temperature limit.** Roughly
   80–150 °C depending on grade (higher for Dy-doped grades). Exceeding it does
   not merely reduce performance until it cools — it permanently reduces remanent
   flux density `B_r`, and the machine never recovers its torque constant. This
   is a destructive failure, not a derating.

3. **It is the hardest node to instrument.** Rotor temperature cannot be measured
   with a fixed thermocouple; the Paderborn dataset obtains it via a
   thermocouple with a slip-ring or telemetry arrangement, which is precisely why
   that dataset is valuable.

Segmenting magnets axially and circumferentially reduces this loss for the same
reason lamination reduces core loss — it interrupts the eddy current path.

---

### 2.4 Mechanical loss

**Bearing friction.** For a rolling-element bearing under load, friction torque
is approximately constant with speed at a given load, so power is roughly linear
in speed:

```
P_bear = τ_friction · ω,     τ_friction ≈ μ · F_load · d_bore / 2
```

with a load-independent preload/lubricant term added.

**Windage.** Aerodynamic drag on the rotating rotor. Drag torque scales with the
square of surface velocity, so power scales with the cube:

```
P_wind = k_w · ω³
```

The cubic law means windage is negligible at low speed and can dominate at high
speed. For a rotor of radius `r` and length `L` in a gap:

```
P_wind = C_f · π · ρ_air · ω³ · r⁴ · L
```

with `C_f` a friction coefficient depending on the gap Couette/Taylor flow
regime [8]. Sensitive to gap geometry and air density; treat `k_w` as a lumped
coefficient rather than computing `C_f` from scratch.

Both mechanical losses appear in the **rotor and bearing**, not the winding —
which matters for where the heat is injected in the network.

---

### 2.5 Summary: what drives what

| Loss | Scales with | Injected at | Temperature-dependent? |
|---|---|---|---|
| Copper (DC) | `I²` ⇒ torque² | Stator winding | **Yes**, linearly via `α` |
| Copper (AC) | `I²·f²` | Stator winding | Yes, via `R_dc` |
| Iron — hysteresis | `f · B^n` ⇒ speed | Stator yoke / teeth | Weakly |
| Iron — eddy | `f² · B²` ⇒ speed² | Stator yoke / teeth | Weakly (via `σ(T)`) |
| Magnet eddy | `f_harm² · B_ripple²` | Rotor / magnet | Weakly |
| Bearing friction | `ω` | Bearings / housing | Via lubricant viscosity |
| Windage | `ω³` | Rotor surface / gap | Via air density |

Only copper loss has a strong, destabilising temperature coupling. That is why
it alone determines whether the node equation stays linear and whether runaway is
possible.

---

## 3. Heat transfer

### 3.1 Conduction

Fourier's law, one-dimensional through a slab of thickness `L`, cross-sectional
area `A` and conductivity `k`:

```
Q = k·A·ΔT / L        ⇒        R_cond = L / (k·A)      [K/W]
```

The conduction chain from the heat source outward is:

```
winding copper → slot insulation → stator tooth/yoke iron → frame interface → housing
```

**The slot liner dominates.** Copper conducts at ~400 W/(m·K) and electrical
steel at ~25–40 W/(m·K), but the slot insulation system — liner, impregnation
varnish, and the air voids left by imperfect impregnation — is 0.15–0.3 W/(m·K).
A 0.3 mm insulation layer can present more thermal resistance than centimetres of
iron. This is the single most important conduction resistance in the machine and
also the least reliably known, because it depends on impregnation quality, which
varies unit to unit.

**Interface / contact resistance.** Where two solid surfaces meet — stator core
to housing, for instance — real contact occurs only at asperities. The resulting
resistance is usually modelled as an equivalent air gap of 0.01–0.08 mm [1]. It
is assembly-dependent and effectively impossible to predict *a priori*. It is a
prime candidate for parameter identification against measured data.

### 3.2 Convection

Newton's law of cooling:

```
Q = h·A·(T_surface − T_fluid)        ⇒        R_conv = 1 / (h·A)     [K/W]
```

`h` is not a material property — it depends on flow regime, geometry and fluid,
and is obtained from empirical correlations of the form `Nu = f(Re, Pr)`:

```
h = Nu · k_fluid / L_char
```

Two convection paths matter:

**Housing to ambient.** Natural convection for a TEFC machine gives
`h ≈ 5–25 W/(m²·K)`; forced convection from a shaft-mounted fan gives
`h ≈ 20–100 W/(m²·K)`, rising with speed. Fan-driven `h` is speed-dependent and
must be a function, not a constant.

**Housing/jacket to coolant.** For a water jacket, `h` is much larger,
`500–5000 W/(m²·K)`, and depends on flow rate through the Reynolds number. For
turbulent internal flow the Dittus–Boelter correlation is standard:

```
Nu = 0.023 · Re^0.8 · Pr^0.4          (heating of the fluid)
Re = ρ·v·D_h / μ
```

The `Re^0.8` dependence means **the coolant-side resistance is a function of flow
rate**, not a constant. Halving flow raises the coolant-side resistance by a
factor of `2^0.8 ≈ 1.74`. Any model with a fixed coolant resistance cannot
represent a pump failure or a flow-rate design change — which is usually the
whole point of building one.

### 3.3 Radiation

Stefan–Boltzmann, from the housing surface to surroundings:

```
Q = ε·σ_SB·A·(T_s⁴ − T_surr⁴),      σ_SB = 5.670 × 10⁻⁸ W/(m²·K⁴)
```

Non-linear in temperature, which would make the network non-linear. Because the
temperature differences involved are modest, it is standard to linearise about an
operating point into an equivalent radiative heat-transfer coefficient:

```
h_rad = ε·σ_SB·(T_s² + T_surr²)·(T_s + T_surr)
```

so it can be added in parallel with `h_conv` at the housing surface. For a
painted housing (`ε ≈ 0.85`) at 60 °C in a 25 °C room, `h_rad ≈ 6 W/(m²·K)` —
comparable to natural convection, and therefore **not negligible** for a
naturally-cooled machine. For a water-jacketed machine it is a rounding error.

### 3.4 The air gap

The air gap between rotor and stator is the thermal bottleneck of the machine,
and deserves separate treatment.

It is typically 0.5–2 mm of air, `k_air ≈ 0.026 W/(m·K)` — roughly four orders of
magnitude worse a conductor than copper. That thin layer is the *only* radial
path between rotor and stator.

It is not pure conduction, because the rotor is spinning. The flow regime depends
on the Taylor number:

```
Ta = ω²·r_m·δ³ / ν²
```

Below the critical Taylor number the flow is laminar Couette and the gap behaves
as a conduction layer. Above it, counter-rotating Taylor vortices form and the
effective conductivity rises sharply. This is captured by an effective
conductivity ratio:

```
k_eff / k_air = f(Ta)      ≈ 1 below Ta_crit, rising above it
```

Consequences for the model:

- **Rotor heat mostly does not go radially.** It leaves through the shaft and
  bearings, and by convection into gap air that then exits axially.
- **Gap resistance is speed-dependent.** A magnet-temperature model with a fixed
  air-gap resistance will misestimate the rotor at high speed.
- **The rotor node is the least observable and the most fragile.** Poor cooling
  path + irreversible demagnetisation limit (§2.3) = the node that matters most
  and is known least.

---

## 4. The governing balance

### First law

Applied to a lumped node `i` of thermal capacitance `C_i`:

```
C_i · dT_i/dt  =  P_i  +  Σ_j (T_j − T_i)/R_ij  −  Σ_b (T_i − T_b)/R_ib
                  ────     ──────────────────      ──────────────────
                  loss     conducted in from       lost to boundaries
                  injected neighbouring nodes      (coolant, ambient)
```

`C_i = m_i · c_p,i` — mass times specific heat capacity.

In matrix form for the whole network, with `T` the state vector and `P` the loss
injection vector:

```
C · dT/dt = −G·T + P + G_b·T_b
```

where `G` is the conductance matrix (`G_ij = 1/R_ij` off-diagonal, negative row
sums on the diagonal) and `G_b` couples to boundary temperatures. `G` is
**symmetric** — `R_ij = R_ji` — which is a structural check worth asserting in
code.

Steady state (`dT/dt = 0`) gives `T = G⁻¹(P + G_b·T_b)`, which is how the
analytical gate computes the expected equilibrium.

### Second law

Heat flows only down a temperature gradient. For every pair of connected nodes,
the flux `Q_ij = (T_j − T_i)/R_ij` must satisfy:

```
sign(Q_ij) = sign(T_j − T_i)
```

With `R_ij > 0` this is automatic — which is precisely why **negative thermal
resistances must be forbidden** during parameter identification. An optimiser
minimising fit error will happily produce a negative resistance if it improves
the residual, and the result is a model that transports heat up a gradient. The
constraint `R > 0` is a physical law, not a regularisation choice, and must be
enforced as a hard bound.

Both laws are asserted in the physics gate (`tests/`), not assumed.

### Energy conservation over a rollout

Integrated over any interval, for the whole network:

```
∫ ΣP_i dt  −  ∫ Q_out dt  =  Σ C_i·ΔT_i
   energy in     energy out      energy stored
```

This must close to numerical tolerance over every rollout. It is the strongest
single check available on an integrator, because it fails loudly if the solver is
taking steps that are too large or if a conductance is mis-signed.

---

## 5. Node network

Four capacitive nodes. Coolant and ambient are **boundary conditions** — they
have prescribed temperatures and infinite effective capacitance, so they are not
integrated.

| Node | Material | Loss injected | Why it exists separately |
|---|---|---|---|
| Stator winding | Copper + insulation | Copper (DC + AC) | Hottest node; insulation life limit |
| Stator yoke/teeth | Electrical steel | Iron (hysteresis + eddy) | Different loss law and time constant |
| Permanent magnet / rotor | NdFeB + rotor iron | Magnet eddy, windage | Demagnetisation limit; isolated by air gap |
| Housing | Aluminium | Bearing friction | Interface to coolant and ambient |

Connections:

```
winding  ──[R_slot_insulation]──  yoke
yoke     ──[R_core_to_housing]──  housing        (interface resistance)
magnet   ──[R_airgap(ω)]────────  winding/teeth  (speed-dependent)
magnet   ──[R_shaft_bearing]────  housing
housing  ──[R_conv_coolant(ṁ)]──  COOLANT (BC)   (flow-rate dependent)
housing  ──[R_conv+rad_ambient]─  AMBIENT (BC)
```

Two resistances are explicitly **functions, not constants**: the air-gap
resistance depends on speed, and the coolant resistance depends on flow rate.

Four nodes is the smallest network that separates the two failure-critical
temperatures (winding insulation, magnet demagnetisation) from the two things
that can actually be measured easily (housing, coolant). Wallscheid & Böcker [2]
show a four-node network is sufficient for accurate PMSM temperature estimation
after identification, which is the direct precedent for this structure.

---

## 6. Stiffness

The nodes have very different time constants `τ = R·C`:

| Node | Typical `τ` | Why |
|---|---|---|
| Stator winding | tens of seconds | Small copper mass, poor path out through the slot liner |
| Magnet / rotor | minutes | Poor path across the air gap |
| Stator yoke | minutes | Large iron mass |
| Housing | tens of minutes | Large aluminium mass, good coupling to coolant |

A spread of 10–100× between the fastest and slowest node makes the system
**stiff**. For an explicit fixed-step integrator, stability requires a step
smaller than the *fastest* time constant, while the simulation must run for
several multiples of the *slowest* — so the step count is set by the ratio.

Explicit Euler on a stiff system fails in one of two ways, and the second is
worse:

1. It becomes unstable and the temperature oscillates or diverges — obvious.
2. It stays stable but accumulates systematic error, producing a smooth,
   plausible-looking curve that is wrong — **silent**.

The fix is an implicit or adaptive-step method. `scipy.integrate.solve_ivp` with
`method="Radau"` (implicit Runge–Kutta) or `"BDF"` (backward differentiation) is
appropriate; both are L-stable and handle stiffness without a step-size penalty.

A useful property to exploit: because `R(T)` is linear in `T` (§2.1b), the whole
network is **linear time-invariant for fixed inputs**. Over an interval of
constant loss and boundary temperature the solution is a matrix exponential:

```
T(t + Δt) = e^{AΔt}·T(t) + A⁻¹(e^{AΔt} − I)·b,     A = −C⁻¹G,  b = C⁻¹(P + G_b T_b)
```

which is exact at any step size. This is worth using where inputs are piecewise
constant, and it makes the analytical gate exact rather than tolerance-bounded.

---

## 7. What a lumped model cannot capture

Stated plainly, because a model's limits are part of its specification.

**No spatial gradient within a node.** The winding is represented by one
temperature. In reality the conductor at the slot centre is hotter than the one
against the tooth wall, and the end-winding is hotter than the active length. A
lumped model reports something between them, and cannot say which. If the
question is "will the hottest spot exceed the insulation class", a lumped model
cannot answer it without an empirically-calibrated hot-spot factor.

**No transient hot spots.** A brief overload heats the copper long before the
iron responds. The four-node model captures this to the extent that the winding
node has its own capacitance, but sub-second thermal excursions within the slot
are invisible.

**Non-uniform slot fill is invisible.** Impregnation quality, conductor packing
and void fraction vary by unit and even by slot. They are folded into a single
`R_slot`, which is why that parameter must be *identified*, not looked up, and
why the identified value legitimately differs between two nominally identical
motors.

**Axial variation is ignored.** End-winding and active length are lumped.

**Contact resistances are unknowable a priori.** Core-to-housing interface
depends on assembly. It must come from fitting.

**No electromagnetic coupling back from temperature.** In reality, magnet
remanence `B_r` falls ~0.1%/K, so a hot magnet produces less torque per amp,
requiring more current for the same torque, producing more copper loss. This is a
second feedback loop the model does not close.

**The consequence for the build:** parameters that a lumped model cannot know
from geometry — `R_slot`, contact resistances, `h` coefficients — are exactly the
ones Part 6 identifies against measured data. Parameters that *are* knowable —
capacitances from `m·c_p`, and copper's `α` — should be fixed at their published
values and **not** handed to the optimiser. Letting an optimiser adjust a known
material property to improve a fit destroys the only ground truth in the model.

---

## 8. Constants and properties

Physical constants (exact or standard):

| Symbol | Value | Note |
|---|---|---|
| `α_Cu` | 3.93 × 10⁻³ K⁻¹ | Annealed copper, at 20 °C reference |
| `σ_SB` | 5.670374 × 10⁻⁸ W/(m²·K⁴) | Stefan–Boltzmann |
| `ρ_Cu` (resistivity) | 1.68 × 10⁻⁸ Ω·m at 20 °C | |

Material properties needed for capacitances (`C = m·c_p`) and conduction
(`R = L/kA`) — to be filled with sourced values and `plausible_range` in
`params/thermal_params.yaml` in step 2:

| Material | `c_p` [J/(kg·K)] | `ρ` [kg/m³] | `k` [W/(m·K)] |
|---|---|---|---|
| Copper | ~385 | ~8960 | ~400 |
| Electrical steel (M-19 / M-235) | ~460 | ~7650 | ~25–40 (in-plane) |
| NdFeB magnet | ~440 | ~7500 | ~9 |
| Aluminium (housing) | ~900 | ~2700 | ~170–200 |
| Slot insulation system | — | — | ~0.15–0.3 |
| Air (gap, 60 °C) | ~1007 | ~1.06 | ~0.028 |

Every one of these carries a citation in step 2. Ranges reflect grade and
temperature dependence, and become the `plausible_range` for domain
randomisation — a genuinely useful dual purpose, since the same spread that
represents "we do not know exactly which alloy" also represents "this must work
across a family of motors".

Anisotropy note: laminated steel conducts well in-plane and poorly through the
stack (roughly an order of magnitude worse across laminations, because of the
interlaminar insulation). If axial conduction matters, one scalar `k` is wrong.

---

## 9. References

1. Boglietti, Cavagnino, Staton, Shanel, Mueller, Mejuto — *Evolution and Modern
   Approaches for Thermal Analysis of Electrical Machines*, IEEE Trans. Ind.
   Electron. **56**(3):871–882, 2009.
2. Wallscheid & Böcker — *Global Identification of a Low-Order Lumped-Parameter
   Thermal Network for Permanent Magnet Synchronous Motors*, IEEE Trans. Energy
   Convers. **31**(1):354–365, 2016.
3. Wallscheid & Böcker — *Design and Identification of a Lumped-Parameter Thermal
   Network for PMSMs Based on Heat Transfer Theory and Particle Swarm
   Optimisation*, EPE 2015.
4. Boglietti, Carpaneto, Cossale, Vaschetto — *Stator-Winding Thermal Models for
   Short-Time Thermal Transients: Definition and Validation*, IEEE Trans. Ind.
   Electron. **63**(5):2713–2721, 2016.
5. Mellor, Roberts, Turner — *Lumped parameter thermal model for electric
   machines of TEFC design*, IEE Proc. B, 1991.
6. Kirchgässner, Wallscheid, Böcker — *Estimating Electric Motor Temperatures
   with Deep Residual Machine Learning*, IEEE Trans. Power Electron.
   **36**(7):7480–7488, 2020.
7. Kirchgässner, Wallscheid, Böcker — *Thermal Neural Networks: Lumped-Parameter
   Thermal Modeling with State-Space Machine Learning*, arXiv:2103.16323.
8. Bergman, Incropera, DeWitt, Lavine — *Fundamentals of Heat and Mass Transfer*,
   Wiley. Source for material properties, convection correlations and the
   Taylor-number air-gap treatment.
