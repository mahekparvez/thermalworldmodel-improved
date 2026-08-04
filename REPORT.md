# Report — Physics-Grounded Motor Thermal World Model

Two models, identical held-out data, reported side by side.

| | |
|---|---|
| Dataset | Paderborn LEA PMSM, 1,330,816 samples @ 2 Hz, 69 profiles, 184.8 h |
| Split | **55 fit / 14 held out**, by entire profile ID, fixed in `params/data_split.json` |
| Held-out | 58 windows, 13 profiles, 58.0 h, never seen in fitting, training or validation |
| Evaluation | **Free-running rollout** — the model is fed its own output, never a measurement, after t = 0 |
| Reported in | **Kelvin** |

---

## 1. Headline

| Node | LPTN fitted<br>MAE (K) | TNN learned<br>MAE (K) | LPTN RMSE | TNN RMSE |
|---|---|---|---|---|
| Stator winding | 3.38 | **2.62** | 4.60 | **4.31** |
| Stator yoke | **1.88** | 2.27 | **2.50** | 3.18 |
| Permanent magnet | **4.30** | 4.44 | **5.53** | 6.45 |
| **All nodes** | 3.19 | **3.11** | **4.40** | 4.84 |

Error against how long the model has been left running alone:

| Horizon | Wall time | LPTN (K) | TNN (K) |
|---|---|---|---|
| 1 step | 5 s | 0.30 | **0.24** |
| 10 steps | 1 min | 1.74 | **1.34** |
| 100 steps | 8 min | **2.89** | 3.02 |
| 700 steps | 58 min | 3.42 | **2.92** |

### The finding that matters

**188× more parameters bought 2.5%.** A 14-parameter physics model fitted to
the same data lands within a tenth of a Kelvin of a 2,631-parameter neural
network on overall MAE, and *beats* it on RMSE (4.40 vs 4.84) and on two of
three nodes.

Where the TNN genuinely wins is narrower and worth stating precisely:

- **Winding, −22%** (3.38 → 2.62 K). This is the node that matters most, since
  insulation life halves per ~10 K.
- **Long horizon, −15%** (3.42 → 2.92 K at 58 min). The LPTN's error *grows*
  with horizon; the TNN's is flat, which is what the rollout training and drift
  penalty were for.

Where it loses: the yoke (+21%) and the magnet RMSE (+17%). The TNN trades
accuracy on the well-observed, slow nodes for accuracy on the fast one.

**If you want one model, take the fitted LPTN.** It is 14 numbers, every one
physically interpretable, it runs in closed form, and it is within 3% of a
neural network on this data. The TNN earns its place only if the winding
specifically, or the hour-long horizon specifically, is what you care about.

---

## 2. What was and was not fitted

Fitted: 14 parameters marked `fit: true` in `params/thermal_params.yaml`, bounded
by their `plausible_range`.

**Not** fitted, and deliberately so: all four capacitances (from `m·c_p`), the
copper temperature coefficient `α`, and every published material property.
Letting an optimiser adjust a knowable constant to improve a residual destroys
the only ground truth in the model and lets a structural error hide inside a
physically meaningless value. `src/params.py` raises if any parameter is both
knowable and marked fittable — which caught one of my own mistakes (see §6).

| Parameter | Prior | Fitted | Ratio |
|---|---|---|---|
| `winding_to_yoke` | 0.025 | 0.0450 | 1.80× |
| `yoke_to_housing` | 0.010 | 0.0109 | 1.09× |
| `magnet_to_winding_airgap` | 1.20 | 2.055 | 1.71× |
| `magnet_to_housing_shaft` | 2.20 | **0.800** | 0.36× ⚠ |
| `housing_to_coolant` | 0.007 | 0.0140 | 2.00× |
| `housing_to_ambient` | 0.55 | **0.200** | 0.36× ⚠ |
| `copper.phase_resistance_20C` | 0.020 | 0.00864 | 0.43× |
| `copper.ac_excess_coefficient` | 0.0 | 0.108 | — |
| `iron.hysteresis_coefficient` | 0.020 | **0.050** | 2.50× ⚠ |
| `iron.eddy_coefficient` | 6.0e-5 | 8.03e-5 | 1.34× |
| `magnet_eddy.coefficient` | 4.0e-6 | 1.99e-5 | 4.99× |
| `bearing_friction_torque` | 0.10 | **0.350** | 3.50× ⚠ |
| `windage_coefficient` | 1.5e-7 | 2.45e-7 | 1.63× |
| `slot_insulation.conductivity` | 0.22 | 0.220 | 1.00× |

---

## 3. Four parameters pinned at a bound — the model structure is imperfect

⚠ marks a parameter that hit its `plausible_range` limit. **A parameter that had
to reach its bound to fit the data means the structure is wrong, not the
parameter.** All four point the same way:

| Pinned | Direction | Reading |
|---|---|---|
| `iron.hysteresis_coefficient` | upper | wants **more** speed-driven heat |
| `bearing_friction_torque` | upper | wants **more** speed-driven heat |
| `magnet_to_housing_shaft` | lower | wants a **better** rotor heat path |
| `housing_to_ambient` | lower | wants **more** heat leaving the housing |

Two of these are the same statement twice: the model cannot generate enough
speed-dependent loss. Corroborating evidence — `magnet_eddy` went up 5×, and
`ac_excess_coefficient` moved off zero to 0.108, meaning the fit wants
frequency-dependent copper loss that the priors set to nothing. Meanwhile
`phase_resistance_20C` was pushed *down* 2.3×, i.e. **less** current-driven heat.

The consistent story: **the split between torque-driven and speed-driven loss is
wrong in the priors**, and the fit is straining against its bounds to correct it.

Likely structural causes, in order of my confidence:

1. **Pole-pair count is unsourced.** `meta.pole_pairs = 3` is a guess and is not
   published with the dataset. Electrical frequency `f = p·n/60` enters iron
   loss *quadratically* in the eddy term. If the real machine has more pole
   pairs, every speed-driven loss is underestimated at the same coefficient, and
   the optimiser can only compensate by pinning those coefficients high. **This
   single unknown could account for most of the pinning**, and it is cheap to
   resolve.
2. **Field weakening is not modelled.** `flux_density_T` is held constant, but
   above base speed the machine weakens the field. Iron loss should fall, not
   keep rising.
3. **The tooth is lumped into the yoke.** The dataset measures `stator_tooth`
   separately and it runs much hotter (up to 111.9 °C vs the yoke's 101.1 °C).
   A four-node model has to average them, which biases the winding→yoke path.

---

## 4. The three gates

### Gate 1 — analytical: **PASS**

Isolated single node, constant power, against `ΔT = P·R(1 − e^{−t/RC})`.

| Check | Result |
|---|---|
| Steady-state rise equals `P·R` | 8.9e-15 K |
| Radau transient vs closed form | 9.1e-8 K over 4τ |
| `step_exact` at dt = 1 s … 10τ | 0 – 4.3e-14 K |
| Recovered time constant vs `R·C` | 50.4 s vs 50.7 s |

Getting this exact required a correction worth recording: the lossless floating
subnetwork settles *at* the housing temperature, so its links carry **zero**
steady current and must not enter the effective resistance. Including them was
wrong by 2.45e-8 K.

### Gate 2 — physics: **PASS** (17/17 checks, `tests/test_gates.py`)

First law over rollouts, instantaneous balance to 9.1e-12 W, second law over 800
randomised states (0 violations), symmetry of the conductance matrix,
boundedness, monotonicity in current and in coolant flow, air-gap resistance
falling with speed, and thermal runaway detected by eigenvalue rather than by
waiting for a blow-up.

Also verified: **a negative resistance is rejected**, not silently accepted. An
optimiser will propose one if it lowers the residual, and it would transport heat
up a gradient.

For the TNN this gate is **structural rather than tested** — conduction is
written as an antisymmetric exchange, so whatever leaves one node arrives at the
other exactly. Measured energy residual: **0.0 W at initialisation**, 1.2e-4 W
after training (float32 accumulation, not a physics violation).

### Gate 3 — reality: **PASS**, and the only one that could have failed

14 profiles, 58 h, never seen. Results in §1. Free-running: no measurement is fed
back after t = 0.

---

## 5. Integrity of the evaluation

- **Split by entire profile ID.** Adjacent 2 Hz samples are near-identical; a
  timestep split would leak almost perfectly.
- **Priors were set from the fit split only.** When the geometric priors turned
  out to predict a 502 °C winding (§6), the regression that rescaled them used
  the 55 fit profiles and never the 14 held-out ones.
- **Validation profiles came out of the fit split**, not the test split. The
  held-out profiles were touched exactly once, at the end.
- **The integrator was verified, not assumed.** RK4 at dt = 5 s vs the exact
  matrix exponential: max 5.3e-5 K over 400 steps.
- **The vectorised system assembly is bit-identical** to the per-step version
  (max diff 0.0).

---

## 6. Errors found and corrected during the build

Recorded because a reader judging these numbers should know which were caught by
a check rather than by luck.

| Error | How it surfaced | Consequence if missed |
|---|---|---|
| Geometric priors predicted a **502 °C** winding | Data max is 141 °C | Optimiser starting 17× off |
| `slot_insulation` marked `MATERIAL` **and** fittable | Loader assertion | A "published" constant silently fitted |
| Effective resistance included zero-current links | Gate 1 off by 2.45e-8 K | A wrong gate that still passed loosely |
| `_system_arrays` looped in Python | Fit ran with no visible progress | ~100× slower; unusable iteration |
| 40-min windows | 1000-step horizon returned NaN | A required metric silently absent |

---

## 7. Limitations

**Stated plainly, in order of how much they affect the result.**

1. **Pole-pair count is unsourced.** See §3. The single cheapest thing to fix.
2. **Housing is never measured.** It is a latent state, seeded from the coolant.
   Its identified capacitance is the least trustworthy number here — note the
   LPTN and TNN disagree most on it (7200 vs 4190 J/K).
3. **The tooth is measured but not modelled.** Structural, not a fitting failure.
4. **Geometry is assumed**, not from the machine's drawings, which are not
   published. Every geometry-derived quantity inherits that.
5. **Coolant flow rate is not in the dataset.** It is held at nominal, so the
   `ṁ^-0.8` coolant law is *implemented and unit-tested but never exercised
   against data*. Any claim about pump degradation is untested.
6. **No spatial gradients within a node**, no transient hot spots, no
   axial variation, no conduction between adjacent joints, no electromagnetic
   feedback from magnet temperature to torque constant.
7. **One machine.** Everything here is fitted to a single prototype PMSM. The
   domain randomisation ranges in `thermal_params.yaml` express what *ought* to
   generalise, but that has not been tested on a second motor.
8. **3 K MAE is not a safety margin.** Max errors reach 26 K (LPTN) and 30 K
   (TNN) on the magnet. For a demagnetisation limit that is irreversible when
   exceeded, the max matters more than the mean.

## 8. What would most improve this

1. **Confirm the pole-pair count.** One integer; would likely release four pinned
   parameters.
2. **Add a fifth node for the stator tooth.** The measurement already exists and
   is currently discarded.
3. **Log coolant flow rate.** Would make the flow-dependent resistance testable
   rather than merely implemented.
4. **A second machine.** Nothing here demonstrates transfer, and transfer is the
   whole claim of a "world model".

---

## 9. Reproducing

```bash
python -m pip install numpy scipy pandas pyyaml torch kaggle
kaggle datasets download -d wkirgsn/electric-motor-temperature -p data/raw --unzip
PYTHONPATH=src python tests/test_gates.py     # gates 1 and 2
PYTHONPATH=src python src/identify.py         # LPTN fit + reality gate
PYTHONPATH=src python src/train.py            # TNN train + reality gate
```

Artefacts: `params/identified.json`, `params/tnn_result.json`, `params/tnn.pt`,
`params/data_split.json`.
