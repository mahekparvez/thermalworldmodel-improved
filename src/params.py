"""
Parameter loading, provenance tracking, and geometry-derived cross-checks.

Two jobs:

1. Load `params/thermal_params.yaml` and expose it as typed objects, keeping
   `source`, `plausible_range` and `fit` attached to every value. Provenance
   that is not carried through to the output is provenance that will be lost.

2. RECOMPUTE the capacitances from geometry x material properties and compare
   against the values recorded in the YAML. The YAML numbers exist for
   traceability; if they ever drift from what the geometry implies, that is a
   transcription error and the loader raises rather than silently preferring one.

Nothing here fits anything. `fit: true` only marks a parameter as eligible for
identification in step 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PARAM_FILE = Path(__file__).resolve().parents[1] / "params" / "thermal_params.yaml"

NODES = ("winding", "yoke", "magnet", "housing")

# Sources that mean "this number is knowable and must never be fitted".
FIXED_SOURCES = {"PHYSICAL", "MATERIAL"}


@dataclass
class Param:
    """One value plus everything needed to defend it."""
    name: str
    value: float
    units: str = ""
    source: str = "UNSOURCED"
    plausible_range: Optional[Tuple[float, float]] = None
    sampling: str = "uniform"
    fit: bool = False
    note: str = ""
    formula: str = ""
    reference: str = ""

    def __post_init__(self):
        if self.plausible_range is not None:
            lo, hi = self.plausible_range
            if not (lo < hi):
                raise ValueError(f"{self.name}: plausible_range {lo}..{hi} is not increasing")
            if self.sampling == "loguniform" and lo <= 0:
                raise ValueError(f"{self.name}: loguniform sampling needs a positive lower bound, got {lo}")
            if not (lo <= self.value <= hi):
                raise ValueError(
                    f"{self.name}: nominal {self.value} lies outside its own "
                    f"plausible_range [{lo}, {hi}]")

    @property
    def is_unsourced(self) -> bool:
        return self.source in ("UNSOURCED", "ESTIMATED")

    def in_range(self, v: float) -> bool:
        if self.plausible_range is None:
            return True
        lo, hi = self.plausible_range
        return lo <= v <= hi


def _walk(node: Any, prefix: str, out: Dict[str, Param]) -> None:
    """Collect every dict that looks like a parameter (has a scalar `value`)."""
    if not isinstance(node, dict):
        return
    if "value" in node and isinstance(node["value"], (int, float)):
        pr = node.get("plausible_range")
        out[prefix] = Param(
            name=prefix,
            value=float(node["value"]),
            units=node.get("units", ""),
            source=node.get("source", "UNSOURCED"),
            plausible_range=(float(pr[0]), float(pr[1])) if pr else None,
            sampling=node.get("sampling", "uniform"),
            fit=bool(node.get("fit", False)),
            note=node.get("note", ""),
            formula=node.get("formula", ""),
            reference=node.get("reference", ""),
        )
        return
    for k, v in node.items():
        _walk(v, f"{prefix}.{k}" if prefix else str(k), out)


@dataclass
class ParamSet:
    raw: Dict[str, Any]
    params: Dict[str, Param] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Param:
        return self.params[key]

    def value(self, key: str) -> float:
        return self.params[key].value

    def fittable(self) -> List[Param]:
        return [p for p in self.params.values() if p.fit]

    def unsourced(self) -> List[Param]:
        return [p for p in self.params.values() if p.is_unsourced]

    def fixed_but_marked_fit(self) -> List[Param]:
        """Published constants someone flagged as fittable. Always a mistake."""
        return [p for p in self.params.values()
                if p.fit and p.source in FIXED_SOURCES]

    # -- geometry-derived cross-checks -------------------------------------

    def derived_capacitances(self) -> Dict[str, float]:
        """C = m * c_p, recomputed from first principles."""
        g, m = self.raw["geometry"], self.raw["materials"]
        cu = m["copper"]["specific_heat"]["value"]
        fe = m["electrical_steel"]["specific_heat"]["value"]
        nd = m["ndfeb"]["specific_heat"]["value"]
        al = m["aluminium"]["specific_heat"]["value"]
        return {
            "winding": g["copper_mass_kg"]["value"] * cu,
            "yoke": g["steel_mass_kg"]["value"] * fe,
            "magnet": (g["magnet_mass_kg"]["value"] * nd
                       + g["rotor_iron_mass_kg"]["value"] * fe),
            "housing": g["housing_mass_kg"]["value"] * al,
        }

    def derived_slot_resistance(self) -> float:
        """R = t / (k A) for the winding-to-yoke path through the slot liner."""
        g, m = self.raw["geometry"], self.raw["materials"]
        t = g["slot_liner_thickness_m"]["value"]
        k = m["slot_insulation"]["conductivity"]["value"]
        # Slot wall area: treat each slot as a rectangular channel whose wall the
        # liner covers. Perimeter estimated from the stator tooth pitch.
        r_in = g["stator_inner_radius_m"]["value"]
        n_slots = g["slot_count"]["value"]
        L = g["active_length_m"]["value"]
        slot_pitch = 2 * math.pi * r_in / n_slots
        slot_depth = 0.55 * (g["stator_outer_radius_m"]["value"] - r_in)
        perimeter = 2 * slot_depth + slot_pitch      # two sides + bottom-ish
        area = n_slots * perimeter * L
        return t / (k * area)

    def check_consistency(self, rel_tol: float = 0.02) -> List[str]:
        """Recorded values vs what the geometry implies. Returns complaints."""
        problems = []
        derived = self.derived_capacitances()
        for node, dval in derived.items():
            recorded = self.value(f"capacitances.{node}.value") \
                if f"capacitances.{node}.value" in self.params \
                else self.value(f"capacitances.{node}")
            if abs(recorded - dval) > rel_tol * dval:
                problems.append(
                    f"capacitance '{node}': YAML records {recorded:.1f} J/K but "
                    f"geometry x material gives {dval:.1f} J/K "
                    f"({100*(recorded-dval)/dval:+.1f}%)")
        for p in self.fixed_but_marked_fit():
            problems.append(
                f"'{p.name}' has source {p.source} (knowable) but fit: true. "
                "Fitting a published constant destroys the model's ground truth.")
        for name, p in self.params.items():
            if "resistances." in name and name.endswith(".value") is False:
                continue
        return problems


def load(path: Path = PARAM_FILE) -> ParamSet:
    raw = yaml.safe_load(path.read_text())
    flat: Dict[str, Param] = {}
    _walk(raw, "", flat)
    ps = ParamSet(raw=raw, params=flat)

    problems = ps.check_consistency()
    if problems:
        raise ValueError("parameter file is internally inconsistent:\n  - "
                         + "\n  - ".join(problems))
    return ps


def provenance_summary(ps: ParamSet) -> str:
    by_source: Dict[str, int] = {}
    for p in ps.params.values():
        by_source[p.source] = by_source.get(p.source, 0) + 1
    lines = ["parameter provenance:"]
    for src in sorted(by_source, key=lambda s: -by_source[s]):
        lines.append(f"  {src:12s} {by_source[src]:3d}")
    lines.append(f"  {'-'*24}")
    lines.append(f"  {'fittable':12s} {len(ps.fittable()):3d}")
    lines.append(f"  {'unsourced':12s} {len(ps.unsourced()):3d}  "
                 "(ESTIMATED or UNSOURCED -> must be identified or flagged)")
    return "\n".join(lines)


if __name__ == "__main__":
    ps = load()
    print(provenance_summary(ps))
    print("\ngeometry-derived capacitances (J/K):")
    for k, v in ps.derived_capacitances().items():
        print(f"  {k:8s} {v:8.1f}")
    print(f"\nderived slot resistance: {ps.derived_slot_resistance():.4f} K/W "
          f"(YAML: {ps.value('resistances.winding_to_yoke'):.4f})")
    print("\nfittable parameters:")
    for p in ps.fittable():
        lo, hi = p.plausible_range if p.plausible_range else (float('nan'),) * 2
        print(f"  {p.name:52s} {p.value:>10.4g}  [{lo:.4g}, {hi:.4g}]  {p.sampling}")
    print("\nunsourced / estimated (flagged in REPORT.md):")
    for p in ps.unsourced():
        print(f"  {p.name:52s} {p.source}")
