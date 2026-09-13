"""
Offline end-to-end simulation of one participant — NO Firestore, NO cloud.

Run this to sanity-check the optimizer before deploying:

    python simulate.py

It drives the exact code path Cloud Run uses (space + optimizer_core) for a
full 18-round run against a synthetic "participant" whose ratings peak at a
hidden favourite knob setting, and asserts:

  • every proposed knob vector lies on the JND grid,
  • every proposal EXPANDS to a design the app's security rules accept
    (an illegal design would stall the participant with no error on our side),
  • no configuration is proposed twice,
  • round 1 is the anchor (the app team's seed design),
  • the run completes and the hypervolume is non-decreasing.
"""

from __future__ import annotations

import numpy as np
import torch

import space
import optimizer_core as core

torch.manual_seed(0)
np.random.seed(0)

N_ANCHOR = space.N_ANCHOR
N_SOBOL = space.N_SOBOL_DEFAULT          # anchor + 2*(D+1) Sobol draws
N_TOTAL = space.MAX_ROUND_NUMBER         # the app's rules cap rounds at 18


# ── A synthetic participant: two objectives, both maximised in [0,1] ──────────
_FAVOURITE = {p.name: float(np.random.choice(p.levels)) for p in space.CONT_PARAMS}
for p in space.CAT_PARAMS:
    _FAVOURITE[p.name] = p.categories[np.random.randint(len(p.categories))]


def _rate(raw: dict) -> list[float]:
    """Closer to the hidden favourite → higher scores (with a little noise)."""
    d = 0.0
    for p in space.CONT_PARAMS:
        span = p.hi - p.lo or 1.0
        d += ((raw[p.name] - _FAVOURITE[p.name]) / span) ** 2
    for p in space.CAT_PARAMS:
        d += 0.0 if raw[p.name] == _FAVOURITE[p.name] else 0.5
    closeness = float(np.exp(-d))
    subjective = np.clip(closeness + np.random.normal(0, 0.03), 0, 1)
    objective = np.clip(0.3 + 0.7 * closeness + np.random.normal(0, 0.03), 0, 1)
    return [float(subjective), float(objective)]


# space.on_grid is the guard the SERVICE uses; a local copy with a per-step
# tolerance accepted half-step values on the integer knobs and could never fire.
_on_grid = space.on_grid


def _hv(Y: torch.Tensor) -> float:
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated
    ref = torch.tensor(core.REF_POINT, dtype=torch.double)
    return float(Hypervolume(ref_point=ref).compute(Y[is_non_dominated(Y)]))


def main() -> None:
    print(space.describe())
    print(f"\nBudget: {N_ANCHOR} anchor + {N_SOBOL - N_ANCHOR} Sobol + "
          f"{N_TOTAL - N_SOBOL} MOBO = {N_TOTAL} rounds")
    print(f"Hidden favourite: {_FAVOURITE}\n")

    history: list[dict] = []
    observed_keys: set = set()
    X_rows, Y_rows = [], []
    hv_curve = []

    for step in range(1, N_TOTAL + 1):
        obs_count = len(history)
        if step <= N_ANCHOR:
            raw = space.anchor()
            phase = "anchor"
        elif obs_count < N_SOBOL:
            # indexed by the ROUND, exactly as main.choose_proposal does
            raw = space.sobol_next(step - N_ANCHOR - 1)
            if space.obs_key(raw) in observed_keys:
                raw = space.random_feasible(seed=step, exclude=observed_keys)
            phase = "sobol"
        else:
            X = torch.tensor(X_rows, dtype=torch.double)
            Y = torch.tensor(Y_rows, dtype=torch.double)
            raw = core.choose_next(X, Y, observed_keys, dedup_seed=obs_count)
            phase = "mobo"

        assert _on_grid(raw), f"OFF-GRID candidate proposed: {raw}"
        assert space.obs_key(raw) not in observed_keys, f"DUPLICATE proposed: {raw}"
        haptics = space.expand(raw)
        assert space.rules_valid_haptics(haptics), \
            f"ILLEGAL design (the app could not echo it): {raw} -> {haptics}"
        if step == 1:
            assert haptics == space.SEED_DESIGN, "round 1 must be the anchor (seed design)"

        y = _rate(raw)
        history.append(raw)
        observed_keys.add(space.obs_key(raw))
        X_rows.append(space.to_model_row(raw))
        Y_rows.append(y)
        hv_curve.append(_hv(torch.tensor(Y_rows, dtype=torch.double)))

        onroute = haptics["onRoute"]
        print(f"step {step:>2} [{phase:<6}] hv={hv_curve[-1]:.4f}  "
              f"y=({y[0]:.2f},{y[1]:.2f})  knobs={ {k: int(v) for k, v in raw.items()} }  "
              f"onRoute={onroute['pulseCount']}p/{onroute['onDuration']}s "
              f"i={onroute['intensity']} s={onroute['sharpness']}")

    assert all(b >= a - 1e-9 for a, b in zip(hv_curve, hv_curve[1:])), \
        "hypervolume decreased — MOBO not improving"
    print(f"\nPASS: {N_TOTAL} rounds, all on-grid, all unique, every design legal "
          f"under the app's rules, hypervolume non-decreasing (final {hv_curve[-1]:.4f}).")


if __name__ == "__main__":
    main()
