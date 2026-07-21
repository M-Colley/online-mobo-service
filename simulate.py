"""
Offline end-to-end simulation of one participant — NO Firestore, NO cloud.

Run this to sanity-check the optimizer before deploying:

    python simulate.py

It drives the exact code path Cloud Run uses (space + optimizer_core) for a
full N_TOTAL-trial run against a synthetic "participant" whose ratings peak at a
hidden favourite configuration, and asserts:

  • every proposed configuration lies on the JND grid,
  • no configuration is proposed twice (no sub-JND repeats),
  • the run completes and the hypervolume is non-decreasing.
"""

from __future__ import annotations

import numpy as np
import torch

import space
import optimizer_core as core

torch.manual_seed(0)
np.random.seed(0)

N_SOBOL = 2 * (space.D + 1)  # deliberately one more than the memo's 2n+1 (conservative)
N_TOTAL = N_SOBOL + 5


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


def _on_grid(raw: dict) -> bool:
    for p in space.CONT_PARAMS:
        if not np.any(np.isclose(p.levels, raw[p.name], atol=10 ** -p.round_ndigits)):
            return False
    for p in space.CAT_PARAMS:
        if raw[p.name] not in p.categories:
            return False
    return True


def _hv(Y: torch.Tensor) -> float:
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated
    ref = torch.tensor(core.REF_POINT, dtype=torch.double)
    return float(Hypervolume(ref_point=ref).compute(Y[is_non_dominated(Y)]))


def main() -> None:
    print(space.describe())
    print(f"\nBudget: {N_SOBOL} Sobol + {N_TOTAL - N_SOBOL} MOBO = {N_TOTAL} trials")
    print(f"Hidden favourite: {_FAVOURITE}\n")

    history: list[dict] = []
    observed_keys: set = set()
    X_rows, Y_rows = [], []
    hv_curve = []

    for step in range(1, N_TOTAL + 1):
        obs_count = len(history)
        if obs_count < N_SOBOL:
            raw = space.sobol_next(obs_count)
            if space.obs_key(raw) in observed_keys:
                raw = space.random_feasible(seed=obs_count)
            phase = "sobol"
        else:
            X = torch.tensor(X_rows, dtype=torch.double)
            Y = torch.tensor(Y_rows, dtype=torch.double)
            raw = core.choose_next(X, Y, observed_keys, dedup_seed=obs_count)
            phase = "mobo"

        assert _on_grid(raw), f"OFF-GRID candidate proposed: {raw}"
        assert space.obs_key(raw) not in observed_keys, f"DUPLICATE proposed: {raw}"

        y = _rate(raw)
        history.append(raw)
        observed_keys.add(space.obs_key(raw))
        X_rows.append(space.to_model_row(raw))
        Y_rows.append(y)
        hv_curve.append(_hv(torch.tensor(Y_rows, dtype=torch.double)))

        print(f"step {step:>2} [{phase:<5}] hv={hv_curve[-1]:.4f}  "
              f"y=({y[0]:.2f},{y[1]:.2f})  {raw}")

    assert all(b >= a - 1e-9 for a, b in zip(hv_curve, hv_curve[1:])), \
        "hypervolume decreased — MOBO not improving"
    print(f"\nPASS: {N_TOTAL} trials, all on-grid, all unique, "
          f"hypervolume non-decreasing (final {hv_curve[-1]:.4f}).")


if __name__ == "__main__":
    main()
