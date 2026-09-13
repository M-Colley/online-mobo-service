"""
Pure MOBO core for the VAM study — no Firestore, no Flask, so it can be unit-
tested / simulated offline (see simulate.py).

Given the observation history it fits a Gaussian-process model per objective and
returns the next candidate as a *snapped, on-grid* raw parameter dict.

  • Categorical dims present  ->  MixedSingleTaskGP + optimize_acqf_mixed
  • No categorical dims        ->  SingleTaskGP    + optimize_acqf   (proven path)

The continuous optimum is snapped to the JND grid in space.snap_candidate(),
so no candidate is ever finer than a JND. qLogNEHVI is used for both — it is far
more sample-efficient than TPE in the ~15-trial regime this study runs in.
"""

from __future__ import annotations

import os

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP, MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood

import space

# ── Acqf-optimiser knobs ─────────────────────────────────────────────────────
# Deliberately lighter than the original single-GP service: optimize_acqf_mixed
# runs the inner optimisation once PER categorical combination, so cost scales
# with (#pattern categories). At ≤16 observations and D=5 these values are
# plenty, and they keep a single /updatePolicy call well under the Cloud Run
# 300 s request timeout even on the pure-Python EHVI fallback.
BATCH_SIZE = 1
NUM_RESTARTS = int(os.environ.get("NUM_RESTARTS", "5"))
RAW_SAMPLES = int(os.environ.get("RAW_SAMPLES", "256"))
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "64"))

# Reference point for hypervolume — just below the min possible objective (0).
REF_POINT = [-0.1] * len(space.OBJECTIVE_FIELDS)

DTYPE = torch.double


def build_and_fit_model(train_X: torch.Tensor, train_Y: torch.Tensor) -> ModelListGP:
    """One GP per objective. Mixed kernel iff the space has categorical dims."""
    models = []
    for i in range(train_Y.shape[1]):
        y_col = train_Y[:, i].unsqueeze(-1)
        if space.D_CAT > 0:
            m = MixedSingleTaskGP(
                train_X=train_X,
                train_Y=y_col,
                cat_dims=space.CAT_DIMS,
                outcome_transform=Standardize(m=1),
            )
        else:
            m = SingleTaskGP(
                train_X=train_X,
                train_Y=y_col,
                outcome_transform=Standardize(m=1),
            )
        models.append(m)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _optimise_acqf(model, train_X: torch.Tensor) -> list[float]:
    """Return the best candidate row in model space (cont normalised + cat ints)."""
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=REF_POINT,
        X_baseline=train_X,
        prune_baseline=True,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
    )
    bounds = torch.tensor(space.model_bounds(), dtype=DTYPE)

    if space.D_CAT > 0:
        candidate, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            fixed_features_list=space.fixed_features_list(),
            options={"batch_limit": 5, "maxiter": 200},
        )
    else:
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
    return candidate.detach().squeeze(0).tolist()


def choose_next(
    X_model: torch.Tensor,
    Y: torch.Tensor,
    observed_keys: set,
    dedup_seed: int = 0,
) -> dict:
    """Fit the GP, optimise qLogNEHVI, snap to the JND grid, ensure novelty.

    Parameters
    ----------
    X_model : [n, D] tensor — normalised-continuous ++ integer-categorical rows.
    Y       : [n, m] tensor — objective values, all maximised, in [0, 1].
    observed_keys : set of space.obs_key(...) already tested for this participant.

    Returns a raw Firestore parameter dict (on-grid, feasible, novel).
    """
    model = build_and_fit_model(X_model, Y)
    row = _optimise_acqf(model, X_model)
    raw = space.snap_candidate(row)
    raw = space.project_feasible(raw)  # minimal repair, keeps the model's choice

    # The snapped optimum can coincide with a point already tested, or violate a
    # hard constraint. Fall back to a fresh feasible on-grid config — the novelty
    # test lives inside random_feasible(exclude=...), the same one every other
    # proposal path uses.
    if space.is_feasible(raw) and space.obs_key(raw) not in observed_keys:
        return raw
    return space.random_feasible(seed=1_000_000 + dedup_seed * 1000, exclude=observed_keys)
