"""
MOBO search space.

────────────────────────────────────────────────────────────────────────────
THIS IS THE ONLY FILE YOU NORMALLY EDIT.
Everything in main.py / optimizer_core.py reads the search space from here, so
when you finalise the Firestore field names and the parameter ranges you only
touch this file.
────────────────────────────────────────────────────────────────────────────

How just-noticeable differences (JNDs) are encoded
---------------------------------------------------
A JND is the smallest change in a stimulus a participant can perceive. Two
configurations that differ by *less* than a JND are, by definition,
indistinguishable — testing both wastes a precious trial. We therefore
DISCRETISE every continuous parameter onto a perceptually-spaced grid whose
neighbouring levels are exactly one JND apart:

  • Weber's-law parameters (intensity 13 %, interval 20 %) -> GEOMETRIC grid
    (each level = previous * (1 + jnd_fraction)).
  • Absolute-JND parameters (duration, in seconds)          -> LINEAR grid
    (fixed additive step).
  • Naturally discrete parameters (pulse count)             -> explicit level list.
  • Nominal parameters (pattern)                            -> categorical dim.

The optimizer proposes candidates in continuous space (sample-efficient GP
MOBO) and then SNAPS them onto these grids, so the app never receives — and a
participant is never asked to compare — two settings closer than a JND. This is
the "model the parameters as discrete based on the JND" idea, done in a way
that scales (we never enumerate the full ~10^5-10^6 Cartesian product).

Removing configurations "that don't make sense"
-----------------------------------------------
Because the space is discrete, dropping infeasible combinations is trivial:
edit `is_feasible()` below. That is the disciplined equivalent of the Optuna
`raise TrialPruned()` trick, but applied to *hard constraints* only — JNDs are
handled by the grid spacing, not by pruning.

Parameters the app ignores in some modes
----------------------------------------
When a parameter has no effect for certain configurations (here: `interval`
when pattern == "constant" — a sustained vibration has no pulse rate), the
configuration is mapped to a canonical form in `canonicalize()` below. That
stops the optimizer from "exploring" a dead dimension and makes dedup treat
physically-identical stimuli as one configuration.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import numpy as np
# NB: torch is imported lazily inside sobol_next() so this module (the search-space
# definition) can be imported / unit-tested without torch installed.

# ── Objective (Y) field names — read from interventionResults, both maximised ──
# (Unchanged from the original study; add a third here to go to 3 objectives.)
OBJECTIVE_FIELDS = ["subjectiveScore", "objectiveScore"]


# ── Grid builders ────────────────────────────────────────────────────────────
def weber_grid(lo: float, hi: float, jnd_fraction: float) -> np.ndarray:
    """Geometric grid for Weber's-law parameters.

    Consecutive levels differ by `jnd_fraction` (e.g. 0.13 for intensity,
    0.20 for interval): level_{k+1} = level_k * (1 + jnd_fraction).
    """
    if lo <= 0:
        raise ValueError(f"weber_grid needs lo > 0 (got {lo}) — geometric spacing can't start at 0")
    if jnd_fraction <= 0:
        raise ValueError(f"weber_grid needs jnd_fraction > 0 (got {jnd_fraction}) — else it never terminates")
    if hi < lo:
        raise ValueError(f"weber_grid needs hi >= lo (got lo={lo}, hi={hi})")
    levels = [float(lo)]
    while levels[-1] * (1.0 + jnd_fraction) <= hi * (1.0 + 1e-9):
        levels.append(levels[-1] * (1.0 + jnd_fraction))
    return np.array(levels, dtype=float)


def linear_grid(lo: float, hi: float, step: float) -> np.ndarray:
    """Additive grid for parameters whose JND is an absolute amount (seconds)."""
    if step <= 0:
        raise ValueError(f"linear_grid needs step > 0 (got {step})")
    if hi < lo:
        raise ValueError(f"linear_grid needs hi >= lo (got lo={lo}, hi={hi})")
    n = int(np.floor((hi - lo) / step + 1e-9))
    return lo + step * np.arange(n + 1, dtype=float)


def list_grid(values) -> np.ndarray:
    """Explicit level list for naturally discrete parameters (e.g. pulse count)."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        raise ValueError("list_grid needs at least one value")
    return np.sort(arr)  # keep ascending regardless of input order


# ── Parameter descriptors ────────────────────────────────────────────────────
@dataclass
class ContinuousParam:
    name: str                 # <-- Firestore field name. EDIT THESE when finalised.
    levels: np.ndarray        # JND-spaced grid, ascending.
    round_ndigits: int = 3    # rounding when written back to Firestore.

    @property
    def lo(self) -> float:
        return float(self.levels[0])

    @property
    def hi(self) -> float:
        return float(self.levels[-1])

    def normalize(self, x_raw: float) -> float:
        span = self.hi - self.lo
        return 0.0 if span <= 0 else (float(x_raw) - self.lo) / span

    def denormalize(self, u: float) -> float:
        return self.lo + float(u) * (self.hi - self.lo)

    def snap(self, x_raw: float) -> float:
        """Snap a raw value to the nearest perceptually-distinct level."""
        idx = int(np.argmin(np.abs(self.levels - float(x_raw))))
        return round(float(self.levels[idx]), self.round_ndigits)


@dataclass
class CategoricalParam:
    name: str                 # <-- Firestore field name. EDIT when finalised.
    categories: list          # human-readable values stored in Firestore.

    def index(self, value) -> int:
        """Category value (str) OR already-numeric code -> integer index."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(round(value))
        return self.categories.index(value)

    def value(self, index) -> object:
        return self.categories[int(round(float(index)))]


# ═══════════════════════════════════════════════════════════════════════════
#  THE SEARCH SPACE — edit names / ranges here.
#  Numbers below come from the JND memo; every value flagged  # TODO  still
#  needs confirmation (exact Firestore name, unit, and range) before the run.
# ═══════════════════════════════════════════════════════════════════════════
# Field names below MATCH the live Firestore schema (confirmed 2026-07-14 from
# interventionResults / parameterValues docs). Units + ranges for duration,
# interval and pattern were confirmed by the app team (Mahdi) 2026-07-21;
# the intensity and sharpness design ranges (0–1) come from the study memo.
# The remaining optimizer-side choices are flagged # TODO below.
#
# PLANNED, NOT YET IN THE APP (confirmed still planned 2026-07-21): the study
# design adds two more parameters — pulse count (1–4; use list_grid([1,2,3,4]))
# and interval-within-pulse-set (Hz; range TBD). Do NOT add them here until the
# app actually writes those fields (every interventionResults doc must carry
# every field in PARAM_NAMES, or docs are skipped as malformed). When they
# land: get the exact Firestore field names + ranges from the app team, add
# them below, and extend canonicalize() — both are meaningless for "constant"
# (no pulses), and interval-within is meaningless when pulse count == 1. D and
# the trial budget adapt automatically (D=7 → 2·(D+1)=16 Sobol, 21 total).
CONT_PARAMS: list[ContinuousParam] = [
    # intensity — Firestore double, design range 0–1 (study memo; Weber JND ≈ 13 %).
    # Grid floor 0.20 = minimum usefully-perceptible cue — an OPTIMIZER-side
    # choice (a Weber grid needs lo > 0), not an app limit. Revisit after pilot.
    ContinuousParam("intensity", weber_grid(0.20, 1.00, 0.13), round_ndigits=3),

    # sharpness — Firestore double, design range 0–1 (study memo). No
    # psychophysical JND literature (memo: "needs pilot testing"); placeholder
    # uniform 0.20 grid (6 levels).  # TODO set the step after the pilot
    ContinuousParam("sharpness", linear_grid(0.00, 1.00, 0.20), round_ndigits=3),

    # duration — Firestore double, SECONDS; length of EACH PULSE (study memo).
    # App range 0.03–20 s (confirmed by the app team 2026-07-21). JND is
    # ASYMMETRIC (+0.240 s to notice an increase, −0.110 s a decrease); we use
    # the larger step (0.240 s) so neighbours are distinguishable BOTH ways
    # -> 84 levels. NOTE: 20 s is what the APP can render, not necessarily what
    # the STUDY should test — lower `hi` here if multi-second stimuli make
    # trials impractically long. For "puls" the duration×interval constraint in
    # is_feasible() caps pulses at 1/interval − 0.01 s (≤ 0.99 s) anyway; the
    # long tail of this grid is reachable only by "constant".
    ContinuousParam("duration", linear_grid(0.03, 20.00, 0.240), round_ndigits=3),

    # interval — Firestore double, HERTZ (pulses per second; 4 = 4 pulses/s).
    # App range 1–20 Hz (confirmed by the app team 2026-07-21). Weber JND ≈ 20 %
    # (geometric) -> 17 levels. The app IGNORES this field when pattern ==
    # "constant" and stores the minimum (1 Hz) — handled in canonicalize().
    ContinuousParam("interval", weber_grid(1.00, 20.00, 0.20), round_ndigits=3),
]

# Nominal parameter(s). Set CAT_PARAMS = [] to disable the categorical machinery
# entirely (then the plain, already-proven SingleTaskGP path is used).
CAT_PARAMS: list[CategoricalParam] = [
    # pattern — Firestore string. The app supports exactly two patterns:
    # "constant" (the sustained vibration) and "puls" (pulsed). Confirmed by the
    # app team 2026-07-15.
    CategoricalParam("pattern", ["constant", "puls"]),
]

# ── Derived layout of the model input vector ─────────────────────────────────
D_CONT = len(CONT_PARAMS)
D_CAT = len(CAT_PARAMS)
D = D_CONT + D_CAT                       # model input dimensionality
CAT_DIMS = list(range(D_CONT, D))        # categorical column indices for MixedSingleTaskGP
PARAM_NAMES = [p.name for p in CONT_PARAMS] + [p.name for p in CAT_PARAMS]


# ── Canonical-form hook ──────────────────────────────────────────────────────
def canonicalize(raw: dict) -> dict:
    """Map a configuration to its canonical, physically-equivalent form.

    The app IGNORES `interval` when pattern == "constant" (a sustained vibration
    has no pulse rate) and writes the minimum, 1 Hz, into the field — confirmed
    by the app team 2026-07-21. Two "constant" configs differing only in
    interval are therefore the SAME stimulus. Pinning interval to 1.0 here
    (a) keeps the optimizer from spending trials "exploring" a dimension that
    has no effect, and (b) makes obs_key() treat identical stimuli as one
    configuration. Applied everywhere a config is encoded, proposed, or keyed.
    """
    out = dict(raw)
    if out.get("pattern") == "constant":
        out["interval"] = 1.0
    return out


# ── Hard-constraint hook ─────────────────────────────────────────────────────
# The app floors the off-time between pulses at 10 ms: off = max(0.01,
# 1/interval − duration). Beyond that floor the pulses still play full-length,
# the effective rate degrades to 1/(duration + 0.01), and `interval` loses any
# real effect (app team, 2026-07-21) — so such configs are mislabelled stimuli
# and must never be proposed.
PULS_MIN_OFF = 0.01  # seconds
_FEAS_EPS = 1e-9     # float slack so exact-boundary grid levels stay feasible


def is_feasible(raw: dict) -> bool:
    """Return False for configurations that must never be tested.

    JNDs are already handled by the grid — use this ONLY for genuine
    constraints. Further examples (uncomment / adapt to your fields):

        # Forbid a specific pattern at very low intensity:
        # if raw["pattern"] == "constant" and raw["intensity"] < 0.4:
        #     return False
    """
    # For "puls", a full pulse plus the 10 ms minimum gap must fit its period.
    if raw["pattern"] == "puls" and \
            raw["duration"] > 1.0 / raw["interval"] - PULS_MIN_OFF + _FEAS_EPS:
        return False
    return True


def project_feasible(raw: dict) -> dict:
    """Minimally repair an infeasible candidate onto the feasible set.

    Used on MOBO proposals (optimizer_core.choose_next): qLogNEHVI actively
    chases high-uncertainty regions, which includes the never-observed
    infeasible corner — rejecting outright would swap the model's choice for a
    random one, so instead we keep the proposal and lower `duration` to the
    largest grid level that fits the period. (Sobol draws keep plain rejection:
    resampling preserves uniform coverage of the feasible region.)
    """
    if is_feasible(raw):
        return raw
    out = dict(raw)
    if out["pattern"] == "puls":
        budget = 1.0 / out["interval"] - PULS_MIN_OFF
        p = next(p for p in CONT_PARAMS if p.name == "duration")
        fits = [lv for lv in p.levels if lv <= budget + _FEAS_EPS]
        if fits:  # always true for our grids: 0.03 s fits even at 20 Hz
            out["duration"] = round(float(fits[-1]), p.round_ndigits)
    return out


# ── Encoding: Firestore record  <->  GP model row ────────────────────────────
def to_model_row(rec: dict) -> list[float]:
    """One interventionResults/parameterValues doc -> model input row.

    Continuous dims are normalised to [0, 1]; categorical dims are integer
    category indices (MixedSingleTaskGP consumes them raw, not normalised).
    """
    rec = canonicalize(rec)
    row = [p.normalize(float(rec[p.name])) for p in CONT_PARAMS]
    row += [float(p.index(rec[p.name])) for p in CAT_PARAMS]
    return row


def snap_candidate(model_row) -> dict:
    """A candidate row in model space -> snapped raw Firestore dict (on-grid)."""
    raw: dict = {}
    for j, p in enumerate(CONT_PARAMS):
        raw[p.name] = p.snap(p.denormalize(float(model_row[j])))
    for k, p in enumerate(CAT_PARAMS):
        raw[p.name] = p.value(model_row[D_CONT + k])
    return canonicalize(raw)


def model_bounds() -> np.ndarray:
    """[2, D] bounds for acqf optimisation: [0,1] for cont dims, [0, K-1] for cat."""
    lo = np.zeros(D, dtype=float)
    hi = np.ones(D, dtype=float)
    for k, p in enumerate(CAT_PARAMS):
        hi[D_CONT + k] = len(p.categories) - 1
    return np.stack([lo, hi])


def fixed_features_list() -> list[dict]:
    """Every categorical combination, as fixed_features dicts for optimize_acqf_mixed.

    e.g. one `pattern` with 4 levels -> [{6: 0}, {6: 1}, {6: 2}, {6: 3}].
    """
    if D_CAT == 0:
        return []
    ranges = [range(len(p.categories)) for p in CAT_PARAMS]
    combos = []
    for combo in itertools.product(*ranges):
        combos.append({CAT_DIMS[k]: float(v) for k, v in enumerate(combo)})
    return combos


def obs_key(raw: dict) -> tuple:
    """Hashable identity of a configuration, for de-duplication.

    Canonicalized first, so e.g. constant@4Hz and constant@1Hz (the same
    physical stimulus) share one key.
    """
    raw = canonicalize(raw)
    key = []
    for p in CONT_PARAMS:
        key.append(round(float(raw[p.name]), p.round_ndigits))
    for p in CAT_PARAMS:
        key.append(str(raw[p.name]))
    return tuple(key)


# ── Exploration: Sobol draws, snapped onto the JND grid ──────────────────────
_SOBOL_SEED = 15


def sobol_next(obs_count: int) -> dict:
    """The `obs_count`-th Sobol point over the continuous dims, snapped to grid,
    with a deterministic categorical choice. Guaranteed feasible."""
    from torch.quasirandom import SobolEngine  # lazy: keep the module torch-free
    for attempt in range(64):
        eng = SobolEngine(dimension=max(D_CONT, 1), scramble=True, seed=_SOBOL_SEED)
        eng.fast_forward(obs_count + attempt)
        u = eng.draw(1).squeeze(0).tolist()
        raw = {p.name: p.snap(p.denormalize(u[j])) for j, p in enumerate(CONT_PARAMS)}
        rng = random.Random(1000 + obs_count + attempt)
        for p in CAT_PARAMS:
            raw[p.name] = rng.choice(p.categories)
        raw = canonicalize(raw)
        if is_feasible(raw):
            return raw
    return raw  # give up on feasibility after 64 tries (shouldn't happen)


def random_feasible(seed: int) -> dict:
    """A random on-grid feasible configuration (dedup fallback)."""
    rng = random.Random(seed)
    for _ in range(256):
        raw = {p.name: round(float(rng.choice(p.levels)), p.round_ndigits)
               for p in CONT_PARAMS}
        for p in CAT_PARAMS:
            raw[p.name] = rng.choice(p.categories)
        raw = canonicalize(raw)
        if is_feasible(raw):
            return raw
    return raw


def _validate() -> None:
    """Fail fast (at import) on a misconfigured search space, with a clear message.

    Cheap insurance: whoever edits the space above gets an immediate, specific
    error instead of a confusing failure deep inside the GP at request time.
    """
    if not (D_CONT or D_CAT):
        raise ValueError("search space is empty — add at least one Continuous/CategoricalParam")
    if len(PARAM_NAMES) != len(set(PARAM_NAMES)):
        raise ValueError(f"duplicate parameter (Firestore field) names: {PARAM_NAMES}")
    for p in CONT_PARAMS:
        lv = np.asarray(p.levels, dtype=float)
        if lv.size < 1:
            raise ValueError(f"{p.name!r}: grid is empty")
        if not np.all(np.isfinite(lv)):
            raise ValueError(f"{p.name!r}: grid has non-finite levels {lv.tolist()}")
        if lv.size > 1 and not np.all(np.diff(lv) > 0):
            raise ValueError(f"{p.name!r}: grid must be strictly ascending, got {lv.tolist()}")
    for p in CAT_PARAMS:
        if len(p.categories) < 1:
            raise ValueError(f"{p.name!r}: needs at least one category")
        if len(p.categories) != len(set(p.categories)):
            raise ValueError(f"{p.name!r}: duplicate categories {p.categories}")
    if not OBJECTIVE_FIELDS:
        raise ValueError("OBJECTIVE_FIELDS is empty — need at least one objective")


def describe() -> str:
    lines = ["VAM search space (JND-discretised):"]
    for p in CONT_PARAMS:
        lines.append(
            f"  {p.name:<16} {len(p.levels):>2} levels  "
            f"[{p.lo:g} .. {p.hi:g}]  e.g. {np.round(p.levels[:5], 3).tolist()}"
        )
    for p in CAT_PARAMS:
        lines.append(f"  {p.name:<16} {len(p.categories):>2} categories {p.categories}")
    lines.append(f"  model dim D = {D}  (cont {D_CONT} + cat {D_CAT}); "
                 f"cat_dims = {CAT_DIMS}")
    return "\n".join(lines)


_validate()  # run on import — a bad edit above fails here, loudly and early.


if __name__ == "__main__":
    print(describe())
