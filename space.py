"""
MOBO search space.

────────────────────────────────────────────────────────────────────────────
THIS IS THE ONLY FILE THAT *DEFINES* THE STUDY.
`main.py` / `optimizer_core.py` read the search space from here. Changing what
is optimised is an edit to this file; changing the WIRE FORMAT (how a design is
written to Firestore) also lives here, in expand() / SEED_DESIGN.
────────────────────────────────────────────────────────────────────────────

What the app actually expects (contract, ruleset 42691aae, released 2026-09-08)
------------------------------------------------------------------------------
One candidate is NOT five flat fields. It is a nested `haptics` map of
**14 navigation cues x 5 burst values = 70 numbers**, written on a
`parameterValues` doc that also declares `schemaVersion: 2` and a `candidateId`.
The app's security rules validate the echo of that map when the app writes its
own `interventionResults` doc, with exact-key checks per burst, an `is int`
check on `pulseCount`, the cross-constraint `pulseCount <= onDuration * 120`,
the literal string 'exploration' for `phase`, and a hard `1 <= roundNumber <= 18`.
A design that violates any of those is accepted from us (the Admin SDK bypasses
the rules) but makes the APP's write illegal — the participant then stalls with
no error visible on our side. rules_valid_haptics() below is a literal mirror of
that predicate and is checked before every write.

Burst semantics (confirmed by the app team, 2026-09-11):
  onDuration  — length of the WHOLE burst, in seconds (0.01 .. 2.0)
  pulseCount  — number of pulses INSIDE that burst (int, 1 .. 120), so the
                pulse rate is pulseCount / onDuration Hz, capped at 120 Hz —
                which is exactly what the rules' `pulseCount <= onDuration*120`
                encodes. All 14 seed cues sit at exactly 40, 60 or 120 Hz.
  offDuration — silence before the burst repeats, in seconds (0.01 .. 2.0)
  intensity, sharpness — Core Haptics CHHapticEvent parameters, 0 .. 1

Why the optimizer does not search those 70 numbers
--------------------------------------------------
The rules cap a participant at 18 rounds and one round yields ONE
(subjectiveScore, objectiveScore) pair for the whole session. 70 unknowns from
18 aggregate observations is under-determined, not merely sample-starved: a GP
over 70 inputs fits 70 lengthscales from 18 points, so the posterior is the
prior and qLogNEHVI degenerates to quasi-random search.

So we optimise **four integer JND offsets from the app team's own seed design**
(SEED_DESIGN below, copied verbatim from the live parameterValues doc). Knobs
all zero == the seed, so the expert design is reachable exactly and is the worst
case, not a lucky draw. expand() turns a knob vector into all 70 numbers.

Intensity, rate and burst length are scaled MULTIPLICATIVELY, which preserves the
contrasts the app team designed between cues at every setting — re-gridding
absolute per-cue values does not (at low settings the 14 cues would collapse onto
one intensity and the navigation signal would be destroyed).

Sharpness is the exception and is ADDITIVE, because it is a bounded 0-1 timbre
control with seed values at 0.15-1.0 that a multiplicative step cannot move
sensibly. Additive means it CLIPS, and clipping does collapse contrast: at
gainSharpness = -3 ten of the fourteen cues hit 0.0 and the seed's five distinct
sharpness values become two. That is a real (documented, measured) limitation of
this knob, not of the other three — and it is why the sharpness step is flagged
as a placeholder pending the pilot rather than treated as settled.

How JNDs are encoded now
------------------------
Each knob's unit IS one JND, so the grid is the integers. This removes a subtle
defect of the old space: ContinuousParam.normalize() is LINEAR, so a geometric
(Weber) grid of raw values was NOT equally spaced in model coordinates and a
stationary kernel saw wildly different distances for what was nominally the same
JND step (the old intensity grid spanned 0.033 to 0.145 of the axis per step).
On an integer axis every step WITHIN a knob is the same distance. Across knobs it
is not — normalize() rescales each knob's own range to [0,1], so one
gainIntensity step is 1/8 of its axis while one gainRate step is 1/10 of its
own — that part is left to the GP's per-dimension (ARD) lengthscales, which is
what they are for.

  gainIntensity    x 1.13 per step  (see the knob comment: this is ~0.67 of a
                                     published amplitude JND, i.e. deliberately
                                     FINER than one JND, not equal to one)
  gainSharpness    + 0.20 per step  (placeholder — no psychophysical basis yet,
                                     and additive, so it clips: see above)
  gainRate         x 1.20 per step  (Weber fraction for pulse rate)
  gainBurstLength  x 1.25 per step  (placeholder — see the comment on the knob)

Two knobs are not perceptually independent at the ceiling: the rules cap
pulseCount at 120 absolutely, so for the four cues seeded at 120 Hz the rate knob
cannot go up at all, and lengthening the burst lowers their realised rate. Expect
gainRate and gainBurstLength to trade off for those cues.

Hard constraints
----------------
Every reachable design is a bounded offset from an expert baseline, so absurd
designs are out of RANGE rather than filtered by a predicate. is_feasible() is
kept as the rules mirror: it catches a bad edit to this file, not a bad
proposal.
"""

from __future__ import annotations

import functools
import itertools
import math
import random
from dataclasses import dataclass

import numpy as np
# NB: torch is imported lazily inside sobol_next() so this module (the search-space
# definition) can be imported / unit-tested without torch installed.

# ── Objective (Y) field names — read from interventionResults, both maximised ──
OBJECTIVE_FIELDS = ["subjectiveScore", "objectiveScore"]

# ── Wire-format constants (the app's contract — do not change unilaterally) ──
SCHEMA_VERSION = 2                 # rules: interventionResults.schemaVersion == 2
HAPTIC_MODE = "burst"              # rules: interventionResults.hapticMode == 'burst'
WIRE_PHASE = "exploration"         # rules PIN the result doc's phase to this literal
MAX_ROUND_NUMBER = 18              # rules: 1 <= roundNumber <= 18
SPACE_VERSION = "v3-knobs-2026-09"  # stamped on every proposal; refuse to mix versions

# The 14 navigation cues, exactly the key set validHaptics() requires.
CUES = (
    "start", "onRoute", "offRoute", "onRouteIntersection", "offRouteIntersection",
    "landmark", "end", "street", "onRouteSidewalk", "offRouteSidewalk",
    "onRouteCrosswalk", "offRouteCrosswalk", "turn", "intersectionCenter",
)
# The 5 burst fields, exactly the key set validBurst() requires (hasAll + hasOnly).
BURST_KEYS = ("intensity", "sharpness", "pulseCount", "onDuration", "offDuration")

# Rules bounds, mirrored literally from validBurst().
INTENSITY_MIN, INTENSITY_MAX = 0.0, 1.0
SHARPNESS_MIN, SHARPNESS_MAX = 0.0, 1.0
PULSE_COUNT_MIN, PULSE_COUNT_MAX = 1, 120
DURATION_MIN, DURATION_MAX = 0.01, 2.0
MAX_PULSE_RATE_HZ = 120.0          # rules: pulseCount <= onDuration * 120

# ── The app team's seed design ───────────────────────────────────────────────
# Copied verbatim from parameterValues/XmNWlxIjB4sGrle1ehBk (pid 101,
# candidateId "101-round-01-seed-20260909T171125Z", written 2026-09-09).
# This is the study's expert baseline AND the origin of the search space:
# expand() with all knobs zero reproduces it exactly (asserted in _validate()).
SEED_DESIGN: dict[str, dict] = {
    "start":                {"intensity": 0.75, "sharpness": 1.00, "pulseCount": 10,  "onDuration": 0.25, "offDuration": 0.50},
    "onRoute":              {"intensity": 1.00, "sharpness": 0.50, "pulseCount": 120, "onDuration": 1.00, "offDuration": 0.01},
    "offRoute":             {"intensity": 0.25, "sharpness": 0.25, "pulseCount": 120, "onDuration": 1.00, "offDuration": 0.01},
    "onRouteIntersection":  {"intensity": 0.75, "sharpness": 0.25, "pulseCount": 15,  "onDuration": 0.25, "offDuration": 0.05},
    "offRouteIntersection": {"intensity": 0.75, "sharpness": 0.25, "pulseCount": 120, "onDuration": 1.00, "offDuration": 0.01},
    "landmark":             {"intensity": 1.00, "sharpness": 0.15, "pulseCount": 60,  "onDuration": 1.00, "offDuration": 0.01},
    "end":                  {"intensity": 0.75, "sharpness": 1.00, "pulseCount": 120, "onDuration": 1.00, "offDuration": 0.01},
    "street":               {"intensity": 0.33, "sharpness": 0.33, "pulseCount": 60,  "onDuration": 1.00, "offDuration": 0.15},
    "onRouteSidewalk":      {"intensity": 0.75, "sharpness": 1.00, "pulseCount": 60,  "onDuration": 1.00, "offDuration": 0.01},
    "offRouteSidewalk":     {"intensity": 0.25, "sharpness": 0.25, "pulseCount": 60,  "onDuration": 1.00, "offDuration": 0.01},
    "onRouteCrosswalk":     {"intensity": 0.75, "sharpness": 1.00, "pulseCount": 60,  "onDuration": 1.00, "offDuration": 0.01},
    "offRouteCrosswalk":    {"intensity": 0.25, "sharpness": 0.25, "pulseCount": 60,  "onDuration": 1.00, "offDuration": 0.01},
    "turn":                 {"intensity": 0.25, "sharpness": 0.25, "pulseCount": 60,  "onDuration": 1.00, "offDuration": 0.01},
    "intersectionCenter":   {"intensity": 0.25, "sharpness": 0.25, "pulseCount": 60,  "onDuration": 1.00, "offDuration": 0.01},
}


# ── Grid builders ────────────────────────────────────────────────────────────
def weber_grid(lo: float, hi: float, jnd_fraction: float) -> np.ndarray:
    """Geometric grid for Weber's-law parameters.

    Consecutive levels differ by `jnd_fraction`:
    level_{k+1} = level_k * (1 + jnd_fraction).
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
    """Additive grid for parameters whose JND is an absolute amount."""
    if step <= 0:
        raise ValueError(f"linear_grid needs step > 0 (got {step})")
    if hi < lo:
        raise ValueError(f"linear_grid needs hi >= lo (got lo={lo}, hi={hi})")
    n = int(np.floor((hi - lo) / step + 1e-9))
    return lo + step * np.arange(n + 1, dtype=float)


def list_grid(values) -> np.ndarray:
    """Explicit level list — used here for the integer JND knob axes."""
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("list_grid needs at least one value")
    return np.sort(arr)  # keep ascending regardless of input order


# ── Parameter descriptors ────────────────────────────────────────────────────
@dataclass
class ContinuousParam:
    name: str                 # knob name (NOT a Firestore field — see the header)
    levels: np.ndarray        # JND-spaced grid, ascending.
    round_ndigits: int = 3    # rounding when the value is recorded.

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
    name: str
    categories: list

    def index(self, value) -> int:
        """Category value (str) OR already-numeric code -> integer index."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(round(value))
        return self.categories.index(value)

    def value(self, index) -> object:
        return self.categories[int(round(float(index)))]


# ═══════════════════════════════════════════════════════════════════════════
#  THE SEARCH SPACE — four integer JND knobs, offsets from SEED_DESIGN.
#  Knobs all zero == the seed design. One step == one JND (see the header).
# ═══════════════════════════════════════════════════════════════════════════
INTENSITY_STEP = 1.13     # multiplicative, per gainIntensity step
SHARPNESS_STEP = 0.20     # additive, per gainSharpness step
RATE_STEP = 1.20          # multiplicative, per gainRate step
BURST_LENGTH_STEP = 1.25  # multiplicative, per gainBurstLength step
MIN_PULSE_RATE_HZ = 0.5   # optimizer-side floor; below this a "burst" is one pulse

CONT_PARAMS: list[ContinuousParam] = [
    # Amplitude, in intensity JNDs. 1.13 is the figure this repo has always used,
    # but published vibrotactile AMPLITUDE Weber fractions are nearer 0.20 (e.g.
    # Pardo et al. 2022, median 20 % [IQR 7 %]). So one step here is only ~0.67 of
    # a real JND and the whole 9-level range spans ~5.4 discriminable steps, not 8:
    # the optimizer can spend rounds separating stimuli the participant cannot
    # tell apart, which is expensive in an 18-round budget. Kept conservative
    # (finer, never coarser, than the truth) until the pilot measures it on THIS
    # device and body site; widening the step to ~1.20 would buy ~3 more usable
    # levels.  # TODO confirm in the pilot
    # Range -6..+2: the seed already puts four cues at intensity 1.0, so upward
    # headroom is small and the useful direction is down.
    ContinuousParam("gainIntensity", list_grid(range(-6, 3)), round_ndigits=0),

    # Timbre, additive in the app's 0-1 sharpness units. NO psychophysical basis
    # for the 0.20 step — inherited placeholder.  # TODO set after the pilot
    # NB four seed cues sit at sharpness 1.0, so ~1/3 of cue-cells clip upward;
    # the app team could give this knob headroom by seeding those nearer 0.85.
    ContinuousParam("gainSharpness", list_grid(range(-3, 4)), round_ndigits=0),

    # Pulse rate (pulseCount / onDuration), in rate JNDs. Weber 1.20 is the
    # figure this repo used for the old `interval` parameter. Range -8..+2: the
    # seed rates are 40/60/120 Hz and 120 Hz is the rules' hard ceiling, so
    # again the useful direction is down (towards countable pulses).
    ContinuousParam("gainRate", list_grid(range(-8, 3)), round_ndigits=0),

    # Burst length (onDuration), multiplicative. The old absolute 0.240 s step
    # came from a JND measured at a single 500 ms standard (~48 %, a RATIO) and
    # is meaningless on the 0.01-2.0 s range the app now accepts, so a
    # multiplicative step is used instead. 1.25 is a placeholder.
    # TODO confirm the duration Weber fraction with the JND owner before the run
    ContinuousParam("gainBurstLength", list_grid(range(-4, 4)), round_ndigits=0),
]

# No nominal parameters: `pattern` is gone from the contract. A sustained cue is
# now just a high rate with the minimum gap (the seed's own `onRoute`).
CAT_PARAMS: list[CategoricalParam] = []

# ── Derived layout of the model input vector ─────────────────────────────────
D_CONT = len(CONT_PARAMS)
D_CAT = len(CAT_PARAMS)
D = D_CONT + D_CAT                       # model input dimensionality
CAT_DIMS = list(range(D_CONT, D))        # categorical column indices for MixedSingleTaskGP
PARAM_NAMES = [p.name for p in CONT_PARAMS] + [p.name for p in CAT_PARAMS]
KNOB_NAMES = PARAM_NAMES                 # readable alias: these are knobs, not fields

# ── Round budget (the ONE place these numbers are defined) ───────────────────
# Round 1 is the anchor (the seed design, a fixed within-participant control).
# The next 2*(D+1) rounds are Sobol draws — deliberately one more than the study
# memo's "2n+1" as a conservative buffer (decision 2026-07-21). Everything up to
# MAX_ROUND_NUMBER is model-driven. main.py reads these; it can be overridden
# there via env vars, but the default is derived here so simulate.py, the tests
# and main.py cannot drift apart.
N_ANCHOR = 1
N_SOBOL_DRAWS = 2 * (D + 1)
N_SOBOL_DEFAULT = N_ANCHOR + N_SOBOL_DRAWS   # rounds 1..N_SOBOL_DEFAULT are not model-driven


# ── Expansion: knob vector  ->  the 70 numbers the app renders ───────────────
def pulse_count(on_duration: float, rate_hz: float) -> int:
    """Pulses in a burst of `on_duration` seconds at `rate_hz`, as a real int.

    Enforces BOTH rules limits: the absolute cap (<= 120) and the rate cap
    (<= onDuration * 120). `on_duration` must ALREADY be rounded to what will be
    written, because the rule is evaluated on the written value. int(round(...))
    can round UP past the rate cap (on=0.04, rate=120 -> round(4.8)=5 > 4.8),
    hence the loop; it walks down at most one or two levels.
    """
    pc = int(round(float(rate_hz) * float(on_duration)))
    pc = max(PULSE_COUNT_MIN, min(pc, PULSE_COUNT_MAX))
    while pc > PULSE_COUNT_MIN and pc > on_duration * MAX_PULSE_RATE_HZ:
        pc -= 1
    return pc


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def expand(knobs: dict) -> dict:
    """Knob vector -> the full 14-cue `haptics` map the app renders.

    Every cue is scaled MULTIPLICATIVELY from its seed value, so the contrasts
    the app team designed between cues survive at every knob setting. Values are
    rounded to what will actually be written BEFORE the rate cap is applied, so
    the numbers we write are the numbers the security rule is evaluated on.
    """
    g_int = int(round(float(knobs["gainIntensity"])))
    g_shp = int(round(float(knobs["gainSharpness"])))
    g_rate = int(round(float(knobs["gainRate"])))
    g_len = int(round(float(knobs["gainBurstLength"])))

    haptics: dict[str, dict] = {}
    for cue, seed in SEED_DESIGN.items():
        rate_seed = seed["pulseCount"] / seed["onDuration"]
        on = round(_clip(seed["onDuration"] * BURST_LENGTH_STEP ** g_len,
                         DURATION_MIN, DURATION_MAX), 3)
        rate = _clip(rate_seed * RATE_STEP ** g_rate, MIN_PULSE_RATE_HZ, MAX_PULSE_RATE_HZ)
        haptics[cue] = {
            "intensity": round(_clip(seed["intensity"] * INTENSITY_STEP ** g_int,
                                     INTENSITY_MIN, INTENSITY_MAX), 3),
            "sharpness": round(_clip(seed["sharpness"] + SHARPNESS_STEP * g_shp,
                                     SHARPNESS_MIN, SHARPNESS_MAX), 3),
            "pulseCount": pulse_count(on, rate),
            "onDuration": on,
            "offDuration": round(seed["offDuration"], 3),
        }
    return haptics


# ── Rules mirror: validBurst() / validHaptics() from ruleset 42691aae ────────
# A LITERAL mirror — no epsilon. The rule is a hard predicate evaluated on the
# written doubles; slack here would let through a design the app cannot echo.
def rules_valid_burst(burst) -> bool:
    """True iff `burst` would pass the app's validBurst() security rule."""
    if not isinstance(burst, dict) or set(burst) != set(BURST_KEYS):
        return False
    pc = burst["pulseCount"]
    # rules: `pulseCount is int`. bool is an int subclass in Python but encodes
    # to a Firestore boolean, so it must be rejected here.
    if isinstance(pc, bool) or not isinstance(pc, int):
        return False
    if not (PULSE_COUNT_MIN <= pc <= PULSE_COUNT_MAX):
        return False
    for key, lo, hi in (("intensity", INTENSITY_MIN, INTENSITY_MAX),
                        ("sharpness", SHARPNESS_MIN, SHARPNESS_MAX),
                        ("onDuration", DURATION_MIN, DURATION_MAX),
                        ("offDuration", DURATION_MIN, DURATION_MAX)):
        v = burst[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        if not math.isfinite(float(v)) or not (lo <= float(v) <= hi):
            return False
    return pc <= burst["onDuration"] * MAX_PULSE_RATE_HZ


def rules_valid_haptics(haptics) -> bool:
    """True iff `haptics` would pass the app's validHaptics() security rule."""
    if not isinstance(haptics, dict) or set(haptics) != set(CUES):
        return False
    return all(rules_valid_burst(b) for b in haptics.values())


# ── Inverse: a rendered design -> the knob vector that produced it ───────────
_INVERSE_TABLE: dict | None = None


def _inverse_table() -> dict:
    """design-key -> knob dict for every point of the grid; built once, at import.

    ~0.3 s. Built eagerly from _validate() so the cost lands on the cold start
    (which torch already dominates) and never inside a participant's request.
    """
    global _INVERSE_TABLE
    if _INVERSE_TABLE is None:
        _INVERSE_TABLE = {_design_key(expand(k)): k for k in all_knob_settings()}
    return _INVERSE_TABLE


def _design_key(haptics: dict) -> tuple:
    return tuple(
        (cue, round(float(haptics[cue]["intensity"]), 3),
         round(float(haptics[cue]["sharpness"]), 3),
         int(haptics[cue]["pulseCount"]),
         round(float(haptics[cue]["onDuration"]), 3),
         round(float(haptics[cue]["offDuration"]), 3))
        for cue in CUES
    )


def all_knob_settings():
    """Every point of the knob grid, as raw dicts (5,544 of them)."""
    for combo in itertools.product(*[p.levels for p in CONT_PARAMS]):
        yield {p.name: round(float(v), p.round_ndigits)
               for p, v in zip(CONT_PARAMS, combo)}


def knobs_from_haptics(haptics: dict) -> dict | None:
    """Recover the knob vector from a rendered design, or None if unreachable.

    The expansion is injective over the grid (asserted in the test suite), so
    this is an exact lookup. Used to train on what the app ECHOED rather than on
    what we proposed, whenever the two differ.
    """
    try:
        key = _design_key(haptics)
    except (KeyError, TypeError, ValueError):
        return None
    hit = _inverse_table().get(key)
    return dict(hit) if hit is not None else None


# ── Decoding Firestore documents: THE one place that answers "which knob vector
#    is behind this doc". main.py, the exporter and the inspector all use it, so
#    they cannot disagree about which rounds count.
def round_number(step) -> int | None:
    """phaseStep as an int, or None if it is not a whole number.

    The rules pin `phaseStep is int`, but an Admin-SDK or QA script can write 5.0;
    the rules themselves treat 5.0 == 5 as true, so we do too. A bool is not a
    round. Anything non-integral is rejected so it can be skipped as a whole,
    rather than half-counted (trained on but not counted as a round — the stall).
    """
    if isinstance(step, bool):
        return None
    if isinstance(step, int):
        return step
    if isinstance(step, float) and step.is_integer():
        return int(step)
    return None


def knobs_from_stored(stored) -> dict | None:
    """A `mobo.knobs` map -> on-grid float knob dict, or None."""
    if not isinstance(stored, dict) or not all(k in stored for k in KNOB_NAMES):
        return None
    try:
        knobs = {k: float(stored[k]) for k in KNOB_NAMES}
    except (TypeError, ValueError):
        return None
    return knobs if on_grid(knobs) else None


def decode_knobs(doc: dict, proposal: dict | None = None) -> tuple[dict | None, str]:
    """The knob vector a Firestore doc stands for, plus WHY if it does not.

    Works for both kinds of doc:
      * a proposal (parameterValues) — ours carry `mobo.knobs`; the app team's
        hand-seeded one carries only `haptics`, which is inverted instead;
      * a result (interventionResults) with its `proposal` — the ECHOED `haptics`
        is authoritative (it is what the participant felt), the proposal only
        supplies the spaceVersion gate and a fallback when the echo is absent.

    Returns (knobs, "echo" | "stored") or (None, reason). A knob tuple is only
    meaningful under the space that produced it, so a different spaceVersion is
    refused outright — the same integers denote a different stimulus there.
    """
    source = proposal if proposal is not None else doc
    mobo = (source.get("mobo") or {}) if isinstance(source, dict) else {}
    version = mobo.get("spaceVersion")
    if version and version != SPACE_VERSION:
        return None, f"proposed under spaceVersion={version!r}, not {SPACE_VERSION!r}"
    echo = doc.get("haptics") if isinstance(doc, dict) else None
    if isinstance(echo, dict):
        knobs = knobs_from_haptics(echo)
        if knobs is None:
            return None, "haptics map is off the knob grid (clamped or substituted)"
        return knobs, "echo"
    knobs = knobs_from_stored(mobo.get("knobs"))
    if knobs is None:
        return None, "neither a haptics map nor an on-grid mobo.knobs"
    return knobs, "stored"


# ── Canonical-form hook ──────────────────────────────────────────────────────
def canonicalize(raw: dict) -> dict:
    """Identity: the integer knob tuple IS the canonical coordinate.

    The old space needed this because `pattern == "constant"` left `interval`
    and `duration` as dead dimensions. In the knob space there is no dead
    dimension — every knob changes every cue — so there is nothing to pin. Kept
    as the documented hook for a future parameter that does have dead regimes.
    """
    return dict(raw)


# ── Hard-constraint hook ─────────────────────────────────────────────────────
def is_feasible(raw: dict) -> bool:
    """True iff this knob vector renders to a design the app's rules accept.

    Every reachable design is a bounded offset from the expert baseline, so this
    is expected to be True everywhere: it is a guard against a bad edit to THIS
    file (a widened knob range, a changed seed, a pulse_count() regression), not
    a filter on proposals. Keeping it means a broken space fails in the test
    suite instead of silently stalling a participant mid-study.
    """
    return rules_valid_haptics(expand(raw))


def project_feasible(raw: dict) -> dict:
    """No-op: expand() clips into the rules' box, so proposals cannot be infeasible.

    Retained so optimizer_core.choose_next() keeps working unchanged if a future
    space reintroduces a genuine constraint.
    """
    return dict(raw)


# ── Encoding: knob dict  <->  GP model row ───────────────────────────────────
def to_model_row(rec: dict) -> list[float]:
    """A knob dict -> model input row (continuous dims normalised to [0, 1])."""
    rec = canonicalize(rec)
    row = [p.normalize(float(rec[p.name])) for p in CONT_PARAMS]
    row += [float(p.index(rec[p.name])) for p in CAT_PARAMS]
    return row


def snap_candidate(model_row) -> dict:
    """A candidate row in model space -> snapped knob dict (on-grid)."""
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

    Empty here (CAT_PARAMS == []), which routes optimizer_core to the plain
    SingleTaskGP + optimize_acqf path.
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

    Exact: the knobs are integers, so there is no rounding ambiguity.
    """
    raw = canonicalize(raw)
    key = []
    for p in CONT_PARAMS:
        key.append(round(float(raw[p.name]), p.round_ndigits))
    for p in CAT_PARAMS:
        key.append(str(raw[p.name]))
    return tuple(key)


def on_grid(raw: dict) -> bool:
    """True iff every knob in `raw` is exactly one of its grid levels.

    to_model_row() does not clamp, so an off-grid value (a hand-edited doc, a
    proposal from a different spaceVersion) would train the GP outside
    model_bounds() without any error.
    """
    for p in CONT_PARAMS:
        if p.name not in raw:
            return False
        try:
            v = float(raw[p.name])
        except (TypeError, ValueError):
            return False
        # Compare the value AS STORED against the levels AS WRITTEN (snap() rounds
        # them to round_ndigits). Rounding the value first would accept 0.5 as "on"
        # an integer grid; a half-rounding-unit tolerance would do the same.
        allowed = {round(float(lv), p.round_ndigits) for lv in p.levels}
        if not any(abs(v - a) <= 1e-9 for a in allowed):
            return False
    for p in CAT_PARAMS:
        if raw.get(p.name) not in p.categories:
            return False
    return True


def anchor() -> dict:
    """The seed design as a knob vector: all zeros. Round 1 of every participant."""
    return {p.name: 0.0 for p in CONT_PARAMS}


# ── Exploration: Sobol draws, snapped onto the JND grid ──────────────────────
_SOBOL_SEED = 15


def sobol_next(index: int) -> dict:
    """The `index`-th Sobol point over the knob dims, on the grid.

    `index` is the position in the exploration sequence, and the caller must make
    it a function of the ROUND being filled, not of how many results happened to
    be usable — otherwise a run of discarded rounds re-issues one design forever.

    Each uniform draw picks a LEVEL INDEX (equal-width bins) rather than being
    denormalised and snapped to the nearest level: nearest-level snapping gives
    the two end levels half-width bins, so the corners of the grid would be drawn
    half as often as the interior — a real coverage loss when the whole
    exploration phase is ten draws.
    """
    u = _sobol_point(index)
    raw = {}
    for j, p in enumerate(CONT_PARAMS):
        n = len(p.levels)
        k = min(n - 1, int(u[j] * n))
        raw[p.name] = round(float(p.levels[k]), p.round_ndigits)
    rng = random.Random(1000 + index)
    for p in CAT_PARAMS:
        raw[p.name] = rng.choice(p.categories)
    raw = canonicalize(raw)
    if is_feasible(raw):
        return raw
    # Unreachable on this grid (every point is feasible — asserted at import and
    # exhaustively in the tests). If a future space adds a real constraint, hand
    # back a feasible point rather than an infeasible one the caller must reject.
    return random_feasible(seed=index)


@functools.lru_cache(maxsize=None)
def _sobol_point(index: int) -> tuple:
    """The `index`-th scrambled Sobol point in [0,1)^D_CONT, cached per process."""
    from torch.quasirandom import SobolEngine  # lazy: keep the module torch-free
    eng = SobolEngine(dimension=max(D_CONT, 1), scramble=True, seed=_SOBOL_SEED)
    eng.fast_forward(index)
    return tuple(float(x) for x in eng.draw(1).squeeze(0).tolist())


def random_feasible(seed: int, exclude: set | None = None) -> dict:
    """A random on-grid feasible configuration (dedup fallback).

    `exclude` is a set of obs_key()s already tested. Passing it matters: every
    point of this grid is feasible, so without a novelty test the loop would
    always return its very first draw and the "fall back to something fresh"
    call sites would silently re-propose a design the participant already felt.
    """
    exclude = exclude or set()
    rng = random.Random(seed)
    first_feasible = None
    for _ in range(1024):
        cand = {p.name: round(float(rng.choice(p.levels)), p.round_ndigits)
                for p in CONT_PARAMS}
        for p in CAT_PARAMS:
            cand[p.name] = rng.choice(p.categories)
        cand = canonicalize(cand)
        # The cheap test first: the novelty set lookup. is_feasible() expands
        # all 14 cues, so run it only on a candidate we might actually return.
        if obs_key(cand) in exclude:
            continue
        if is_feasible(cand):
            return cand
    # Grid exhausted for novelty (only in tiny spaces): return SOME feasible point.
    for _ in range(1024):
        cand = {p.name: round(float(rng.choice(p.levels)), p.round_ndigits)
                for p in CONT_PARAMS}
        for p in CAT_PARAMS:
            cand[p.name] = rng.choice(p.categories)
        cand = canonicalize(cand)
        if is_feasible(cand):
            return cand
    return first_feasible or anchor()


def _validate() -> None:
    """Fail fast (at import) on a misconfigured search space, with a clear message.

    Deliberately CHEAP — it runs on every Cloud Run cold start. The exhaustive
    checks (all 5,544 designs valid, injective) live in tests/test_space.py.
    """
    if not (D_CONT or D_CAT):
        raise ValueError("search space is empty — add at least one Continuous/CategoricalParam")
    if len(PARAM_NAMES) != len(set(PARAM_NAMES)):
        raise ValueError(f"duplicate knob names: {PARAM_NAMES}")
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

    # The wire format must match the app's contract exactly.
    if set(SEED_DESIGN) != set(CUES):
        raise ValueError(f"SEED_DESIGN cues {sorted(SEED_DESIGN)} != CUES {sorted(CUES)}")
    if not rules_valid_haptics(SEED_DESIGN):
        raise ValueError("SEED_DESIGN itself violates the app's validHaptics() rule")
    if expand(anchor()) != SEED_DESIGN:
        raise ValueError("expand(anchor()) != SEED_DESIGN — the baseline is unreachable")
    # Corners of the knob box: cheap proxy for "the whole grid renders legally".
    for combo in itertools.product(*[(p.lo, p.hi) for p in CONT_PARAMS]):
        raw = {p.name: float(v) for p, v in zip(CONT_PARAMS, combo)}
        if not is_feasible(raw):
            raise ValueError(f"knob corner {raw} renders an illegal design")
    # The budget has to fit the rules' round cap, with the anchor round included.
    if N_SOBOL_DEFAULT >= MAX_ROUND_NUMBER:
        raise ValueError(
            f"budget does not fit: anchor + 2*(D+1) = {N_SOBOL_DEFAULT} non-model rounds "
            f"leave nothing for the GP inside MAX_ROUND_NUMBER = {MAX_ROUND_NUMBER}. "
            f"Drop a knob (D={D})."
        )
    # Build the inverse table now (cold start, not a participant's request) and
    # get the injectivity proof for free: one entry per grid point.
    n_designs = 1
    for p in CONT_PARAMS:
        n_designs *= len(p.levels)
    if len(_inverse_table()) != n_designs:
        raise ValueError(
            f"expand() is not injective: {n_designs} knob settings render only "
            f"{len(_inverse_table())} distinct designs — two knob vectors would be "
            "indistinguishable to the participant AND to knobs_from_haptics()"
        )


def describe() -> str:
    lines = ["Haptic-design search space (integer JND knobs on the seed design):"]
    for p in CONT_PARAMS:
        lines.append(
            f"  {p.name:<16} {len(p.levels):>2} levels  [{p.lo:+g} .. {p.hi:+g}]  (0 = seed)"
        )
    for p in CAT_PARAMS:
        lines.append(f"  {p.name:<16} {len(p.categories):>2} categories {p.categories}")
    n_designs = 1
    for p in CONT_PARAMS:
        n_designs *= len(p.levels)
    lines.append(f"  model dim D = {D}  (cont {D_CONT} + cat {D_CAT}); "
                 f"{n_designs} reachable designs")
    lines.append(f"  wire: {len(CUES)} cues x {len(BURST_KEYS)} burst fields = "
                 f"{len(CUES) * len(BURST_KEYS)} numbers per candidate, "
                 f"schemaVersion {SCHEMA_VERSION}, rounds <= {MAX_ROUND_NUMBER}")
    return "\n".join(lines)


_validate()  # run on import — a bad edit above fails here, loudly and early.


if __name__ == "__main__":
    print(describe())
    print()
    a = expand(anchor())
    print("anchor (knobs = 0) reproduces the app team's seed design, e.g.")
    for cue in ("start", "onRoute", "onRouteIntersection"):
        print(f"  {cue:<20} {a[cue]}")
