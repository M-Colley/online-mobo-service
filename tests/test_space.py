"""
Fast unit tests for the search-space logic in space.py. Run it every time you
edit space.py:

    python tests/test_space.py

Covers the grid builders, snapping/encoding round-trips, dedup keys, the acqf
plumbing, Sobol/random draws, and — the important one — an EXHAUSTIVE check that
every one of the 5,544 reachable designs satisfies the app's live security rules
and that the knob -> design map is injective. That exhaustive pass is what stops
a bad edit here from silently stalling a participant mid-study: an illegal
design is accepted from us (Admin SDK bypasses rules) but makes the APP's own
write illegal, which produces no error on our side at all.

Plain asserts — no pytest needed. The exhaustive pass takes a few seconds.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import space


def test_grid_builders():
    w = space.weber_grid(0.2, 1.0, 0.13)
    assert w[0] == 0.2 and w[-1] <= 1.0 + 1e-9
    ratios = w[1:] / w[:-1]
    assert np.allclose(ratios, 1.13), ratios          # each step = one Weber JND
    assert np.all(np.diff(w) > 0)                       # ascending

    lin = space.linear_grid(0.0, 1.0, 0.2)
    assert np.allclose(lin, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    lg = space.list_grid([4, 1, 3, 2])
    assert np.allclose(lg, [1, 2, 3, 4])                # sorted ascending


def test_grid_builder_guards():
    for bad in (lambda: space.weber_grid(0.0, 1.0, 0.1),   # lo must be > 0
                lambda: space.weber_grid(0.2, 1.0, 0.0),   # jnd must be > 0 (else infinite loop)
                lambda: space.linear_grid(0.0, 1.0, 0.0),  # step must be > 0
                lambda: space.list_grid([])):              # non-empty
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError from {bad}")


def test_snap_and_normalize_roundtrip():
    for p in space.CONT_PARAMS:
        for lvl in p.levels:
            # snapping an on-grid value returns itself
            assert p.snap(float(lvl)) == round(float(lvl), p.round_ndigits)
            # normalize/denormalize is an identity up to rounding
            assert abs(p.denormalize(p.normalize(lvl)) - lvl) < 1e-9
        # a value between two levels snaps to the nearer one
        if len(p.levels) >= 2:
            mid = (p.levels[0] + p.levels[1]) / 2 - 1e-6
            assert p.snap(mid) == round(float(p.levels[0]), p.round_ndigits)


def test_knob_axes_are_uniform_in_model_space():
    # The whole point of integer JND knobs: one JND is a CONSTANT distance to the
    # GP's stationary kernel. A geometric grid of raw values is not (the old
    # space's intensity grid spanned 0.033 to 0.145 of the [0,1] axis per step).
    for p in space.CONT_PARAMS:
        us = [p.normalize(float(lv)) for lv in p.levels]
        steps = np.diff(us)
        assert np.allclose(steps, steps[0]), (p.name, steps)


def test_encode_decode_roundtrip():
    raw = {p.name: round(float(p.levels[len(p.levels) // 2]), p.round_ndigits)
           for p in space.CONT_PARAMS}
    for p in space.CAT_PARAMS:
        raw[p.name] = p.categories[-1]
    row = space.to_model_row(raw)
    assert len(row) == space.D
    back = space.snap_candidate(row)
    assert back == raw, (back, raw)


def test_anchor_reproduces_the_seed_design():
    # The app team's seed is the origin of the search space AND the study's
    # baseline arm; if this drifts, every knob vector means something else.
    assert space.expand(space.anchor()) == space.SEED_DESIGN
    assert space.rules_valid_haptics(space.SEED_DESIGN)
    assert set(space.SEED_DESIGN) == set(space.CUES) and len(space.CUES) == 14
    for burst in space.SEED_DESIGN.values():
        assert set(burst) == set(space.BURST_KEYS)


def test_every_reachable_design_is_legal_and_unique():
    """EXHAUSTIVE: all 5,544 knob settings x 14 cues against the live rules."""
    seen: dict = {}
    n = 0
    for knobs in space.all_knob_settings():
        haptics = space.expand(knobs)
        n += 1
        assert space.rules_valid_haptics(haptics), (knobs, haptics)
        assert set(haptics) == set(space.CUES)
        for cue, burst in haptics.items():
            assert set(burst) == set(space.BURST_KEYS), (cue, burst)
            # `pulseCount is int` in the rules — a float 2.0 is REJECTED there,
            # and Firestore stores a Python float as a double.
            assert type(burst["pulseCount"]) is int, (cue, burst)
            assert burst["pulseCount"] <= burst["onDuration"] * space.MAX_PULSE_RATE_HZ
        key = space._design_key(haptics)
        assert key not in seen, f"collision: {knobs} and {seen[key]} render the same design"
        seen[key] = knobs
    expected = 1
    for p in space.CONT_PARAMS:
        expected *= len(p.levels)
    assert n == expected == 5544, (n, expected)


def test_knobs_recover_from_a_rendered_design():
    # Needed because we train on what the app ECHOED, not on what we proposed.
    for knobs in (space.anchor(),
                  {"gainIntensity": -6.0, "gainSharpness": 3.0,
                   "gainRate": -8.0, "gainBurstLength": 3.0},
                  {"gainIntensity": 2.0, "gainSharpness": -3.0,
                   "gainRate": 2.0, "gainBurstLength": -4.0}):
        assert space.knobs_from_haptics(space.expand(knobs)) == knobs
    # a design the knobs cannot produce is reported as unreachable, not guessed
    tampered = {c: dict(b) for c, b in space.SEED_DESIGN.items()}
    tampered["turn"]["intensity"] = 0.123
    assert space.knobs_from_haptics(tampered) is None


def test_rules_mirror_rejects_what_the_app_rejects():
    good = space.SEED_DESIGN["onRoute"]
    assert space.rules_valid_burst(good)
    bad_cases = [
        dict(good, pulseCount=120.0),                  # float, not int64
        dict(good, pulseCount=True),                   # bool is an int subclass
        dict(good, pulseCount=0),                      # below the rules minimum
        dict(good, pulseCount=121),                    # above the rules maximum
        dict(good, onDuration=0.005),                  # below 0.01 s
        dict(good, offDuration=2.5),                   # above 2.0 s
        dict(good, intensity=1.5),                     # outside [0,1]
        dict(good, sharpness=-0.1),                    # outside [0,1]
        dict(good, onDuration=0.5),                    # 120 pulses in 0.5 s > 120 Hz
        {**good, "extra": 1},                          # sixth key: hasOnly() fails
        {k: v for k, v in good.items() if k != "sharpness"},   # missing key
    ]
    for burst in bad_cases:
        assert not space.rules_valid_burst(burst), burst
    # a haptics map missing a cue, or carrying an unknown one, must fail too
    assert not space.rules_valid_haptics({c: space.SEED_DESIGN[c] for c in list(space.CUES)[:13]})
    assert not space.rules_valid_haptics({**space.SEED_DESIGN, "unknownCue": good})


def test_pulse_count_respects_both_caps():
    # the rate cap, evaluated in the rules' own arithmetic
    assert space.pulse_count(0.01, 120.0) == 1          # floor(1.2) -> 1
    assert space.pulse_count(1.0, 120.0) == 120
    assert space.pulse_count(0.04, 120.0) <= 0.04 * 120  # round() would give 5 > 4.8
    # the absolute cap binds above 1 s, where rate x on would exceed 120
    assert space.pulse_count(2.0, 120.0) == 120
    # never below one pulse
    assert space.pulse_count(0.01, 0.5) == 1
    assert type(space.pulse_count(0.5, 60.0)) is int


def test_sobol_covers_every_level_uniformly():
    # Nearest-level snapping would give the two END levels half-width bins, so the
    # corners of the grid would be drawn half as often as the interior. With only
    # ten exploration draws in the whole study that is a real coverage loss.
    from collections import Counter
    # 32 draws per level is enough: a scrambled Sobol net fills equal-width bins
    # almost exactly, and the old nearest-level snap gave the end levels HALF
    # the interior count — a 2x ratio this bound catches with room to spare.
    for p in space.CONT_PARAMS:
        n_draws = 32 * len(p.levels)
        counts = Counter(space.sobol_next(i)[p.name] for i in range(n_draws))
        assert len(counts) == len(p.levels), (p.name, sorted(counts))
        lo, hi = min(counts.values()), max(counts.values())
        assert hi / lo < 1.35, (p.name, dict(sorted(counts.items())))


def test_random_feasible_respects_what_was_already_tested():
    # Every point of this grid is feasible, so without a novelty test the loop
    # returns its first draw forever and the dedup fallback is a no-op.
    first = space.random_feasible(seed=5)
    assert space.random_feasible(seed=5) == first, "should stay deterministic"
    alt = space.random_feasible(seed=5, exclude={space.obs_key(first)})
    assert space.obs_key(alt) != space.obs_key(first), "fallback returned a tested design"
    # excluding (nearly) everything still returns something on-grid rather than hanging
    everything = {space.obs_key(k) for k in space.all_knob_settings()}
    assert space.on_grid(space.random_feasible(seed=1, exclude=everything))


def test_round_number_coercion():
    # the rules treat 5.0 == 5 as true, so we do too; nothing else is a round
    assert space.round_number(5) == 5
    assert space.round_number(5.0) == 5
    for bad in (5.5, True, False, "5", None, float("nan")):
        assert space.round_number(bad) is None, bad


def test_decode_knobs_is_the_single_source_of_truth():
    ours = {"haptics": space.expand({**space.anchor(), "gainRate": -3.0}),
            "mobo": {"knobs": {**{k: 0 for k in space.KNOB_NAMES}, "gainRate": -3},
                     "spaceVersion": space.SPACE_VERSION}}
    knobs, how = space.decode_knobs(ours)
    assert how == "echo" and knobs == {**space.anchor(), "gainRate": -3.0}
    # the app team's hand-seeded doc: haptics only, no mobo map -> still decodes
    seeded = {"haptics": {c: dict(b) for c, b in space.SEED_DESIGN.items()}}
    assert space.decode_knobs(seeded) == (space.anchor(), "echo")
    # a result echoing an off-grid design is refused with a reason
    clamped = {"haptics": {c: dict(b) for c, b in space.SEED_DESIGN.items()}}
    clamped["haptics"]["turn"]["intensity"] = 0.123
    knobs, why = space.decode_knobs(clamped, ours)
    assert knobs is None and "off the knob grid" in why
    # a proposal from another space version excludes the round, echo or not
    stale = dict(ours, mobo={**ours["mobo"], "spaceVersion": "v2-old"})
    knobs, why = space.decode_knobs(seeded, stale)
    assert knobs is None and "v2-old" in why
    # no haptics -> stored knobs; off-grid stored knobs -> refused
    assert space.decode_knobs({"mobo": ours["mobo"]})[1] == "stored"
    assert space.decode_knobs({"mobo": {"knobs": {**ours["mobo"]["knobs"], "gainRate": 99}}})[0] is None


def test_on_grid_guard():
    assert space.on_grid(space.anchor())
    assert space.on_grid(space.sobol_next(3))
    for bad in ({**space.anchor(), "gainRate": 99.0},        # outside the range
                {**space.anchor(), "gainRate": 0.5},         # between levels
                {**space.anchor(), "gainRate": "x"},         # not a number
                {k: v for k, v in space.anchor().items() if k != "gainRate"}):  # missing
        assert not space.on_grid(bad), bad


def test_obs_key_dedup():
    raw = space.random_feasible(seed=1)
    assert space.obs_key(raw) == space.obs_key(dict(raw))       # stable
    keys = {space.obs_key(space.sobol_next(i)) for i in range(8)}
    assert len(keys) >= 1                                        # hashable / usable in a set
    # integer knobs mean the key is exact — no float-rounding near-misses
    assert space.obs_key({**space.anchor(), "gainRate": -0.0}) == space.obs_key(space.anchor())


def test_model_bounds_and_fixed_features():
    b = space.model_bounds()
    assert b.shape == (2, space.D)
    assert np.all(b[0] == 0.0)
    for k, p in enumerate(space.CAT_PARAMS):
        assert b[1, space.D_CONT + k] == len(p.categories) - 1   # cat upper bound = K-1
    ff = space.fixed_features_list()
    expected = int(np.prod([len(p.categories) for p in space.CAT_PARAMS])) if space.CAT_PARAMS else 0
    assert len(ff) == expected


def test_sobol_and_random_are_on_grid_and_feasible():
    # space.on_grid is the guard the service uses. (A local copy with
    # atol=10**-round_ndigits accepted half-step values on integer knobs.)
    for i in range(12):
        for raw in (space.sobol_next(i), space.random_feasible(seed=i)):
            assert space.on_grid(raw), raw
            assert space.is_feasible(raw)
    # Sobol is deterministic
    assert space.sobol_next(3) == space.sobol_next(3)


def test_budget_fits_the_rules_round_cap():
    # anchor + 2*(D+1) Sobol rounds must leave room for at least one GP round.
    assert space.N_SOBOL_DEFAULT == space.N_ANCHOR + 2 * (space.D + 1)
    assert space.N_SOBOL_DEFAULT < space.MAX_ROUND_NUMBER
    assert space.MAX_ROUND_NUMBER == 18


def test_validator_catches_bad_space():
    # duplicate names
    saved = space.CONT_PARAMS
    try:
        space.CONT_PARAMS = saved + [space.ContinuousParam(saved[0].name, space.list_grid([1, 2]))]
        # rebuild PARAM_NAMES the way the module derives it
        space.PARAM_NAMES = [p.name for p in space.CONT_PARAMS] + [p.name for p in space.CAT_PARAMS]
        try:
            space._validate()
        except ValueError:
            pass
        else:
            raise AssertionError("validator should reject duplicate names")
    finally:
        space.CONT_PARAMS = saved
        space.PARAM_NAMES = [p.name for p in space.CONT_PARAMS] + [p.name for p in space.CAT_PARAMS]

    # a seed design that violates the app's rules must not import
    saved_seed = space.SEED_DESIGN
    try:
        space.SEED_DESIGN = {c: dict(b) for c, b in saved_seed.items()}
        space.SEED_DESIGN["turn"]["pulseCount"] = 999
        try:
            space._validate()
        except ValueError:
            pass
        else:
            raise AssertionError("validator should reject an illegal SEED_DESIGN")
    finally:
        space.SEED_DESIGN = saved_seed


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nPASS: {len(tests)} search-space unit tests")


if __name__ == "__main__":
    main()
