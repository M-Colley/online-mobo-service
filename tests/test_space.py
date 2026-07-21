"""
Fast unit tests for the search-space logic in space.py — NO GP fitting, so
this runs in well under a second. Run it every time you edit space.py:

    python tests/test_space.py

Covers the grid builders, snapping/encoding round-trips, dedup keys, the acqf
plumbing (bounds + categorical combinations), Sobol/​random draws, the
canonical-form hook, and the fail-fast validator. Plain asserts — no pytest
needed.
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


def test_encode_decode_roundtrip():
    # Build an on-grid raw config, encode to a model row, snap back — must match.
    raw = {p.name: round(float(p.levels[len(p.levels) // 2]), p.round_ndigits)
           for p in space.CONT_PARAMS}
    for p in space.CAT_PARAMS:
        raw[p.name] = p.categories[-1]
    row = space.to_model_row(raw)
    assert len(row) == space.D
    back = space.snap_candidate(row)
    assert back == raw, (back, raw)


def test_canonicalize_constant_pins_interval():
    # constant = sustained vibration → the app ignores `interval` and stores the
    # minimum (1 Hz), so all constant configs collapse onto interval == 1.0.
    base = {p.name: round(float(p.levels[0]), p.round_ndigits) for p in space.CONT_PARAMS}
    a = dict(base, pattern="constant", interval=4.0)
    b = dict(base, pattern="constant", interval=1.0)
    assert space.canonicalize(a)["interval"] == 1.0
    assert space.obs_key(a) == space.obs_key(b), "same stimulus must share one dedup key"
    # encoding + snapping goes through the canonical form too
    assert space.snap_candidate(space.to_model_row(a))["interval"] == 1.0
    # "puls" keeps its interval untouched
    c = dict(base, pattern="puls", interval=4.0)
    assert space.canonicalize(c)["interval"] == 4.0
    assert space.obs_key(c) != space.obs_key(a)
    # every proposal path emits canonical configs
    for i in range(8):
        for raw in (space.sobol_next(i), space.random_feasible(seed=i)):
            assert raw == space.canonicalize(raw), f"non-canonical proposal {raw}"


def test_puls_duration_interval_constraint():
    # App floors the off-time at 10 ms: a pulse plus that gap must fit 1/interval.
    base = {p.name: round(float(p.levels[0]), p.round_ndigits) for p in space.CONT_PARAMS}
    # the app team's own example: 0.5 s pulses @ ~4 Hz render at ~2 Hz -> blocked
    assert not space.is_feasible(dict(base, pattern="puls", duration=0.51, interval=4.3))
    assert space.is_feasible(dict(base, pattern="puls", duration=0.03, interval=4.3))
    # exact boundary stays feasible: 1 Hz -> budget 0.99, grid level 0.99
    assert space.is_feasible(dict(base, pattern="puls", duration=0.99, interval=1.0))
    assert not space.is_feasible(dict(base, pattern="puls", duration=1.23, interval=1.0))
    # "constant" is exempt (no pulses)
    assert space.is_feasible(dict(base, pattern="constant", duration=19.95, interval=1.0))
    # projection repairs duration only, keeping the rest of the proposal
    bad = dict(base, pattern="puls", duration=12.75, interval=4.3)
    proj = space.project_feasible(bad)
    assert space.is_feasible(proj)
    assert proj["duration"] == 0.03 and proj["interval"] == 4.3
    assert proj["intensity"] == bad["intensity"] and proj["pattern"] == "puls"
    # feasible configs pass through projection untouched
    good = dict(base, pattern="puls", duration=0.27, interval=2.074)
    assert space.project_feasible(good) == good


def test_obs_key_dedup():
    raw = space.random_feasible(seed=1)
    assert space.obs_key(raw) == space.obs_key(dict(raw))       # stable
    keys = {space.obs_key(space.sobol_next(i)) for i in range(8)}
    assert len(keys) >= 1                                        # hashable / usable in a set


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
    def on_grid(raw):
        for p in space.CONT_PARAMS:
            if not np.any(np.isclose(p.levels, raw[p.name], atol=10 ** -p.round_ndigits)):
                return False
        return all(raw[p.name] in p.categories for p in space.CAT_PARAMS)

    for i in range(12):
        for raw in (space.sobol_next(i), space.random_feasible(seed=i)):
            assert on_grid(raw), raw
            assert space.is_feasible(raw)
    # Sobol is deterministic
    assert space.sobol_next(3) == space.sobol_next(3)


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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nPASS: {len(tests)} search-space unit tests")


if __name__ == "__main__":
    main()
