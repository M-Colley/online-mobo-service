"""
Offline smoke test of the FULL Flask service (main.py) — no cloud, no emulator.

    python test_service.py

Drives the real HTTP endpoints through Flask's test client against an in-memory
fake of the exact Firestore API surface main.py uses (FieldFilter where /
order_by / stream / create / set / update / SERVER_TIMESTAMP). Simulates a
complete participant:

    POST /registerUser
    repeat: read parameterValues -> rate -> write interventionResults
            -> POST /updatePolicy
    until studyCompleted

and asserts: every step doc is on-grid & feasible, steps are contiguous,
duplicate /updatePolicy invocations are suppressed, failed attention checks are
excluded from training data, and the study completes at exactly N_TOTAL.
"""

from __future__ import annotations

import os

# Must be set BEFORE importing main so firestore.Client() constructs offline.
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:1")  # never contacted
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "demo-vam-test")

import numpy as np
import torch

import main
import vam_space as space

torch.manual_seed(0)
np.random.seed(0)


# ── In-memory Firestore fake (only the surface main.py touches) ──────────────
class FakeSnapshot:
    def __init__(self, ref, data):
        self.reference = ref
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class FakeDocRef:
    def __init__(self, store: dict, doc_id: str):
        self._store, self.id = store, doc_id

    def get(self):
        return FakeSnapshot(self, self._store.get(self.id))

    def create(self, data: dict):
        if self.id in self._store:
            raise RuntimeError(f"AlreadyExists: {self.id}")
        self._store[self.id] = dict(data)

    def set(self, data: dict, merge: bool = False):
        if merge and self.id in self._store:
            self._store[self.id].update(data)
        else:
            self._store[self.id] = dict(data)

    def update(self, data: dict):
        self._store[self.id].update(data)


class FakeQuery:
    def __init__(self, store: dict, filters=(), order=None):
        self._store, self._filters, self._order = store, list(filters), order

    def where(self, filter):  # noqa: A002 — mirrors the real client kwarg
        return FakeQuery(self._store, self._filters + [filter], self._order)

    def order_by(self, field):
        return FakeQuery(self._store, self._filters, field)

    def stream(self):
        rows = []
        for doc_id, data in self._store.items():
            ok = True
            for f in self._filters:
                fv = data.get(f.field_path)
                if f.op_string == "==" and fv != f.value:
                    ok = False
                    break
            if ok:
                rows.append((doc_id, data))
        if self._order:
            rows.sort(key=lambda kv: kv[1].get(self._order))
        for doc_id, data in rows:
            yield FakeSnapshot(FakeDocRef(self._store, doc_id), data)


class FakeCollection(FakeQuery):
    def document(self, doc_id: str) -> FakeDocRef:
        return FakeDocRef(self._store, doc_id)


class FakeFirestore:
    def __init__(self):
        self._collections: dict[str, dict] = {}

    def collection(self, name: str) -> FakeCollection:
        return FakeCollection(self._collections.setdefault(name, {}))


# ── Synthetic participant ─────────────────────────────────────────────────────
_FAVOURITE = {p.name: float(np.random.choice(p.levels)) for p in space.CONT_PARAMS}
for p in space.CAT_PARAMS:
    _FAVOURITE[p.name] = p.categories[np.random.randint(len(p.categories))]


def _rate(raw: dict) -> tuple[float, float]:
    d = 0.0
    for p in space.CONT_PARAMS:
        span = p.hi - p.lo or 1.0
        d += ((raw[p.name] - _FAVOURITE[p.name]) / span) ** 2
    for p in space.CAT_PARAMS:
        d += 0.0 if raw[p.name] == _FAVOURITE[p.name] else 0.5
    closeness = float(np.exp(-d))
    return (
        float(np.clip(closeness + np.random.normal(0, 0.03), 0, 1)),
        float(np.clip(0.3 + 0.7 * closeness + np.random.normal(0, 0.03), 0, 1)),
    )


def _params_of(doc: dict) -> dict:
    return {k: doc[k] for k in space.PARAM_NAMES}


def _assert_on_grid(raw: dict, ctx: str):
    for p in space.CONT_PARAMS:
        assert np.any(np.isclose(p.levels, raw[p.name], atol=10 ** -p.round_ndigits)), \
            f"{ctx}: {p.name}={raw[p.name]} off-grid"
    for p in space.CAT_PARAMS:
        assert raw[p.name] in p.categories, f"{ctx}: bad category {raw[p.name]}"
    assert space.is_feasible(raw), f"{ctx}: infeasible {raw}"


# ── The test ──────────────────────────────────────────────────────────────────
def run() -> None:
    fake = FakeFirestore()
    main.db = fake
    client = main.app.test_client()
    uid = "sim_user"

    # health
    r = client.get("/health").get_json()
    assert r["ok"] and r["space"] == space.PARAM_NAMES

    # register -> step_1 from Sobol
    r = client.post("/registerUser", json={"userId": uid}).get_json()
    assert r["ok"], r
    step1 = fake.collection("parameterValues")._store.get(f"{uid}_step_1")
    assert step1 is not None, "step_1 not written on registration"
    _assert_on_grid(_params_of(step1), "step_1")

    # duplicate registration is a no-op
    r = client.post("/registerUser", json={"userId": uid}).get_json()
    assert r.get("skipped"), "duplicate registerUser not suppressed"

    seen = set()
    step, completed = 1, False
    while not completed:
        doc = fake.collection("parameterValues")._store[f"{uid}_step_{step}"]
        raw = _params_of(doc)
        _assert_on_grid(raw, f"step_{step}")
        key = space.obs_key(raw)
        assert key not in seen, f"step_{step}: duplicate config {raw}"
        seen.add(key)

        # a failed attention check mid-study must be ignored by the optimizer
        if step == 3:
            fake.collection("interventionResults").document(f"{uid}_r3_failed").create(
                {"pid": uid, "phaseStep": step, "attentionCheckPassed": False,
                 "subjectiveScore": 0.0, "objectiveScore": 0.0, **raw})

        subj, obj = _rate(raw)
        fake.collection("interventionResults").document(f"{uid}_r{step}").create(
            {"pid": uid, "phaseStep": step, "attentionCheckPassed": True,
             "subjectiveScore": subj, "objectiveScore": obj, **raw})

        r = client.post("/updatePolicy",
                        json={"userId": uid, "type": "interventionResult"}).get_json()
        assert r["ok"], r

        if r.get("studyCompleted"):
            completed = True
            break

        assert r["obsCount"] == step, f"obsCount {r['obsCount']} != {step}"
        nxt = fake.collection("parameterValues")._store.get(f"{uid}_step_{step + 1}")
        assert nxt is not None, f"step_{step + 1} not written"
        expected = "exploration" if step < main.N_SOBOL else "optimization"
        assert r["phase"] == expected, f"step {step}: phase={r['phase']} != {expected}"

        # duplicate delivery (Cloud Functions is at-least-once) must be a no-op
        r2 = client.post("/updatePolicy",
                         json={"userId": uid, "type": "interventionResult"}).get_json()
        assert r2.get("skipped"), f"step {step}: duplicate updatePolicy not suppressed"

        print(f"step {step:>2} [{r['phase']:<12}] hv={r['hypervolume']:.4f} "
              f"y=({subj:.2f},{obj:.2f})")
        step += 1

    assert step == main.N_TOTAL, f"completed at {step}, expected {main.N_TOTAL}"
    user_doc = fake.collection("users")._store[uid]
    assert user_doc.get("studyCompleted") is True

    # non-result payloads are ignored
    r = client.post("/updatePolicy", json={"userId": uid, "type": "other"}).get_json()
    assert r.get("ignored")

    print(f"\nPASS: full HTTP loop, {main.N_TOTAL} observations "
          f"({main.N_SOBOL} Sobol + {main.N_TOTAL - main.N_SOBOL} MOBO), "
          f"all on-grid/unique, duplicates suppressed, "
          f"failed attention check excluded, studyCompleted set.")


if __name__ == "__main__":
    run()
