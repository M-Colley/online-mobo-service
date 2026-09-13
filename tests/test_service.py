"""
Offline smoke test of the FULL Flask service (main.py) — no cloud, no emulator.

    python tests/test_service.py

Drives the real HTTP endpoints through Flask's test client against an in-memory
fake of the exact Firestore API surface main.py uses (FieldFilter where /
order_by / stream / create / set / update / SERVER_TIMESTAMP). Simulates a
complete participant in the app's schemaVersion-2 shape:

    POST /registerUser
    repeat: read parameterValues -> render -> rate -> write interventionResults
            (echoing the haptics map, as the app does) -> POST /updatePolicy
    until studyCompleted

and asserts the things that would otherwise stall a real participant SILENTLY:

  • every proposal carries schemaVersion 2 and the literal phase "exploration"
    (the app's rules pin the result doc's phase to that string),
  • every proposal's haptics map passes the rules mirror, with pulseCount a real
    int and exactly 5 keys x 14 cues,
  • no proposal exceeds roundNumber 18,
  • round 1 is the anchor (the app team's seed design),
  • steps are contiguous, duplicate deliveries are suppressed, failed attention
    checks are excluded from training but still consume their round,
  • the study completes at exactly N_TOTAL and a completed participant cannot be
    re-seeded by the registration trigger.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set BEFORE importing main so firestore.Client() constructs offline.
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:1")  # never contacted
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "demo-vam-test")

import numpy as np
import torch
from google.api_core import exceptions as gapi_exceptions

import main
import space

torch.manual_seed(0)
np.random.seed(0)


# ── In-memory Firestore fake (only the surface main.py touches) ──────────────
class FakeSnapshot:
    def __init__(self, ref, data):
        self.reference = ref
        self._data = data

    @property
    def id(self):  # the real DocumentSnapshot exposes the document id
        return self.reference.id

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
            # the real client raises this specific type; main.py narrows to it
            raise gapi_exceptions.AlreadyExists(f"document {self.id} already exists")
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

    def limit(self, n):  # the real client's limit(); the fake keeps yielding lazily
        return self

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


def _rate(knobs: dict) -> tuple[float, float]:
    d = 0.0
    for p in space.CONT_PARAMS:
        span = p.hi - p.lo or 1.0
        d += ((knobs[p.name] - _FAVOURITE[p.name]) / span) ** 2
    for p in space.CAT_PARAMS:
        d += 0.0 if knobs[p.name] == _FAVOURITE[p.name] else 0.5
    closeness = float(np.exp(-d))
    return (
        float(np.clip(closeness + np.random.normal(0, 0.03), 0, 1)),
        float(np.clip(0.3 + 0.7 * closeness + np.random.normal(0, 0.03), 0, 1)),
    )


def _assert_proposal_is_sane(doc: dict, step: int, ctx: str):
    """Everything the app's security rules will be evaluated against."""
    assert doc["schemaVersion"] == space.SCHEMA_VERSION, f"{ctx}: wrong schemaVersion"
    assert doc["hapticMode"] == space.HAPTIC_MODE, f"{ctx}: wrong hapticMode"
    # The rules pin the RESULT doc's phase to the literal 'exploration'. If the
    # app echoes our label, anything else here makes every GP round illegal.
    assert doc["phase"] == "exploration", f"{ctx}: phase={doc['phase']!r} would be rejected"
    assert doc["phaseStep"] == step == doc["roundNumber"], f"{ctx}: step mismatch"
    assert step <= space.MAX_ROUND_NUMBER, f"{ctx}: roundNumber {step} exceeds the cap"
    assert doc["candidateId"].startswith(f"{doc['pid']}-round-{step:02d}-"), f"{ctx}: candidateId"
    assert space.rules_valid_haptics(doc["haptics"]), f"{ctx}: illegal haptics {doc['haptics']}"
    assert set(doc["haptics"]) == set(space.CUES), f"{ctx}: wrong cue set"
    for cue, burst in doc["haptics"].items():
        assert set(burst) == set(space.BURST_KEYS), f"{ctx}: {cue} has {sorted(burst)}"
        assert type(burst["pulseCount"]) is int, f"{ctx}: {cue} pulseCount is not an int"
    knobs = doc["mobo"]["knobs"]
    assert set(knobs) == set(space.KNOB_NAMES), f"{ctx}: knob names"
    assert all(type(v) is int for v in knobs.values()), f"{ctx}: knobs must be ints"
    assert doc["mobo"]["spaceVersion"] == space.SPACE_VERSION, f"{ctx}: spaceVersion"
    # the design must be recoverable from the echo — that is how history is read
    assert space.knobs_from_haptics(doc["haptics"]) == {k: float(v) for k, v in knobs.items()}


def _result_doc(uid: str, proposal: dict, step: int, subj: float, obj: float,
                attention: bool = True) -> dict:
    """An interventionResults doc in exactly the shape the app's rules demand."""
    rid = f"{uid}_r{step}" if attention else f"{uid}_r{step}_failed"
    return {
        "authUid": f"auth_{uid}",
        "pid": uid,
        "resultId": rid,
        "sessionId": f"{uid}_session",
        "parameterDocumentId": f"{uid}_step_{step}",
        "candidateId": proposal["candidateId"],
        "mapName": "map_a",
        "createdAt": "2026-09-11T12:00:00Z",
        "schemaVersion": space.SCHEMA_VERSION,
        "hapticMode": space.HAPTIC_MODE,
        "phase": "exploration",
        "phaseStep": step,
        "roundNumber": step,
        "subjectiveScore": subj,
        "objectiveScore": obj,
        "attentionCheckPassed": attention,
        "touchedTarget": True,
        "haptics": {c: dict(b) for c, b in proposal["haptics"].items()},
    }


# ── The test ──────────────────────────────────────────────────────────────────
def run() -> None:
    fake = FakeFirestore()
    main.db = fake
    client = main.app.test_client()
    uid = "sim_user"

    # health
    r = client.get("/health").get_json()
    assert r["ok"] and r["space"] == space.PARAM_NAMES
    assert r["schemaVersion"] == 2 and r["cues"] == 14
    assert r["N_TOTAL"] == space.MAX_ROUND_NUMBER == 18

    # a missing userId is a client error, not a write against a default pid
    assert client.post("/updatePolicy", json={"type": "interventionResult"}).status_code == 400
    assert client.post("/registerUser", json={}).status_code == 400

    # register -> step_1 is the ANCHOR (the app team's seed design)
    r = client.post("/registerUser", json={"userId": uid}).get_json()
    assert r["ok"], r
    step1 = fake.collection("parameterValues")._store.get(f"{uid}_step_1")
    assert step1 is not None, "step_1 not written on registration"
    _assert_proposal_is_sane(step1, 1, "step_1")
    assert step1["haptics"] == space.SEED_DESIGN, "round 1 must be the seed design"
    assert step1["mobo"]["phase"] == "anchor"
    assert step1["isFinalRound"] is False

    # duplicate registration is a no-op
    r = client.post("/registerUser", json={"userId": uid}).get_json()
    assert r.get("skipped"), "duplicate registerUser not suppressed"

    seen = set()
    step, completed = 1, False
    while not completed:
        doc = fake.collection("parameterValues")._store[f"{uid}_step_{step}"]
        _assert_proposal_is_sane(doc, step, f"step_{step}")
        knobs = {k: float(v) for k, v in doc["mobo"]["knobs"].items()}
        key = space.obs_key(knobs)
        assert key not in seen, f"step_{step}: duplicate design {knobs}"
        seen.add(key)

        # a failed attention check mid-study must be ignored by the optimizer
        if step == 3:
            fake.collection("interventionResults").document(f"{uid}_r3_failed").create(
                _result_doc(uid, doc, step, 0.0, 0.0, attention=False))

        subj, obj = _rate(knobs)
        fake.collection("interventionResults").document(f"{uid}_r{step}").create(
            _result_doc(uid, doc, step, subj, obj))

        r = client.post("/updatePolicy",
                        json={"userId": uid, "type": "interventionResult"}).get_json()
        assert r["ok"], r

        if r.get("studyCompleted"):
            completed = True
            break

        assert r["roundsDone"] == step, f"roundsDone {r['roundsDone']} != {step}"
        assert r["obsCount"] == step, f"obsCount {r['obsCount']} != {step}"
        nxt = fake.collection("parameterValues")._store.get(f"{uid}_step_{step + 1}")
        assert nxt is not None, f"step_{step + 1} not written"
        expected = "sobol" if step + 1 <= main.N_SOBOL else "mobo"
        assert nxt["mobo"]["phase"] == expected, \
            f"step {step + 1}: optimizer phase {nxt['mobo']['phase']} != {expected}"
        assert nxt["isFinalRound"] == (step + 1 >= main.N_TOTAL)
        if expected == "sobol":
            # The exploration sequence must be a function of the ROUND, not of how
            # many results happened to be usable. Tying it to the observation count
            # degrades Sobol into a repeated draw the moment a round is discarded.
            want = space.sobol_next(step + 1 - main.N_ANCHOR - 1)
            if space.obs_key(want) not in seen:   # no collision -> must be the draw
                got = {k: float(v) for k, v in nxt["mobo"]["knobs"].items()}
                assert got == want, (
                    f"step {step + 1}: {got} is not the Sobol draw for this round "
                    f"({want}) — the exploration pointer is not tied to the round"
                )

        # duplicate delivery (Cloud Functions is at-least-once) must be a no-op
        r2 = client.post("/updatePolicy",
                         json={"userId": uid, "type": "interventionResult"}).get_json()
        assert r2.get("skipped"), f"step {step}: duplicate updatePolicy not suppressed"

        print(f"step {step:>2} [{nxt['mobo']['phase']:<6}] hv={r['hypervolume']:.4f} "
              f"y=({subj:.2f},{obj:.2f})")
        step += 1

    assert step == main.N_TOTAL, f"completed at {step}, expected {main.N_TOTAL}"
    user_doc = fake.collection("users")._store[uid]
    assert user_doc.get("studyCompleted") is True

    # the last proposal must have told the app it was the last one
    final = fake.collection("parameterValues")._store[f"{uid}_step_{main.N_TOTAL}"]
    assert final["isFinalRound"] is True, "the app cannot read users/ — it needs this flag"

    # never propose past the rules' cap
    assert f"{uid}_step_{main.N_TOTAL + 1}" not in fake.collection("parameterValues")._store

    # hypervolume went to OUR collection, never into the app's result docs
    assert fake.collection("moboMetrics")._store, "hypervolume not recorded"
    for res in fake.collection("interventionResults")._store.values():
        assert "hypervolume" not in res, "must not annotate the app's own documents"

    # completion must not let the registration trigger re-seed the participant
    r = client.post("/registerUser", json={"userId": uid}).get_json()
    assert r.get("skipped") and r.get("studyCompleted"), "completed participant was re-seeded"

    # non-result payloads are ignored
    r = client.post("/updatePolicy", json={"userId": uid, "type": "other"}).get_json()
    assert r.get("ignored")

    print(f"\nPASS: full HTTP loop, {main.N_TOTAL} rounds "
          f"({main.N_ANCHOR} anchor + {main.N_SOBOL - main.N_ANCHOR} Sobol + "
          f"{main.N_TOTAL - main.N_SOBOL} MOBO), every design legal under the app's "
          f"rules, duplicates suppressed, failed attention check excluded, "
          f"studyCompleted set.")


def run_discarded_round_keeps_the_counter() -> None:
    """A round with no usable result still consumed its roundNumber.

    The rules pin phaseStep == roundNumber, so if the service counted clean
    OBSERVATIONS it would re-propose a step whose doc already exists and return
    {"skipped": true} forever.
    """
    fake = FakeFirestore()
    main.db = fake
    client = main.app.test_client()
    uid = "discard_user"

    client.post("/registerUser", json={"userId": uid})
    p1 = fake.collection("parameterValues")._store[f"{uid}_step_1"]
    fake.collection("interventionResults").document(f"{uid}_r1").create(
        _result_doc(uid, p1, 1, 0.5, 0.5))
    client.post("/updatePolicy", json={"userId": uid, "type": "interventionResult"})

    # round 2 comes back with a failed attention check only — no usable data
    p2 = fake.collection("parameterValues")._store[f"{uid}_step_2"]
    fake.collection("interventionResults").document(f"{uid}_r2_failed").create(
        _result_doc(uid, p2, 2, 0.0, 0.0, attention=False))
    r = client.post("/updatePolicy", json={"userId": uid, "type": "interventionResult"}).get_json()

    assert r["roundsDone"] == 2, f"roundsDone={r['roundsDone']} — a discarded round must still count"
    assert r["obsCount"] == 1, f"obsCount={r['obsCount']} — the failed round must not train the GP"
    assert r["nextPhaseStep"] == 3, f"nextPhaseStep={r['nextPhaseStep']} — would collide with step 2"
    p3 = fake.collection("parameterValues")._store.get(f"{uid}_step_3")
    assert p3 is not None
    # ...and the DESIGN must advance too. Indexing the Sobol pointer by the usable
    # observation count would hand out step 2's design again here.
    assert p3["mobo"]["knobs"] != p2["mobo"]["knobs"], \
        "step 3 re-issued step 2's design — the exploration pointer is pinned"
    print("PASS: a discarded round advances both the round counter and the design.")


def run_unusable_results_never_pin_the_design() -> None:
    """The worst case the round counter alone does not cover.

    If every echo comes back off the knob grid (the app clamps or substitutes
    something), NO result is usable, so a design chosen by usable-observation count
    would be re-issued for all 17 remaining rounds — the participant feels one
    stimulus for the entire study and the only trace is a per-round log line.
    """
    fake = FakeFirestore()
    main.db = fake
    client = main.app.test_client()
    uid = "clamped_user"

    client.post("/registerUser", json={"userId": uid})
    designs = []
    for step in range(1, 8):
        prop = fake.collection("parameterValues")._store[f"{uid}_step_{step}"]
        designs.append(tuple(sorted(prop["mobo"]["knobs"].items())))
        doc = _result_doc(uid, prop, step, 0.6, 0.6)
        # the app "rendered" something off-grid: nudge one cue's intensity
        doc["haptics"]["turn"]["intensity"] = 0.123
        fake.collection("interventionResults").document(f"{uid}_r{step}").create(doc)
        r = client.post("/updatePolicy",
                        json={"userId": uid, "type": "interventionResult"}).get_json()
        assert r["ok"], r
        assert r["obsCount"] == 0, "an off-grid echo must not be trained on"
        assert r["roundsDone"] == step

    assert len(set(designs)) == len(designs), \
        f"only {len(set(designs))} distinct designs in {len(designs)} rounds — pinned"
    # Stronger: each exploration round must be ITS OWN Sobol draw. Distinctness
    # alone can be satisfied by the random dedup fallback, which would silently
    # replace the space-filling design with arbitrary points.
    for step, got in enumerate(designs[1:], start=2):
        want = space.sobol_next(step - main.N_ANCHOR - 1)
        assert dict(got) == {k: int(v) for k, v in want.items()}, (
            f"round {step} proposed {dict(got)}, not its Sobol draw {want} — "
            "the exploration pointer is tied to the observation count"
        )
    print("PASS: unusable results do not pin the participant to one design.")


def run_foreign_schema_docs_do_not_move_the_counter() -> None:
    """One stray doc must not be able to end the study before it begins."""
    fake = FakeFirestore()
    main.db = fake
    client = main.app.test_client()
    uid = "stray_user"

    client.post("/registerUser", json={"userId": uid})
    p1 = fake.collection("parameterValues")._store[f"{uid}_step_1"]
    fake.collection("interventionResults").document(f"{uid}_r1").create(
        _result_doc(uid, p1, 1, 0.5, 0.5))
    # a legacy / pilot / QA doc for the same pid, at a late round
    stray = _result_doc(uid, p1, 17, 0.9, 0.9)
    stray["schemaVersion"] = 1
    fake.collection("interventionResults").document(f"{uid}_stray").create(stray)

    r = client.post("/updatePolicy",
                    json={"userId": uid, "type": "interventionResult"}).get_json()
    assert r["roundsDone"] == 1, \
        f"roundsDone={r['roundsDone']} — a schemaVersion-1 doc set the round counter"
    assert r["nextPhaseStep"] == 2 and not r.get("studyCompleted")
    print("PASS: a foreign-schema result document cannot end the study early.")


def run_illegal_design_is_loud() -> None:
    """A design the app's rules would reject must fail LOUDLY — and deterministically,
    so as a 4xx the Cloud Function will not retry it for 24 h."""
    fake = FakeFirestore()
    main.db = fake
    client = main.app.test_client()
    uid = "illegal_user"

    real_expand = space.expand
    try:
        def broken_expand(knobs):
            h = real_expand(knobs)
            h["turn"]["pulseCount"] = 999          # violates validBurst
            return h
        space.expand = broken_expand
        resp = client.post("/registerUser", json={"userId": uid})
        assert resp.status_code == 422, \
            f"an illegal design answered {resp.status_code} — must be a non-retried 4xx"
        assert resp.get_json()["retry"] is False
        assert f"{uid}_step_1" not in fake.collection("parameterValues")._store, \
            "an illegal design was written anyway"
    finally:
        space.expand = real_expand
    print("PASS: an illegal design fails loudly and is never written.")


def run_seeded_round_one_is_in_the_novelty_set() -> None:
    """The app team seeds round 1 with an auto-id doc that has NO mobo map.

    It must still count as issued: if round 1's result is unusable, the anchor
    is in neither the observed set nor (formerly) the issued set, and a later GP
    optimum at knobs=0 would hand the participant the seed design a second time.
    """
    fake = FakeFirestore()
    main.db = fake
    client = main.app.test_client()
    uid = "101"
    fake.collection("parameterValues").document("XmNWlxIjB4sGrle1ehBk").create({
        "candidateId": "101-round-01-seed-20260909T171125Z", "schemaVersion": 2,
        "phase": "exploration", "phaseStep": 1, "pid": uid, "createdAt": "2026-09-09",
        "haptics": {c: dict(b) for c, b in space.SEED_DESIGN.items()}})
    r = client.post("/registerUser", json={"userId": uid}).get_json()
    assert r.get("skipped"), "must not write a competing round-1 design"
    assert f"{uid}_step_1" not in fake.collection("parameterValues")._store
    # round 1 fails the attention check: no observation, but the design WAS felt
    seed_doc = fake.collection("parameterValues")._store["XmNWlxIjB4sGrle1ehBk"]
    fake.collection("interventionResults").document(f"{uid}_r1_failed").create(
        {**_result_doc(uid, {**seed_doc, "candidateId": seed_doc["candidateId"]}, 1, 0.0, 0.0,
                       attention=False), "parameterDocumentId": "XmNWlxIjB4sGrle1ehBk"})
    assert space.obs_key(space.anchor()) in main.Proposals(uid).issued, \
        "the seeded anchor is not in the novelty set"
    r = client.post("/updatePolicy", json={"userId": uid, "type": "interventionResult"}).get_json()
    assert r["roundsDone"] == 1 and r["nextPhaseStep"] == 2, r
    assert r["nextKnobs"] != space.anchor(), "the seed design was re-issued"
    print("PASS: a hand-seeded round-1 doc without a mobo map still counts as issued.")


def run_integral_float_phase_step_counts_as_a_round() -> None:
    """phaseStep 5.0 (an Admin-SDK/QA writer) must be round 5, not a half-admitted
    doc that trains the GP yet leaves the counter at 4 — the permanent skip."""
    fake = FakeFirestore()
    main.db = fake
    client = main.app.test_client()
    uid = "float_user"
    client.post("/registerUser", json={"userId": uid})
    for step in range(1, 6):
        prop = fake.collection("parameterValues")._store[f"{uid}_step_{step}"]
        doc = _result_doc(uid, prop, step, 0.5, 0.5)
        if step == 5:
            doc["phaseStep"] = 5.0
            doc["roundNumber"] = 5.0
        fake.collection("interventionResults").document(f"{uid}_r{step}").create(doc)
        r = client.post("/updatePolicy", json={"userId": uid, "type": "interventionResult"}).get_json()
        assert r["ok"] and not r.get("skipped"), (step, r)
    assert r["roundsDone"] == 5 and r["obsCount"] == 5 and r["nextPhaseStep"] == 6, r
    assert f"{uid}_step_6" in fake.collection("parameterValues")._store
    # and a NON-integral step is ignored entirely rather than half-admitted
    prop = fake.collection("parameterValues")._store[f"{uid}_step_6"]
    bad = _result_doc(uid, prop, 6, 0.5, 0.5)
    bad["phaseStep"] = 6.5
    fake.collection("interventionResults").document(f"{uid}_r6_bad").create(bad)
    r = client.post("/updatePolicy", json={"userId": uid, "type": "interventionResult"}).get_json()
    assert r.get("skipped") and f"{uid}_step_7" not in fake.collection("parameterValues")._store
    print("PASS: an integral-float phaseStep is a round; a non-integral one is ignored whole.")


def run_exporter_matches_the_optimizer() -> None:
    """analysis/export_firestore.build_rows must admit exactly the rounds main.py trains on."""
    import importlib
    export = importlib.import_module("analysis.export_firestore")
    uid = "p1"
    prop = main.build_proposal_doc(uid, space.anchor(), "anchor", 2)
    prop["createdAt"] = "x"
    by_id = {f"{uid}_step_2": prop}
    by_step = {(uid, 2): prop}
    failed = _result_doc(uid, prop, 2, 0.1, 0.1, attention=False)
    passed = _result_doc(uid, prop, 2, 0.9, 0.9)
    # the FAILED doc streams first (doc ids are the app's resultIds — any order)
    rows, long_rows, skipped, _ = export.build_rows(
        [("a_failed", failed), ("b_passed", passed)], by_id, by_step, {})
    kept = {(r["phaseStep"], r["attentionCheckPassed"]) for r in rows}
    assert (2, True) in kept, "the passing repeat was dropped as a duplicate of the failed doc"
    assert (2, False) in kept, "failed rows are exported (R drops them), not silently lost"
    assert len(long_rows) == 2 * len(space.CUES)
    # a proposal from another spaceVersion excludes the round, exactly like main.py
    stale = {**prop, "mobo": {**prop["mobo"], "spaceVersion": "v2-old"}}
    rows, _, skipped, _ = export.build_rows(
        [("c", passed)], {f"{uid}_step_2": stale}, {(uid, 2): stale}, {})
    assert rows == [] and skipped == 1, "a stale-spaceVersion round leaked into the CSV"
    # numeric pids and integral-float steps do not crash the export
    odd = {**passed, "pid": 101, "phaseStep": 2.0}
    rows, *_ = export.build_rows([("d", odd)], {"101_step_2": prop}, {("101", 2): prop}, {})
    assert rows and rows[0]["pid"] == "101" and rows[0]["phaseStep"] == 2
    print("PASS: the exporter admits exactly what the optimizer trains on.")


if __name__ == "__main__":
    run()
    run_discarded_round_keeps_the_counter()
    run_unusable_results_never_pin_the_design()
    run_foreign_schema_docs_do_not_move_the_counter()
    run_illegal_design_is_loud()
    run_seeded_round_one_is_in_the_novelty_set()
    run_integral_float_phase_step_counts_as_a_round()
    run_exporter_matches_the_optimizer()
