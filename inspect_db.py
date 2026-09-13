"""
Read-only Firestore inspector — verify the live database against the contract
this service assumes (space.py), BEFORE touching anything.

It only READS. It never writes, updates, or deletes anything.

This is the tool that would have shown, in one run, that a candidate is a nested
14-cue `haptics` map rather than five flat fields. Run it first, every time the
app team says "the parameters changed".

Auth (either one):
    # a) service-account key file
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    # b) gcloud application-default login
    gcloud auth application-default login

Run:
    GOOGLE_CLOUD_PROJECT=project-multinav python inspect_db.py
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

import space

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise SystemExit(
        "Set GOOGLE_CLOUD_PROJECT to your GCP project id, e.g.\n"
        "  GOOGLE_CLOUD_PROJECT=project-multinav python inspect_db.py"
    )
DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")

RESULTS = "interventionResults"
PROPOSALS = "parameterValues"
METRICS = "moboMetrics"
USERS = "users"

# Exactly what the app's validResult() rule requires on every result doc.
REQUIRED_RESULT_FIELDS = [
    "authUid", "pid", "resultId", "sessionId", "parameterDocumentId", "candidateId",
    "mapName", "createdAt", "schemaVersion", "hapticMode", "phase", "phaseStep",
    "roundNumber", "subjectiveScore", "objectiveScore", "attentionCheckPassed",
    "touchedTarget", "haptics",
]


def scan(db, collection):
    docs = [(d.id, d.to_dict()) for d in db.collection(collection).stream()]
    print(f"\n=== {collection}  ({len(docs)} docs) ===")
    if not docs:
        return docs
    keys = Counter()
    for _, d in docs:
        keys.update(d.keys())
    print("  fields present (doc count):")
    for k, c in sorted(keys.items()):
        print(f"    {k:<24} {c}/{len(docs)}")
    return docs


def _bursts(docs):
    """Every (source, pid, step, cue, burst) in a list of docs that has haptics."""
    for doc_id, d in docs:
        haptics = d.get("haptics")
        if isinstance(haptics, dict):
            for cue, burst in haptics.items():
                if isinstance(burst, dict):
                    yield doc_id, d.get("pid"), d.get("phaseStep"), cue, burst


def main():
    db = firestore.Client(project=PROJECT_ID, database=DATABASE)
    print(f"project: {PROJECT_ID}  database: {DATABASE}")
    print(f"this service expects: schemaVersion {space.SCHEMA_VERSION}, "
          f"{len(space.CUES)} cues x {len(space.BURST_KEYS)} burst fields, "
          f"rounds <= {space.MAX_ROUND_NUMBER}, spaceVersion {space.SPACE_VERSION}")

    results = scan(db, RESULTS)
    proposals = scan(db, PROPOSALS)
    scan(db, METRICS)
    scan(db, USERS)

    all_docs = results + proposals

    # ── CHECK 1 — shape: is it the nested schema this service writes? ───────
    print("\n" + "=" * 70)
    print("CHECK 1 — document shape")
    versions = Counter(d.get("schemaVersion") for _, d in all_docs)
    print(f"    schemaVersion values seen: {dict(versions)}")
    if any(v != space.SCHEMA_VERSION for v in versions):
        print(f"    ^ docs without schemaVersion == {space.SCHEMA_VERSION} are INVISIBLE to the")
        print("      app's listener (it filters on pid + schemaVersion + createdAt).")
    cue_sets = Counter()
    for _, d in all_docs:
        h = d.get("haptics")
        if isinstance(h, dict):
            cue_sets[tuple(sorted(h))] += 1
    for cues, n in cue_sets.items():
        status = "OK" if set(cues) == set(space.CUES) else "MISMATCH vs space.CUES"
        print(f"    {n} doc(s) with {len(cues)} cues — {status}")
        if set(cues) != set(space.CUES):
            print(f"      only in db   : {sorted(set(cues) - set(space.CUES))}")
            print(f"      only in space: {sorted(set(space.CUES) - set(cues))}")

    # ── CHECK 2 — observed ranges per burst field ───────────────────────────
    print("\nCHECK 2 — observed ranges per burst field (vs the app's rules)")
    vals = defaultdict(list)
    for _, _, _, _, burst in _bursts(all_docs):
        for k, v in burst.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals[k].append(v)
    bounds = {
        "intensity": (space.INTENSITY_MIN, space.INTENSITY_MAX),
        "sharpness": (space.SHARPNESS_MIN, space.SHARPNESS_MAX),
        "pulseCount": (space.PULSE_COUNT_MIN, space.PULSE_COUNT_MAX),
        "onDuration": (space.DURATION_MIN, space.DURATION_MAX),
        "offDuration": (space.DURATION_MIN, space.DURATION_MAX),
    }
    for k in space.BURST_KEYS:
        v = vals.get(k, [])
        if not v:
            print(f"    {k:<12} (no values)")
            continue
        lo, hi = bounds[k]
        flag = "" if (min(v) >= lo and max(v) <= hi) else "   <-- OUTSIDE THE RULES BOUNDS"
        print(f"    {k:<12} n={len(v):<5} min={min(v):<8g} max={max(v):<8g} "
              f"distinct={len(set(v)):<4} rules=[{lo}, {hi}]{flag}")
    # Only bursts the rules mirror accepts are divided — a string onDuration or
    # a bool pulseCount would otherwise crash this line BEFORE CHECK 3 reports it.
    rates = [b["pulseCount"] / b["onDuration"]
             for *_, b in _bursts(all_docs) if space.rules_valid_burst(b)]
    if rates:
        print(f"    pulse rate   min={min(rates):g} Hz  max={max(rates):g} Hz  "
              f"distinct={sorted(set(round(r, 3) for r in rates))[:8]}  "
              f"(rules cap {space.MAX_PULSE_RATE_HZ:g} Hz)")

    # ── CHECK 3 — would the app's own write be legal? ───────────────────────
    print("\nCHECK 3 — every burst against the rules mirror (validBurst)")
    bad = [(doc_id, cue, burst) for doc_id, _, _, cue, burst in _bursts(all_docs)
           if not space.rules_valid_burst(burst)]
    total = sum(1 for _ in _bursts(all_docs))
    print(f"    {total - len(bad)}/{total} bursts would pass validBurst()")
    for doc_id, cue, burst in bad[:10]:
        why = []
        if not isinstance(burst.get("pulseCount"), int) or isinstance(burst.get("pulseCount"), bool):
            why.append("pulseCount is not an int64")
        if set(burst) != set(space.BURST_KEYS):
            why.append(f"keys {sorted(burst)}")
        print(f"      {doc_id} / {cue}: {', '.join(why) or 'out of range'} -> {burst}")
    if bad:
        print("    ^ these make the APP's interventionResults write ILLEGAL — the")
        print("      participant stalls and nothing is logged on our side.")

    # ── CHECK 4 — contract: does a result carry what the rules demand? ──────
    print("\nCHECK 4 — interventionResults against validResult()")
    if not results:
        print("    (no results yet)")
    else:
        offenders = Counter()
        ok = 0
        for _, d in results:
            miss = [f for f in REQUIRED_RESULT_FIELDS if f not in d]
            if miss:
                for f in miss:
                    offenders[f] += 1
            else:
                ok += 1
        print(f"    {ok}/{len(results)} docs carry every required field")
        if offenders:
            print(f"    missing-field tally: {dict(offenders)}")
        steps = [d.get("phaseStep") for _, d in results if isinstance(d.get("phaseStep"), int)]
        if steps and max(steps) > space.MAX_ROUND_NUMBER:
            print(f"    ^ phaseStep {max(steps)} exceeds the rules cap {space.MAX_ROUND_NUMBER}")

    # ── CHECK 5 — can we recover the knobs from what is stored? ─────────────
    print("\nCHECK 5 — knob recoverability (can the optimizer learn from this?)")
    recoverable = unreachable = 0
    for _, d in all_docs:
        h = d.get("haptics")
        if not isinstance(h, dict):
            continue
        if space.knobs_from_haptics(h) is not None:
            recoverable += 1
        else:
            unreachable += 1
    print(f"    {recoverable} design(s) invert to a knob vector; {unreachable} do not")
    if unreachable:
        print(f"    ^ designs outside the {space.SPACE_VERSION} grid (a hand-seeded")
        print("      variant, an app-side clamp, or a stale spaceVersion). They are")
        print("      excluded from GP training — check they are not the whole study.")

    # ── CHECK 6 — the app's own listener query (catches a deleted index) ────
    print("\nCHECK 6 — the app's listener query shape")
    pids = sorted({str(d.get("pid")) for _, d in proposals if d.get("pid") is not None})
    if not pids:
        print("    (no proposals yet)")
    for pid in pids[:3]:
        try:
            docs = list(
                db.collection(PROPOSALS)
                .where(filter=FieldFilter("pid", "==", pid))
                .where(filter=FieldFilter("schemaVersion", "==", space.SCHEMA_VERSION))
                .order_by("createdAt", direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream()
            )
            if docs:
                d = docs[0].to_dict()
                print(f"    pid={pid}: newest visible proposal is step "
                      f"{d.get('phaseStep')} ({d.get('candidateId')})")
            else:
                print(f"    pid={pid}: NO doc matches — the app would receive nothing")
        except Exception as e:  # a deleted composite index surfaces here
            print(f"    pid={pid}: QUERY FAILED — {type(e).__name__}: {e}")
            print("      ^ most likely the parameterValues(pid, schemaVersion, createdAt DESC)")
            print("        composite index is missing. See firebase/firestore.indexes.json.")


if __name__ == "__main__":
    main()
