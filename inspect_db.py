"""
Read-only Firestore inspector — run it with your DB access to verify the ranges
baked into space.py against live data (and to extract the facts still missing:
the intensity/sharpness ranges) and to sanity-check the optimizer's data
contract.

It only READS. It never writes, updates, or deletes anything.

Auth (either one):
    # a) service-account key file
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    # b) gcloud application-default login
    gcloud auth application-default login

Run:
    python inspect_db.py
"""

from __future__ import annotations

import os
from collections import Counter

from google.cloud import firestore

import space

# GCP project to read from. Cloud Shell sets GOOGLE_CLOUD_PROJECT automatically;
# elsewhere: GOOGLE_CLOUD_PROJECT=<your-project-id> python inspect_db.py
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise SystemExit(
        "Set GOOGLE_CLOUD_PROJECT to your GCP project id, e.g.\n"
        "  GOOGLE_CLOUD_PROJECT=my-project python inspect_db.py"
    )
# Reads the "(default)" database; override for a named database:
#   FIRESTORE_DATABASE=<name> python inspect_db.py
DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")

RESULTS = "interventionResults"
PARAMS = "parameterValues"
SURVEY = "surveyResponses"


def _numeric_summary(name, values):
    vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        print(f"    {name:<12} (no numeric values)")
        return
    uniq = sorted(set(vals))
    shown = uniq if len(uniq) <= 12 else uniq[:6] + ["..."] + uniq[-3:]
    print(f"    {name:<12} n={len(vals):<4} min={min(vals):<8g} max={max(vals):<8g} "
          f"distinct={len(uniq):<4} {shown}")


def scan(db, collection):
    docs = [d.to_dict() for d in db.collection(collection).stream()]
    print(f"\n=== {collection}  ({len(docs)} docs) ===")
    if not docs:
        return docs
    keys = Counter()
    for d in docs:
        keys.update(d.keys())
    print("  fields present (doc count):")
    for k, c in sorted(keys.items()):
        print(f"    {k:<22} {c}/{len(docs)}")
    return docs


def main():
    db = firestore.Client(project=PROJECT_ID, database=DATABASE)
    print(f"project: {PROJECT_ID}  database: {DATABASE}")

    results = scan(db, RESULTS)
    params = scan(db, PARAMS)
    survey = scan(db, SURVEY)

    # ── Range / category checks against space.py ────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 1 — pattern: category strings seen in live data")
    patterns = Counter()
    for d in results + params:
        if "pattern" in d:
            patterns[d["pattern"]] += 1
    for p, c in patterns.most_common():
        print(f"    {p!r:<18} {c}")
    print(f"  -> must be a subset of space.py CAT_PARAMS. Currently: "
          f"{space.CAT_PARAMS[0].categories if space.CAT_PARAMS else '[]'} "
          f"(app team confirmed constant/puls, 2026-07-21)")

    print("\nCHECK 2 — observed ranges vs the JND grids in space.py")
    for field in ("intensity", "sharpness", "duration", "interval"):
        vals = [d[field] for d in results + params if field in d]
        _numeric_summary(field, vals)
    print("  -> duration 0.03–20 s and interval 1–20 Hz are app-team-confirmed; "
          "use the intensity/sharpness min/max here to close the remaining TODOs.")

    # ── Contract / surveyResponses question ─────────────────────────────────
    print("\n" + "=" * 70)
    print("CONTRACT CHECK — does interventionResults carry everything the optimizer needs?")
    required = space.PARAM_NAMES + space.OBJECTIVE_FIELDS + ["pid", "phaseStep", "attentionCheckPassed"]
    missing_any = 0
    for d in results:
        miss = [f for f in required if f not in d]
        if miss:
            missing_any += 1
    print(f"    {len(results) - missing_any}/{len(results)} interventionResults docs "
          f"have all required fields: {required}")
    if missing_any:
        # show which fields are the usual offenders
        offenders = Counter()
        for d in results:
            for f in required:
                if f not in d:
                    offenders[f] += 1
        print(f"    missing-field tally: {dict(offenders)}")

    print("\n  surveyResponses shape (the open question):")
    has_phasestep = sum(1 for d in survey if "phaseStep" in d)
    has_objscore = sum(1 for d in survey if "objectiveScore" in d)
    print(f"    {has_phasestep}/{len(survey)} have phaseStep, "
          f"{has_objscore}/{len(survey)} have objectiveScore")
    print("    -> if the app is migrating results into surveyResponses, it must ALSO "
          "carry phaseStep + objectiveScore + the 5 params, or the optimizer can't use it.")

    # ── Most recent trials: which collection is actually being written? ─────
    print("\n  most recent createdAt per collection (is interventionResults still live?):")
    for name, docs in ((RESULTS, results), (PARAMS, params), (SURVEY, survey)):
        stamps = [d["createdAt"] for d in docs if "createdAt" in d]
        newest = max(stamps) if stamps else None
        print(f"    {name:<20} newest: {newest}")


if __name__ == "__main__":
    main()
