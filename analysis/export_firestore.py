"""
Export the study data from Firestore to tidy CSVs for the R analysis.

Read-only: it never writes, updates or deletes anything.

Auth (either one):
    # a) service-account key file
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    # b) gcloud application-default login
    gcloud auth application-default login

Run:
    GOOGLE_CLOUD_PROJECT=project-multinav python analysis/export_firestore.py
    # -> analysis/trials.csv             one row per completed round
    # -> analysis/trials_haptics_long.csv  one row per (round, cue)

Under schemaVersion 2 a candidate is a nested 14-cue `haptics` map, so there is
no longer a flat parameter column to read off the result doc. This joins:

    interventionResults  --parameterDocumentId-->  parameterValues   (the knobs,
                                                                      the real
                                                                      optimizer
                                                                      phase)
                         --{pid}_step_{n}------->  moboMetrics       (the HV the
                                                                      service
                                                                      computed)

The wide CSV carries the four KNOBS (what was optimized); the long CSV carries
the 70 rendered numbers (what the participant actually felt). Analysing the
design means the long file — the wide one cannot represent a 14-cue design.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import space  # noqa: E402
from study_contract import (  # noqa: E402
    QUESTIONNAIRE_VERSION,
    validate_questionnaire,
    validate_scores,
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_CSV", os.path.join(HERE, "trials.csv"))
OUT_LONG = os.environ.get("OUT_LONG_CSV", os.path.join(HERE, "trials_haptics_long.csv"))

# Test participants to leave out of the analysis. Extend as needed.
EXCLUDE_PREFIXES = ("claude_", "test_", "debug_", "sim_", "discard_")

# `phase` is the literal the app's rules require ('exploration' on every round);
# `optimizerPhase` is the real label (anchor / sobol / mobo / sobol-fallback /
# retest) and is the one the analysis must split on. `retest` is the seed design
# again in the final round(s): a drift measurement, not an optimizer choice.
COLUMNS = (["pid", "phaseStep", "roundNumber", "phase", "optimizerPhase",
            "attentionCheckPassed", "touchedTarget", "mapName", "sessionId",
            "resultId", "candidateId", "parameterDocumentId", "schemaVersion",
            "spaceVersion", "questionnaireVersion", "trainingEligible",
            "trainingExclusionReason", "q_cognitive_load", "q_clear",
            "q_likability", "q_comfort", "q_pleasant", "q_distinct",
            "timeToTargetSeconds"]
           + space.KNOB_NAMES
           + space.OBJECTIVE_FIELDS
           + ["hypervolume", "createdAt"])

LONG_COLUMNS = ["pid", "sessionId", "resultId", "phaseStep", "questionnaireVersion",
                "trainingEligible", "cue"] + list(space.BURST_KEYS)


def excluded(pid: str) -> bool:
    return not pid or pid.startswith(EXCLUDE_PREFIXES)


def _matching_proposal(proposal, pid):
    """Match the service's per-pid, current-schema proposal query.

    Missing schema/pid remain accepted for callers supplying old in-memory
    fixtures; Firestore snapshots are filtered strictly when loaded in main().
    """
    if (not isinstance(proposal, dict)
            or proposal.get("schemaVersion", space.SCHEMA_VERSION) != space.SCHEMA_VERSION
            or str(proposal.get("pid", pid)) != pid):
        return None
    return proposal


def build_rows(results, proposals_by_id, proposals_by_step, hv_by_step):
    """(rows, long_rows, skipped, excluded) from raw result docs — no Firestore.

    Keep valid historic scores and raw questionnaires without pooling their
    different meanings. `trainingEligible` identifies the subset admitted by
    main.load_observations under the current questionnaire contract:
      * schemaVersion must match, else the doc is not a round of this study;
      * phaseStep must be a whole number (space.round_number), else ignored;
      * the knob vector comes from space.decode_knobs — the ECHO is authoritative
        and a proposal from another spaceVersion excludes the round;
      * objectives must be finite numeric values in [0, 1];
      * questionnaires retain their recorded version (absent means legacy);
      * a round is claimed by the first valid PASSING result in its questionnaire
        version. Invalid current questionnaires never shadow a valid result;
      * a failed attention check
        is still exported (evaluate_mobo.R drops it) but never claims the round,
        so it cannot shadow the passing repeat behind it whatever order Firestore
        streams them in.
    """
    rows, long_rows, skipped, excluded_n = [], [], 0, 0
    claimed: set[tuple[str, int, str]] = set()
    for doc_id, r in results:
        pid = r.get("pid")
        pid = str(pid) if pid is not None else None
        if excluded(pid):
            excluded_n += 1
            continue
        step = space.round_number(r.get("phaseStep"))
        if step is None:
            skipped += 1
            print(f"  skipping {doc_id}: phaseStep={r.get('phaseStep')!r} is not a round number")
            continue
        if not 1 <= step <= space.MAX_ROUND_NUMBER:
            skipped += 1
            print(f"  skipping {doc_id}: phaseStep={step} is outside "
                  f"1..{space.MAX_ROUND_NUMBER}")
            continue
        if r.get("schemaVersion") != space.SCHEMA_VERSION:
            skipped += 1
            print(f"  skipping {pid} step={step}: schemaVersion="
                  f"{r.get('schemaVersion')!r}, not this study")
            continue
        passed = r.get("attentionCheckPassed") is not False   # absent == passed
        # Questionnaire versions define different response scales, even though
        # all subjective scores use the numeric range [0, 1]. Never let a
        # legacy result claim the current version's round during export.
        questionnaire_version = r.get("questionnaireVersion", "")
        version_key = (type(questionnaire_version).__name__, repr(questionnaire_version))
        claim_key = (pid, step, str(version_key))
        training_eligible = passed
        exclusion_reason = "" if passed else "failed legacy attention check"
        try:
            validate_questionnaire(r)
        except ValueError as exc:
            training_eligible = False
            exclusion_reason = str(exc)
        if passed and claim_key in claimed:
            skipped += 1
            print(f"  skipping {pid} step={step}: duplicate result document "
                  f"(the optimizer trains on the first passing one only)")
            continue

        try:
            scores = validate_scores(r)
        except ValueError as exc:
            skipped += 1
            print(f"  skipping {pid} step={step}: {exc}")
            continue

        proposal = _matching_proposal(
            proposals_by_id.get(str(r.get("parameterDocumentId"))), pid)
        if proposal is None:
            proposal = _matching_proposal(proposals_by_step.get((pid, step)), pid)
        mobo = (proposal or {}).get("mobo")
        mobo = mobo if isinstance(mobo, dict) else {}
        knobs, why = space.decode_knobs(r, proposal)
        if knobs is None:
            skipped += 1
            print(f"  skipping {pid} step={step}: {why}")
            continue
        # Historical questionnaires are still useful exports. For current
        # version docs, a malformed questionnaire cannot hide a later valid
        # duplicate that the optimizer actually trained on.
        if passed and (training_eligible or questionnaire_version != QUESTIONNAIRE_VERSION):
            claimed.add(claim_key)

        row = {
            "pid": pid,
            "phaseStep": step,
            "roundNumber": r.get("roundNumber", step),
            "phase": r.get("phase", (proposal or {}).get("phase", "")),
            "optimizerPhase": mobo.get("phase", ""),
            "attentionCheckPassed": passed,
            "touchedTarget": r.get("touchedTarget", ""),
            "mapName": r.get("mapName", ""),
            "sessionId": r.get("sessionId", ""),
            "resultId": r.get("resultId", doc_id),
            "candidateId": r.get("candidateId", ""),
            "parameterDocumentId": r.get("parameterDocumentId", ""),
            "spaceVersion": mobo.get("spaceVersion", ""),
            "schemaVersion": r.get("schemaVersion", ""),
            "questionnaireVersion": questionnaire_version,
            "trainingEligible": training_eligible,
            "trainingExclusionReason": exclusion_reason,
            "timeToTargetSeconds": r.get("timeToTargetSeconds", ""),
            # Current service metrics describe only the current instrument's
            # training set; attaching them to legacy scores would mislabel it.
            "hypervolume": hv_by_step.get((pid, step), "") if training_eligible else "",
            "createdAt": r.get("createdAt", ""),
        }
        row.update({k: int(round(v)) for k, v in knobs.items()})
        row.update(dict(zip(space.OBJECTIVE_FIELDS, scores)))
        row.update({field: r.get(field, "") for field in (
            "q_cognitive_load", "q_clear", "q_likability", "q_comfort",
            "q_pleasant", "q_distinct")})
        rows.append(row)

        haptics = r.get("haptics")
        rendered = haptics if isinstance(haptics, dict) else space.expand(knobs)
        for cue in space.CUES:
            burst = rendered.get(cue, {})
            long_rows.append({"pid": pid, "sessionId": row["sessionId"],
                              "resultId": row["resultId"], "phaseStep": step,
                              "questionnaireVersion": questionnaire_version,
                              "trainingEligible": training_eligible, "cue": cue,
                              **{k: burst.get(k, "") for k in space.BURST_KEYS}})
    return rows, long_rows, skipped, excluded_n


def main() -> None:
    if not PROJECT_ID:
        raise SystemExit(
            "Set GOOGLE_CLOUD_PROJECT, e.g.\n"
            "  GOOGLE_CLOUD_PROJECT=project-multinav python analysis/export_firestore.py"
        )
    from google.cloud import firestore

    db = firestore.Client(project=PROJECT_ID, database=DATABASE)
    print(f"project: {PROJECT_ID}  database: {DATABASE}")

    # Proposals, keyed BOTH by document id (the parameterDocumentId join) and by
    # (pid, phaseStep) (the fallback, and what a hand-seeded auto-id doc needs).
    proposals_by_id: dict[str, dict] = {}
    proposals_by_step: dict[tuple[str, int], dict] = {}
    for d in db.collection("parameterValues").stream():
        r = d.to_dict()
        if r.get("schemaVersion") != space.SCHEMA_VERSION:
            continue
        proposals_by_id[d.id] = r
        pid, step = r.get("pid"), space.round_number(r.get("phaseStep"))
        if pid is not None and step is not None:
            proposals_by_step.setdefault((str(pid), step), r)

    # Hypervolume lives in our own collection now — the service must not
    # annotate the app's result docs (their rules only allow a no-op retry).
    hv_by_step: dict[tuple[str, int], float] = {}
    for d in db.collection("moboMetrics").stream():
        r = d.to_dict()
        pid, step = r.get("pid"), space.round_number(r.get("phaseStep"))
        if (pid is not None and step is not None
                and r.get("hypervolume") is not None
                and r.get("questionnaireVersion") == QUESTIONNAIRE_VERSION):
            hv_by_step[(str(pid), step)] = r["hypervolume"]

    results = [(d.id, d.to_dict()) for d in db.collection("interventionResults").stream()]
    rows, long_rows, skipped, excluded_n = build_rows(results, proposals_by_id,
                                                      proposals_by_step, hv_by_step)

    rows.sort(key=lambda r: (r["pid"], r["phaseStep"]))
    long_rows.sort(key=lambda r: (r["pid"], r["phaseStep"], r["cue"]))

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(OUT_LONG, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LONG_COLUMNS)
        w.writeheader()
        w.writerows(long_rows)

    pids = sorted({r["pid"] for r in rows})
    print(f"\nwrote {len(rows)} rounds from {len(pids)} participants -> {OUT}")
    print(f"      {len(long_rows)} cue-rows -> {OUT_LONG}")
    print(f"  excluded (test pids): {excluded_n}   skipped (malformed): {skipped}")
    for pid in pids:
        n = sum(1 for r in rows if r["pid"] == pid)
        opt = sum(1 for r in rows if r["pid"] == pid and r["optimizerPhase"] == "mobo")
        retest = sum(1 for r in rows if r["pid"] == pid and r["optimizerPhase"] == "retest")
        print(f"    {pid:<24} {n:>3} rounds ({opt} model-driven, {retest} baseline re-test)")
    if any(not r["optimizerPhase"] for r in rows):
        print("\n  NOTE: some rounds have an empty `optimizerPhase` — their parameterValues\n"
              "  doc is missing or predates schemaVersion 2. evaluate_mobo.R falls back to\n"
              "  the round-number cutoffs (N_SOBOL, N_TOTAL - N_RETEST) for those; check\n"
              "  they are not the seeded round 1.")
    versions = {r["spaceVersion"] for r in rows if r["spaceVersion"]}
    if len(versions) > 1:
        print(f"\n  WARNING: rounds span multiple spaceVersions {sorted(versions)} — the knob\n"
              "  vectors are NOT comparable across versions. Split the analysis by version.")
    questionnaire_versions = {str(r["questionnaireVersion"] or "legacy") for r in rows}
    print(f"\nquestionnaire cohorts: {', '.join(sorted(questionnaire_versions))}; "
          f"{sum(r['trainingEligible'] for r in rows)} training-eligible rounds")
    print(f"evaluate_mobo.R defaults to questionnaire version {QUESTIONNAIRE_VERSION}; "
          "select legacy explicitly to analyse unversioned results.")


if __name__ == "__main__":
    main()
