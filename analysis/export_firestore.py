"""
Export the study data from Firestore to a tidy CSV for the R analysis.

Read-only: it never writes, updates or deletes anything.

Auth (either one):
    # a) service-account key file
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    # b) gcloud application-default login
    gcloud auth application-default login

Run:
    GOOGLE_CLOUD_PROJECT=project-multinav python analysis/export_firestore.py
    # -> analysis/trials.csv

One row per completed trial, joining what the app wrote (interventionResults)
with what the optimizer proposed (parameterValues, for the `phase` label).
Columns are exactly what analysis/evaluate_mobo.R expects.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import firestore  # noqa: E402

import space  # noqa: E402

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise SystemExit(
        "Set GOOGLE_CLOUD_PROJECT, e.g.\n"
        "  GOOGLE_CLOUD_PROJECT=project-multinav python analysis/export_firestore.py"
    )
DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")
OUT = os.environ.get("OUT_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "trials.csv"))

# Test participants to leave out of the analysis. Extend as needed.
EXCLUDE_PREFIXES = ("claude_", "test_", "debug_")

COLUMNS = (["pid", "phaseStep", "phase", "attentionCheckPassed"]
           + space.PARAM_NAMES
           + space.OBJECTIVE_FIELDS
           + ["hypervolume", "createdAt"])


def excluded(pid: str) -> bool:
    return not pid or pid.startswith(EXCLUDE_PREFIXES)


def main() -> None:
    db = firestore.Client(project=PROJECT_ID, database=DATABASE)
    print(f"project: {PROJECT_ID}  database: {DATABASE}")

    # phase label lives on the optimizer's proposal, keyed "{pid}_step_{n}"
    phases: dict[tuple[str, int], str] = {}
    for d in db.collection("parameterValues").stream():
        r = d.to_dict()
        pid, step = r.get("pid"), r.get("phaseStep")
        if pid is not None and step is not None:
            phases[(pid, int(step))] = r.get("phase", "")

    rows, skipped, excluded_n = [], 0, 0
    for d in db.collection("interventionResults").stream():
        r = d.to_dict()
        pid, step = r.get("pid"), r.get("phaseStep")
        if excluded(pid):
            excluded_n += 1
            continue
        missing = [f for f in space.PARAM_NAMES + space.OBJECTIVE_FIELDS if f not in r]
        if pid is None or step is None or missing:
            skipped += 1
            print(f"  skipping {pid} step={step}: missing {missing or 'pid/phaseStep'}")
            continue
        row = {
            "pid": pid,
            "phaseStep": int(step),
            "phase": phases.get((pid, int(step)), ""),
            # absent == passed: the app only writes False on an actual failure
            "attentionCheckPassed": r.get("attentionCheckPassed", True),
            "hypervolume": r.get("hypervolume", ""),
            "createdAt": r.get("createdAt", ""),
        }
        row.update({f: r[f] for f in space.PARAM_NAMES + space.OBJECTIVE_FIELDS})
        rows.append(row)

    rows.sort(key=lambda r: (r["pid"], r["phaseStep"]))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    pids = sorted({r["pid"] for r in rows})
    print(f"\nwrote {len(rows)} trials from {len(pids)} participants -> {OUT}")
    print(f"  excluded (test pids): {excluded_n}   skipped (malformed): {skipped}")
    for pid in pids:
        n = sum(1 for r in rows if r["pid"] == pid)
        opt = sum(1 for r in rows if r["pid"] == pid and r["phase"] == "optimization")
        print(f"    {pid:<24} {n:>3} trials ({opt} optimization)")
    if any(not r["phase"] for r in rows):
        print("\n  NOTE: some trials have an empty `phase` — their parameterValues doc is\n"
              "  missing. evaluate_mobo.R falls back to the N_SOBOL cutoff for those.")


if __name__ == "__main__":
    main()
