# AGENTS.md

Orientation for AI coding agents working in this repository. The full
documentation is [README.md](README.md); this file is the short version plus the
traps that are expensive to rediscover.

## What this is

A stateless **multi-objective Bayesian optimization (MOBO)** service on Cloud
Run that personalizes stimulus parameters per participant during a user study.
It never calls the app and the app never calls it — everything goes through
Firestore:

```
app  →  interventionResults/{id}  →  Cloud Function  →  POST /updatePolicy  →  Cloud Run
Cloud Run  →  parameterValues/{pid}_step_N  →  app's snapshot listener
```

First `2·(D+1)` trials are Sobol exploration, the rest are GP `qLogNEHVI`. Every
proposal is snapped onto a **JND-spaced grid** so no two tested settings are
closer than a just-noticeable difference.

## Ground rules

1. **`space.py` is the only file you normally edit.** Parameters, grids,
   feasibility, canonical forms, objectives. `main.py` and `optimizer_core.py`
   read the space generically. If you are editing them to add a parameter, stop —
   you are solving it in the wrong place.
2. **Validate locally before claiming anything works.** No cloud needed:
   ```bash
   python space.py              # print resolved grids — eyeball the level counts
   python tests/test_space.py   # sub-second search-space unit tests
   python simulate.py           # offline loop: on-grid, unique, hypervolume rises
   python tests/test_service.py # full HTTP service vs an in-memory Firestore fake
   ```
   Or `make test` / `.\tasks.ps1 test` for all four. `_validate()` runs on import
   and rejects duplicate names, non-ascending grids and empty category lists, so
   a bad edit fails loudly at import, not deep inside the GP at request time.
3. **`N_SOBOL = 2·(D+1)` is a deliberate decision**, one more than the study
   memo's `2n+1`. It is not an off-by-one bug — do not "fix" it.
4. **Never deploy `firestore:rules`.** The app team owns the live security rules;
   a blind deploy overwrites them. `firebase.json` here has no `"rules"` key on
   purpose.
5. **`firebase deploy --only firestore:indexes` is declarative.** Any composite
   index on the database that is *not* in `firestore.indexes.json` gets
   **deleted** — including ones the app team created by hand. List the live
   indexes and merge them into the file first. This has already bitten this
   project once.
6. **Don't commit or push unless asked.** The working tree may already be
   committed mid-session by the human.

## Deploy model

Nothing builds locally. `gcloud run deploy --source .` uploads the tree (trimmed
by `.gcloudignore`) and Cloud Build builds the Dockerfile remotely; the first
build in a fresh project takes ~10–15 min because CPU torch is large.

The repo is **project-agnostic**: there is no project id, service URL, or
credential anywhere in it. Everything arrives at deploy time via `PROJECT` /
`--project` / `GOOGLE_CLOUD_PROJECT`. Moving to a new GCP project is therefore a
re-run of Deploy Steps 0–4, not a code change — see
[Move to a new GCP project](README.md#move-to-a-new-gcp-project) for the full
runbook and the list of project state that does *not* migrate.

The one code-coupled cloud value is the **region**, pinned in
`firebase/index.js` (`setGlobalOptions`). It must stay compatible with the
Firestore location (`nam5` → `us-central1`).

### Current deployment

| | |
|---|---|
| GCP project | `project-multinav` (org-owned, billing enabled) |
| Firestore | `(default)`, **nam5**, Native mode |
| Region | `us-central1` (Cloud Run + Cloud Functions) |
| Cloud Run service | `vam-optimizer` |

Get the live URL rather than hardcoding it:

```bash
gcloud run services describe vam-optimizer --region us-central1 \
  --project project-multinav --format='value(status.url)'
```

`GET <url>/health` echoes the live `PARAM_NAMES`, database and trial budget —
the fastest check that the space you think you deployed is the one running.

> The predecessor project `multinav-a6ade` is retired and its billing is
> disabled, which makes its Firestore unreadable (`403 … requires billing`). Do
> not point anything at it.

## Firestore contract

| Collection | Written by | Must contain |
|-----------|-----------|--------------|
| `users/{pid}` | app | creation triggers `registerUserOnCreate` → step 1 |
| `interventionResults/{id}` | app | `pid`, `phaseStep`, `attentionCheckPassed`, every field in `space.OBJECTIVE_FIELDS`, **and every field in `space.PARAM_NAMES`** |
| `parameterValues/{pid}_step_N` | this service | the next config, same field names |

`users/{pid}.studyCompleted = true` is set once `N_TOTAL` observations exist.

## Traps

- **A parameter change mid-study stalls participants.** `to_model_row()` reads
  every name in `PARAM_NAMES` out of each stored doc, so adding/renaming/removing
  one makes all earlier docs raise `KeyError` → silently skipped as malformed →
  `obs_count` collapses → the service computes a `next_phase_step` whose
  `parameterValues` doc already exists → the idempotency guard returns
  `{"skipped": true}` forever and the participant never gets another stimulus.
  Log signature: `skipping malformed doc` followed by `already exists — skipping
  duplicate`. Change the space **between** participants. Full table in
  [README](README.md#changing-parameters-after-deploy).
- **Malformed docs fail silently by design.** `load_observations()` skips them
  with a warning so one bad doc can't take down a participant's whole update —
  which also means a broken contract looks like "the optimizer just isn't
  learning". Check the Cloud Run logs for `skipping malformed doc`.
- **`(default)` vs a database literally named `default`** are different stores.
  The service, the triggers and `inspect_db.py` all use `(default)`. A named
  database needs `database:` on each Firestore trigger in `index.js` and
  `FIRESTORE_DATABASE` on the service.
- **Dead dimensions are canonicalized, not optimized.** For
  `pattern == "constant"` the app ignores `interval` and `duration` is
  conceptually infinite, so `canonicalize()` pins them (`1.0` Hz,
  `CONSTANT_DURATION` = grid max). Don't "fix" the sentinel to the grid minimum
  without the app team confirming the app ignores duration for constant.
- **`interval` is in Hz**, not seconds, and `duration` is the length of *each
  pulse*. `is_feasible()` blocks `duration > 1/interval − 0.01` for `puls`
  because the app floors the off-time at 10 ms.

## Open questions

The parameter set is **not final**. Two params (pulse count 1–4, interval within
a pulse set) are planned in the app but not yet written to Firestore; do not add
them to `space.py` until the app actually writes the fields, or every doc becomes
malformed. Adding them moves `D` 5 → 7, and the budget 12+5 → 16+5. The
sharpness JND step is a placeholder pending pilot data. Ranges flagged `TODO` in
`space.py` are the authoritative list of what is still open.

Use `inspect_db.py` (read-only; needs `GOOGLE_CLOUD_PROJECT` and ADC) to check
what the app is actually writing before changing the contract.
