# MOBO optimizer service

A small, self-contained **multi-objective Bayesian optimization (MOBO)** service
that personalizes stimulus parameters per participant during a study. Built for
a *Vibrotactile Accessible Maps* (VAM) study, but written so it can be reused
for any study that wants per-participant parameter optimization over a
Firestore-backed app.

It runs on **Cloud Run**, talks to the app only through **Firestore**, and
encodes perceptual **just-noticeable differences (JNDs)** so it never proposes
two settings a participant couldn't tell apart.

## Quick start (no cloud needed)

See the optimizer converge on a simulated participant in two commands:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python simulate.py     # full study loop: Sobol → MOBO, hypervolume rising
```

---

## Contents
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [The method (and why not Optuna)](#the-method)
- [Reuse: adapting it to your study](#reuse-adapting-it-to-your-study)
- [Configuration (env vars)](#configuration)
- [Local development & validation](#local-development--validation)
- [Deploy](#deploy)
- [Operate & verify](#operate--verify)
- [Troubleshooting](#troubleshooting)
- [Data contract with the app](#data-contract-with-the-app)
- [License & citing](#license--citing)

---

## What it does

Each participant runs a fixed number of trials. For every trial the service
picks the next stimulus configuration to test:

- **First `2·D + 1` trials** — a scrambled **Sobol** sequence (space-filling
  exploration), where `D` is the number of parameters.
- **Remaining trials** — **GP-based MOBO** (`qLogNEHVI`), which fits a Gaussian
  process to the two objective scores and proposes the configuration expected to
  most improve the Pareto front.

Every proposed configuration is **snapped onto a JND-spaced grid**, so no two
tested settings are closer than one just-noticeable difference.

## Architecture

The app and the optimizer never call each other directly — everything is
mediated by Firestore, so the app stays simple and the optimizer stays stateless.

```
 ┌────────┐  writes result   ┌───────────────────┐  onCreate   ┌──────────────────┐
 │  App   │ ───────────────▶ │  interventionResults │ ─────────▶ │  Cloud Function  │
 │(device)│                  │    (Firestore)       │            │ updatePolicyOn…  │
 └────────┘                  └───────────────────┘            └────────┬─────────┘
     ▲                                                           POST /updatePolicy
     │ snapshot listener                                                │
     │ on newest params                                                 ▼
 ┌───┴────────────────┐   writes next config   ┌───────────────────────────────────┐
 │  parameterValues   │ ◀───────────────────── │  Cloud Run: this service (BoTorch) │
 │    (Firestore)     │                        │  reads history → picks next config │
 └────────────────────┘                        └───────────────────────────────────┘
```

1. App writes a completed trial to `interventionResults` (the params it rendered
   + the objective scores + an attention-check flag).
2. A Cloud Function fires and `POST`s to the service's `/updatePolicy`.
3. The service reads that participant's full history straight from Firestore,
   computes the next configuration, and writes it to `parameterValues`.
4. The app's snapshot listener picks up the new config and renders it.

Registration: creating a `users/{uid}` doc fires `registerUserOnCreate` →
`POST /registerUser` → the service writes `step_1` (the Sobol seed).

## Repository layout

| File | Role |
|------|------|
| **`space.py`** | **The one file you edit to adapt the study.** Parameters, JND grids, encoding, hard-constraint hook, fail-fast validation. Torch-free. |
| `optimizer_core.py` | Pure GP + `qLogNEHVI` candidate selection. No Firestore — unit-testable. |
| `main.py` | Flask service: Firestore I/O, idempotency, hypervolume logging, HTTP endpoints. |
| `simulate.py` | Offline optimizer loop against a synthetic participant (asserts on-grid, unique, HV rises). |
| `tests/test_service.py` | Full HTTP service against an in-memory Firestore fake (register → N trials → done, dedup, attention-check exclusion). |
| `tests/test_space.py` | Fast unit tests for the search-space logic (no GP fitting; sub-second). |
| `inspect_db.py` | Read-only Firestore inspector — extracts real parameter ranges/categories to finalize `space.py`. |
| `firebase/` | Reference copies of the Cloud Functions glue + Firestore indexes (see its README). |
| `Makefile` / `tasks.ps1` | Task runners for the common validate/deploy commands (Linux / Windows). |
| `requirements.txt` / `Dockerfile` | Pinned deps (incl. `torch==2.13.0`); CPU-only torch; JIT-compiles the fast EHVI kernel. |
| `.github/workflows/ci.yml` | CI: compile + the three test scripts on every push/PR. |
| `LICENSE` | MIT license. |

## The method

**JND = perceptual resolution, not a hard constraint.** Two configs closer than
a JND are indistinguishable, so testing both wastes a trial. We encode this by
**discretizing each continuous parameter onto a perceptually-spaced grid**:

| Grid type | For | Spacing |
|-----------|-----|---------|
| `weber_grid` (geometric) | Weber's-law params (intensity 13 %, interval 20 %) | ×(1 + jnd) |
| `linear_grid` (additive) | absolute-JND params (duration, seconds) | fixed step |
| `list_grid` (explicit) | naturally discrete params (e.g. pulse count) | given levels |
| categorical | nominal params (pattern) | `MixedSingleTaskGP` |

We do **not** enumerate the full grid (~10⁵–10⁶ combos). The GP optimizes in
continuous space (sample-efficient) and the winner is **snapped** to the grid
before it's written — same "never sub-JND" guarantee, but scalable.

**Why GP MOBO and not Optuna/TPE:** with ~15–20 trials per participant, GP
`qLogNEHVI` is far more sample-efficient — it adapts right after the Sobol seed,
whereas TPE needs many random startup trials before it does anything smart. And
a JND is a *grid-resolution* question (discretize), not an *infeasibility*
question (prune) — so the Optuna `TrialPruned` pattern is the wrong tool for it.
Hard constraints, when you have them, go in `is_feasible()`.

## Reuse: adapting it to your study

Everything study-specific lives in **`space.py`**. To repurpose the service:

1. **Parameters** — edit `CONT_PARAMS` / `CAT_PARAMS`. Each `name` must match the
   Firestore field the app reads/writes. Choose a grid builder per parameter:
   ```python
   CONT_PARAMS = [
       ContinuousParam("intensity", weber_grid(0.2, 1.0, 0.13)),   # Weber JND
       ContinuousParam("duration",  linear_grid(0.06, 1.0, 0.24)), # absolute JND
       ContinuousParam("pulseCount", list_grid([1, 2, 3, 4])),     # discrete
   ]
   CAT_PARAMS = [ CategoricalParam("pattern", ["a", "b", "c"]) ]   # or [] for none
   ```
   `_validate()` runs on import and rejects duplicate names, empty/​non-ascending
   grids, and empty category lists — so a bad edit fails immediately with a clear
   message, not deep inside the GP at request time.
2. **Objectives** — `OBJECTIVE_FIELDS` (default `["subjectiveScore",
   "objectiveScore"]`, both maximized, in `[0,1]`). Add a third for 3 objectives;
   everything downstream adapts automatically.
3. **Trial budget** — auto-derived: `N_SOBOL = 2·D + 1`, `N_TOTAL = N_SOBOL + 5`.
   Override with the `N_SOBOL` / `N_MOBO` / `N_TOTAL` env vars.
4. **Hard constraints** — implement `is_feasible(raw)` (returns `False` for
   configs that must never be tested). Applied to Sobol draws and MOBO candidates.

No other file needs editing to change the study. `main.py` and `optimizer_core.py`
read the space generically from `space.py`.

> **Finalizing real ranges:** run `inspect_db.py` against your existing data to
> read actual value ranges and category lists, then set them in `space.py`.

## Configuration

All optional; sensible defaults built in. Set on the Cloud Run service with
`--set-env-vars`.

| Env var | Default | Meaning |
|---------|---------|---------|
| `FIRESTORE_DATABASE` | `(default)` | Firestore database id to use |
| `N_SOBOL` | `2·D + 1` | random exploration trials before MOBO |
| `N_MOBO` | `5` | MOBO trials after exploration |
| `N_TOTAL` | `N_SOBOL + N_MOBO` | total trials before `studyCompleted` |
| `NUM_RESTARTS` | `5` | acqf optimizer restarts |
| `RAW_SAMPLES` | `256` | acqf raw samples |
| `MC_SAMPLES` | `64` | MC samples for the hypervolume estimate |

The acqf knobs are deliberately light: `optimize_acqf_mixed` runs once per
categorical combination, so cost scales with the number of categories. These
values keep a single `/updatePolicy` call well under the Cloud Run 300 s timeout.

## Local development & validation

No cloud needed. Run these after any edit to `space.py`:

```bash
python space.py       # print the resolved grids — eyeball the level counts
python test_space.py  # fast unit tests (sub-second): grids, snapping, encoding
python simulate.py        # offline optimizer loop: on-grid, unique, hypervolume rises
python test_service.py    # full HTTP service against an in-memory Firestore fake
```

`test_space.py` is the quick guard while editing the search space;
`simulate.py` / `test_service.py` exercise the GP and the full request loop.

Or use the task runner (`make` on Linux/Cloud Shell, `tasks.ps1` on Windows):

```bash
make test                            # compile + all three test scripts
.\tasks.ps1 test                     # same, on Windows PowerShell
make deploy PROJECT=<gcp-project-id> # gcloud run deploy (see below)
```

**CI:** `.github/workflows/ci.yml` runs the same compile + three test scripts on
every push/PR. Dependabot (`.github/dependabot.yml`) keeps the pip pins and
action versions fresh monthly.

## Deploy

Pick a Cloud Run **region co-located with your Firestore database** (e.g. a
`nam5` database → `us-central1`) — Firestore triggers must run from a region
compatible with the database's location.

Prereqs (one time): billing enabled, and the APIs `run`, `cloudbuild`,
`artifactregistry`, `cloudfunctions`, `eventarc`, `firestore` enabled:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com cloudfunctions.googleapis.com \
  eventarc.googleapis.com firestore.googleapis.com --project <PROJECT_ID>
```

```bash
# 1. Optimizer → Cloud Run
make deploy PROJECT=<PROJECT_ID>          # or the explicit command:
# gcloud run deploy vam-optimizer --source . --project <PROJECT_ID> \
#   --region us-central1 --allow-unauthenticated \
#   --memory 2Gi --cpu 2 --timeout 300
# → copy the printed Service URL

# 2. Deploy the Cloud Functions glue wired to that URL (see firebase/README.md)

# 3. Let Cloud Run's service account read/write Firestore
PROJECT_NUMBER=$(gcloud projects describe <PROJECT_ID> --format='value(projectNumber)')
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/datastore.user"
```

**Don't blind-deploy `firestore:rules`** — that would overwrite your app's live
security rules. Deploy rules separately, only after reviewing them.

## Operate & verify

```bash
curl https://vam-optimizer-XXXXX.us-central1.run.app/health   # params, database, budget
gcloud run services logs read vam-optimizer --region us-central1 --limit 30
firebase functions:log --only updatePolicyOnResult
```
Then create a test `users/{id}` doc and watch `parameterValues/{id}_step_1` appear.

Concurrency & idempotency are handled: a per-user lock serializes same-user
requests, `parameterValues` docs are written with `create()` (first-wins), and
duplicate Cloud Function deliveries are detected and skipped. Failed attention
checks (`attentionCheckPassed == false`) are excluded from the training data.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Cloud Function never fires | Trigger is on the wrong database. It must match the app's DB (default `(default)`); a **named** DB needs `database:` set on the trigger. |
| `/updatePolicy` 500s | Check Cloud Run logs. A single malformed doc is skipped with a warning, not fatal. |
| Optimizer proposes a value the app can't render | A `CONT_PARAMS` range or `CAT_PARAMS` category doesn't match the app — reconcile via `inspect_db.py`. |
| Deploy/queries need a Firestore index | `firebase deploy --only firestore:indexes` (the `pid`+`phaseStep` composite indexes are in `firestore.indexes.json`). |
| Slow MOBO step / timeout | Lower `RAW_SAMPLES`/`NUM_RESTARTS`/`MC_SAMPLES`; ensure the image built the fast EHVI kernel (needs `build-essential`+`ninja`, already in the Dockerfile). |

## Data contract with the app

- App writes `interventionResults/{id}` with `pid`, `phaseStep`,
  `attentionCheckPassed`, the objective fields, **and every field in
  `space.PARAM_NAMES`** (the exact values it rendered).
- Service writes `parameterValues/{pid}_step_N` with the next config (same field
  names) → the app's snapshot listener applies it.
- Study ends at `N_TOTAL` observations → the service sets
  `users/{pid}.studyCompleted = true`.

## License & citing

MIT — see [LICENSE](LICENSE). If you use this in academic work, please cite it;
machine-readable metadata is in [CITATION.cff](CITATION.cff) (a paper reference
will be added there once published).
