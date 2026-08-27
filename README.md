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
- [Move to a new GCP project](#move-to-a-new-gcp-project)
- [Operate & verify](#operate--verify)
- [Troubleshooting](#troubleshooting)
- [Data contract with the app](#data-contract-with-the-app)
- [License & citing](#license--citing)

---

## What it does

Each participant runs a fixed number of trials. For every trial the service
picks the next stimulus configuration to test:

- **First `2·(D+1)` trials** — a scrambled **Sobol** sequence (space-filling
  exploration), where `D` is the number of parameters. (Deliberately one more
  than the study memo's `2n+1` rule, as a conservative buffer before the GP
  takes over.)
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
| `analysis/` | R evaluation of the study data: `export_firestore.py` (Firestore → tidy CSV) and `evaluate_mobo.R` (hypervolume, Pareto fronts, IGD+, APA reporting via `colleyRstats` + `moocore`). |
| `AGENTS.md` | Short orientation for AI coding agents: ground rules, the deploy model, and the traps. |
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
3. **Trial budget** — auto-derived: `N_SOBOL = 2·(D+1)`, `N_TOTAL = N_SOBOL + 5`.
   Override with the `N_SOBOL` / `N_MOBO` / `N_TOTAL` env vars.
4. **Hard constraints** — implement `is_feasible(raw)` (returns `False` for
   configs that must never be tested). Applied to Sobol draws and MOBO candidates.

No other file needs editing to change the study. `main.py` and `optimizer_core.py`
read the space generically from `space.py`.

> **Finalizing real ranges:** run `inspect_db.py` against your existing data to
> read actual value ranges and category lists, then set them in `space.py`.

### Changing parameters after deploy

Editing `space.py` and redeploying is safe **between participants**. Doing it
while participants are mid-study is not: `to_model_row()` reads *every* name in
`PARAM_NAMES` out of each stored doc, so a change to the space retroactively
reinterprets — or invalidates — history that is already in Firestore.

| Edit | Effect on a participant who is already mid-study |
|------|--------------------------------------------------|
| **Add, rename, or remove a parameter** | Every earlier doc now lacks (or gains) a field → `to_model_row()` raises `KeyError` → `load_observations()` skips it as malformed and the history collapses to 0 observations. `D` also changes, so `N_SOBOL = 2·(D+1)` and `N_TOTAL` change mid-study. |
| **Change a grid's `lo` / `hi`** | Docs still parse, but `normalize()` rescales them: values outside the new range map outside `[0, 1]` and the GP trains on out-of-bounds inputs without complaining. |
| **Change a step, `round_ndigits`, or a category list** | `obs_key()` identity changes → configurations that were already tested are no longer recognised as observed, and can be proposed a second time. |

> **⚠️ The stall.** The first row has a sharp failure mode worth knowing by
> sight. When the history collapses, `obs_count` drops, so the service computes
> `next_phase_step = obs_count + 1` — a step whose `parameterValues` doc
> **already exists**. The idempotency guard then returns
> `{"ok": true, "skipped": true}` on every later trial and the participant never
> receives another stimulus. In the Cloud Run logs it reads as `skipping
> malformed doc` followed by `already exists — skipping duplicate`.

**Safe procedure**

1. Edit `space.py`, then run the full local suite (`make test` / `.\tasks.ps1
   test`). `_validate()` and `tests/test_space.py` catch bad grids at import.
2. Tell the app team **before** deploying if a Firestore field name changed —
   the app must write the new field or every incoming doc is malformed.
3. Deploy when no participant is mid-study. If that is impossible, restart the
   affected participants explicitly: delete their `parameterValues/{pid}_step_*`
   and `interventionResults` docs, then re-create `users/{pid}`.
4. Check `/health` afterwards — it echoes the live `PARAM_NAMES`, `N_SOBOL` and
   `N_TOTAL`, which is the fastest confirmation that the space you edited is the
   space that is running.

If a study is already collecting data and you need a different space for a new
cohort, prefer deploying a **second Cloud Run service** (same code, different
`space.py`) over mutating the live one.

## Configuration

All optional; sensible defaults built in. Set on the Cloud Run service with
`--set-env-vars`.

| Env var | Default | Meaning |
|---------|---------|---------|
| `FIRESTORE_DATABASE` | `(default)` | Firestore database id to use |
| `N_SOBOL` | `2·(D+1)` | random exploration trials before MOBO |
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
python space.py                # print the resolved grids — eyeball the level counts
python tests/test_space.py     # fast unit tests (sub-second): grids, snapping, encoding
python simulate.py             # offline optimizer loop: on-grid, unique, hypervolume rises
python tests/test_service.py   # full HTTP service against an in-memory Firestore fake
```

`tests/test_space.py` is the quick guard while editing the search space;
`simulate.py` / `tests/test_service.py` exercise the GP and the full request loop.

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

### Prerequisites (one time)

| What | Why | How to check |
|------|-----|--------------|
| A GCP project with **billing enabled** | Cloud Run and Cloud Build refuse to run without it | Console → Billing: a billing account must be linked to the project |
| IAM: **Owner**, or `Cloud Run Admin` + `Cloud Functions Admin` + `Service Account User` | deploy rights | Console → IAM & Admin → IAM → find your account's Role column |
| **Google Cloud CLI** (`gcloud`) | every command below | [install guide](https://cloud.google.com/sdk/docs/install); open a *fresh* terminal after installing |
| **Node.js ≥ 20** (for `npx`) | deploys the Cloud Functions glue via `firebase-tools` | `node --version` |

Sign in **twice** — the CLI and the client libraries use separate credential
stores, and you'll need both:

```bash
gcloud auth login                       # used by the gcloud commands below
gcloud auth application-default login   # used by inspect_db.py and firebase-tools
```

> **Windows note:** the `$env:NAME = 'value'` syntax used by `tasks.ps1` docs is
> PowerShell-only. In cmd.exe use `set NAME=value` instead.

Pick a Cloud Run/Functions **region co-located with your Firestore database**
(e.g. a `nam5` database → `us-central1`) — Firestore triggers must run from a
region compatible with the database's location. If the database does not exist yet,
[Step 0](#step-0--the-firestore-database) creates it and settles the region.

If your project lists **more than one database**, make sure everything points at the
one the app actually writes: the service, the triggers, and `inspect_db.py`
all default to the special `(default)` database. A second, *named* database —
even one literally named `default` — is a completely separate store.

Enable the required APIs (once):

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com cloudfunctions.googleapis.com \
  eventarc.googleapis.com compute.googleapis.com firestore.googleapis.com \n  --project <PROJECT_ID>
```

### Step 0 — the Firestore database

Every later step assumes the database exists. A brand-new project has none —
`gcloud firestore databases list` prints `Listed 0 items` — and **its location is
permanent**: it cannot be changed afterwards, only recreated in another project.
Decide it first, because it also fixes the Cloud Run / Cloud Functions region for
the rest of the deploy.

```bash
gcloud firestore databases list --project <PROJECT_ID>            # already there?
gcloud firestore databases create --location=nam5 --project <PROJECT_ID>
```

`nam5` (US multi-region) pairs with `us-central1`, which is what
[`firebase/index.js`](firebase/index.js) pins in `setGlobalOptions`. Choose a
different location and you must change that line **and** every `--region` flag
below — Firestore triggers only run in a region compatible with the database.

If the app team already created a database, do **not** create another: read its
location and match it.

### Step 1 — optimizer → Cloud Run

Run from the repo root. Nothing is built locally: the source is uploaded
(trimmed by `.gcloudignore`) and Cloud Build builds the Dockerfile remotely.
The first build takes ~10–15 min (CPU torch is large); later deploys are faster.

```bash
make deploy PROJECT=<PROJECT_ID>     # Linux/macOS/Cloud Shell
```
```powershell
$env:PROJECT = '<PROJECT_ID>'        # Windows PowerShell
.\tasks.ps1 deploy
```

Both expand to the same explicit command:

```bash
gcloud run deploy vam-optimizer --source . --project <PROJECT_ID> \
  --region us-central1 --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 300
```

On the **first** source deploy in a fresh project, `gcloud` asks to create the
`cloud-run-source-deploy` Artifact Registry repository. Answer yes, or pass
`--quiet` to accept it automatically in a non-interactive shell.

Copy the printed **Service URL** and check `<service-url>/health` — it should
report the five parameter names, the database, and the trial budget.

### Step 2 — let the service read/write Firestore

Cloud Run runs as the project's default compute service account; grant it
Firestore access:

```bash
PROJECT_NUMBER=$(gcloud projects describe <PROJECT_ID> --format='value(projectNumber)')
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/datastore.user"
```

### Step 3 — Cloud Functions glue + Firestore indexes

The two Firestore triggers live in [`firebase/`](firebase/) as reference
copies. Assemble a standard Firebase folder anywhere (it does **not** need to
be inside this repo):

```
firebase-deploy/
├── firebase.json            ← see below
├── firestore.indexes.json   ← copy from firebase/  (READ THE WARNING BELOW)
└── functions/
    ├── index.js             ← copy from firebase/
    ├── package.json         ← copy from firebase/
    └── .env                 ← one line:  CLOUD_RUN_URL=<service-url from step 1>
```

> **⚠️ Index deploys are declarative.** `firestore:indexes` makes the database
> match the file: any composite index that exists on the database but is *not*
> listed in `firestore.indexes.json` is **deleted** — including indexes your
> app team created by hand for the app's own queries. Before the first deploy,
> list what's live and merge it into the file:
>
> ```bash
> gcloud firestore indexes composite list --project <PROJECT_ID> --database="(default)"
> ```

Install the functions' dependencies once — `firebase-tools` loads the code
locally to discover the triggers, so deploy fails without them:

```bash
cd firebase-deploy/functions && npm install && cd ..
```

That discovery step has a 10 s budget and misses it on cold or Windows machines,
failing with `Cannot determine backend specification. Timeout after 10000`. It is
not a code error — raise the budget:

```bash
export FUNCTIONS_DISCOVERY_TIMEOUT=120     # PowerShell: $env:FUNCTIONS_DISCOVERY_TIMEOUT = '120'
```

`firebase.json` — the `firestore` block is **required**; without it the
`firestore:indexes` target is *silently skipped*:

```json
{
  "functions": { "source": "functions" },
  "firestore": { "indexes": "firestore.indexes.json" }
}
```

There is deliberately no `"rules"` key: **never blind-deploy
`firestore:rules`** — it would overwrite the app's live security rules.

Deploy (no `firebase login` needed — `firebase-tools` falls back to the
application-default credentials from the prerequisites):

```bash
npx firebase-tools deploy --only functions,firestore:indexes --project <PROJECT_ID>
```

If the very first functions deploy fails with **`Permission denied while using
the Eventarc Service Agent`**, that is a first-use propagation delay, not a real
failure — the agent already holds `roles/eventarc.serviceAgent`, the grant just
hasn't reached Eventarc yet. Verify with:

```bash
gcloud projects get-iam-policy <PROJECT_ID>   --flatten='bindings[].members' --filter='bindings.members:gcp-sa-eventarc'   --format='value(bindings.members,bindings.role)'
```

then wait a few minutes and rerun. The indexes deploy *succeeds* in that same
run, so the rerun only needs `--only functions`.

One more non-zero exit to expect on a fresh project: after the functions
deploy cleanly, `firebase-tools` exits `1` because it could not set an
Artifact Registry **cleanup policy** without a prompt. The functions are fine—
old container images would just accumulate. Set it once:

```bash
npx firebase-tools functions:artifacts:setpolicy --project <PROJECT_ID> --force
```

If your database is *named* rather than `(default)`, set `database: "<name>"`
in each trigger's options in `index.js` first (see `firebase/README.md`).

### Step 4 — verify end-to-end

See [Operate & verify](#operate--verify) below: hit `/health`, create a test
`users/{uid}` doc and watch `parameterValues/{uid}_step_1` appear, then write a
fake `interventionResults` doc for that uid and watch `step_2` appear. Use a
clearly-fake uid and clear the test docs before the real study starts.

## Move to a new GCP project

Projects get replaced for reasons that have nothing to do with the code: a
billing account moves, the institution requires the project to live under its
GCP **organization**, an IRB wants a fresh data home. **Nothing in this
repository is project-specific** — there is no project id, service URL, or
credential anywhere in it; everything arrives at deploy time via `PROJECT` /
`--project` / `GOOGLE_CLOUD_PROJECT`.

**So the whole procedure is: re-run [Deploy](#deploy) Steps 0–4 against the new
project id.** The only code-coupled value is the region in
[`firebase/index.js`](firebase/index.js), and only if the new Firestore location
differs from the old one.

### What does not come along

Project state is not portable. Each of these must be recreated, and a skipped
row is the usual reason for "it deployed but nothing happens":

| Thing | Who recreates it | Note |
|-------|------------------|------|
| Firestore database | you — [Step 0](#step-0--the-firestore-database) | location is permanent; matching the old one avoids code changes |
| Enabled APIs | you — [Prerequisites](#prerequisites-one-time) | `run`, `cloudbuild`, `artifactregistry`, `cloudfunctions`, `eventarc`, `compute` |
| Cloud Run service **and its URL** | you — [Step 1](#step-1--optimizer--cloud-run) | the URL embeds the project number, so it always changes |
| `roles/datastore.user` on the compute SA | you — [Step 2](#step-2--let-the-service-readwrite-firestore) | the SA is `<NEW_PROJECT_NUMBER>-compute@developer.gserviceaccount.com` |
| Cloud Functions + `functions/.env` | you — [Step 3](#step-3--cloud-functions-glue--firestore-indexes) | `.env` must carry the **new** `CLOUD_RUN_URL` |
| Composite indexes | you — [Step 3](#step-3--cloud-functions-glue--firestore-indexes) | deploy them *before* the app team hand-creates theirs, and re-read the declarative warning there |
| Registered app + `google-services.json` / `GoogleService-Info.plist` | **app team** | the Firebase console shows "There are no apps in your project" until they do |
| Firestore **security rules** | **app team** | a new database starts locked; this repo deliberately never deploys rules |
| Auth sign-in methods (e.g. anonymous) | **app team** | console settings do not migrate |
| Existing study data | nobody, unless you export it | see below |

Until the app team's half lands, the optimizer deploys cleanly and simply never
receives a trial — a healthy `/health` proves nothing about the app side.

### Org-policy gotchas

A personal project has no org policies; an org-owned one usually does, so these
appear on the *first* deploy after a move and not before.

| Constraint | Symptom | Fix |
|-----------|---------|-----|
| `iam.allowedPolicyMemberDomains` | `gcloud run deploy --allow-unauthenticated` fails while adding `allUsers` | ask the org admin for a project-level exception, or drop the flag and instead grant the functions' service account `roles/run.invoker` and send an ID token from `index.js` |
| `iam.automaticIamGrantsForDefaultServiceAccounts` | the default compute service account exists with no roles | harmless here — [Step 2](#step-2--let-the-service-readwrite-firestore) grants `roles/datastore.user` explicitly anyway |

### Leaving the old project behind

Disabling billing does **not** delete Firestore data, but it does make it
unreachable — every client call returns `403 This API method requires billing to
be enabled`. Export anything you might want *before* the billing account goes
away:

```bash
gcloud firestore export gs://<BUCKET> --project <OLD_PROJECT_ID>
```

Then point nothing at the old project. Do not park a billing-disabled project
indefinitely on the assumption that the data is safe there.

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
| `firebase deploy` fails with `User code failed to load. Cannot determine backend specification. Timeout after 10000` | Not a code error — `firebase-tools` loads `functions/` in a subprocess to discover the triggers and gives it 10 s, which Windows/cold-cache machines routinely miss. Confirm the code is fine with `node -e "require('./index.js')"` in `functions/`, then re-run with `FUNCTIONS_DISCOVERY_TIMEOUT=120` set. Also check `functions/package.json`'s `engines.node` matches your local `node --version`. |
| Participant stops receiving stimuli; logs show `skipping malformed doc` then `already exists — skipping duplicate` | `space.py` changed while they were mid-study — their history no longer parses. See [Changing parameters after deploy](#changing-parameters-after-deploy). |
| Every Firestore call returns `403 This API method requires billing to be enabled` | Billing is disabled on that project. The data is retained but unreadable until billing is restored. |
| `--allow-unauthenticated` rejected on deploy | Org policy `iam.allowedPolicyMemberDomains` blocks `allUsers` — see [Org-policy gotchas](#org-policy-gotchas). |
| Slow MOBO step / timeout | Lower `RAW_SAMPLES`/`NUM_RESTARTS`/`MC_SAMPLES`; ensure the image built the fast EHVI kernel (needs `build-essential`+`ninja`, already in the Dockerfile). |

## Data contract with the app

- App writes `interventionResults/{id}` with `pid`, `phaseStep`,
  `attentionCheckPassed`, the objective fields, **and every field in
  `space.PARAM_NAMES`** (the exact values it rendered).
- Service writes `parameterValues/{pid}_step_N` with the next config (same field
  names) → the app's snapshot listener applies it.
- Study ends at `N_TOTAL` observations → the service sets
  `users/{pid}.studyCompleted = true`.

### Parameter fields (VAM study)

Ranges marked ✅ are confirmed (app team 2026-07-21, or the study memo). The
remaining optimizer-side choices — the intensity grid floor and the sharpness
step — are flagged in `space.py`.

| Field | Type | Unit / values | Range | Grid |
|-------|------|---------------|-------|------|
| `intensity` | double | amplitude | ✅ 0–1 (memo; grid floor 0.20 is an optimizer choice) | Weber, 13 % (14 levels) |
| `sharpness` | double | — | ✅ 0–1 (memo; JND step pending pilot) | linear, step 0.20 (6 levels) |
| `duration`  | double | seconds (length of each pulse) | ✅ 0.03 – 20 s (app) | linear, step 0.240 (84 levels) |
| `interval`  | double | **Hz** (pulse-set rate) | ✅ 1 – 20 Hz (app) | Weber, 20 % (17 levels) |
| `pattern`   | string | `"constant"` \| `"puls"` | ✅ | categorical |

**Canonical form:** for `pattern == "constant"` (a sustained vibration) the
pulse-shaping fields don't apply: the app *ignores* `interval` (stores the
minimum, 1 Hz), and `duration` is conceptually infinite — the cue plays for as
long as the interaction lasts. `space.canonicalize()` bakes both in: every
proposed/encoded/deduplicated `constant` config has `interval == 1.0` and
`duration == 19.95` (the grid max, "as long as possible" — correct whether the
app ignores duration or plays it to the end). `constant` therefore reduces to
intensity × sharpness, and the optimizer never spends trials varying fields
that have no effect.

**Puls feasibility:** the app floors the off-time between pulses at 10 ms
(`off = max(0.01, 1/interval − duration)`); past that floor the pulses still
play full-length and `interval` loses its effect. `is_feasible()` therefore
blocks `duration > 1/interval − 0.01` for `puls`, and MOBO proposals in that
region are minimally repaired by `project_feasible()` (duration lowered to the
largest grid level that fits the period).

## License & citing

MIT — see [LICENSE](LICENSE). If you use this in academic work, please cite it;
machine-readable metadata is in [CITATION.cff](CITATION.cff) (a paper reference
will be added there once published).
