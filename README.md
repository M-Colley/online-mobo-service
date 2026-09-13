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
python simulate.py     # full 18-round loop: anchor → Sobol → MOBO, HV rising
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

Each participant runs a fixed number of **rounds** (18 for the current study —
the app's security rules cap `roundNumber` there). For every round the service
picks the next configuration to test:

- **Round 1** — the **anchor**: a fixed expert baseline design, identical for
  every participant. It is the origin of the search space and the natural
  within-participant control.
- **Rounds 2…`2·(D+1)`+1** — a scrambled **Sobol** sequence (space-filling
  exploration), where `D` is the number of optimized dimensions. (Deliberately
  one more than the study memo's `2n+1` rule, as a conservative buffer.)
- **Remaining rounds** — **GP-based MOBO** (`qLogNEHVI`), which fits a Gaussian
  process to the two objective scores and proposes the configuration expected to
  most improve the Pareto front.

Every proposal lies on a **JND-spaced grid**, so no two tested settings are
closer than one just-noticeable difference.

> **What a "configuration" is here.** For the VAM/multinav study a candidate is a
> whole haptic *design*: 14 navigation cues × 5 burst values = 70 numbers. With
> at most 18 session-level score pairs those 70 numbers are unidentifiable, so
> the optimizer searches **4 integer JND knobs** that scale the expert baseline
> and expand deterministically to all 70. See
> [Data contract with the app](#data-contract-with-the-app).

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
`POST /registerUser` → the service writes `step_1` (the anchor design). Note the
live security rules deny client writes to `users/`, so in this study that doc is
created by an admin, not by the app.

## Repository layout

| File | Role |
|------|------|
| **`space.py`** | **The one file you edit to adapt the study.** Parameters, JND grids, encoding, hard-constraint hook, fail-fast validation. Torch-free. |
| `optimizer_core.py` | Pure GP + `qLogNEHVI` candidate selection. No Firestore — unit-testable. |
| `main.py` | Flask service: Firestore I/O, idempotency, hypervolume logging, HTTP endpoints. |
| `simulate.py` | Offline optimizer loop against a synthetic participant (asserts on-grid, unique, HV rises). |
| `tests/test_service.py` | Full HTTP service against an in-memory Firestore fake (register → N trials → done, dedup, attention-check exclusion). |
| `tests/test_space.py` | Unit tests for the search-space logic (no GP fitting; ~5 s, including an exhaustive pass of every reachable design against the app's rules). |
| `inspect_db.py` | Read-only Firestore inspector — document shape, observed ranges, every burst against the rules mirror, and the app's own listener query. **Run it first** whenever the app team says the parameters changed. |
| `firebase/` | Reference copies of the Cloud Functions glue + Firestore indexes (see its README). |
| `Makefile` / `tasks.ps1` | Task runners for the common validate/deploy commands (Linux / Windows). |
| `requirements.txt` / `Dockerfile` | Pinned deps (incl. `torch==2.13.0`); CPU-only torch; JIT-compiles the fast EHVI kernel. |
| `.github/workflows/ci.yml` | CI: compile + the three test scripts on every push/PR. |
| `analysis/` | R evaluation of the study data: `export_firestore.py` (Firestore → two tidy CSVs — one row per round, one row per round×cue) and `evaluate_mobo.R` (hypervolume, Pareto fronts, IGD+, APA reporting via `colleyRstats` + `moocore`). |
| `AGENTS.md` | Short orientation for AI coding agents: ground rules, the deploy model, and the traps. |
| `LICENSE` | MIT license. |

## The method

**JND = perceptual resolution, not a hard constraint.** Two configs closer than
a JND are indistinguishable, so testing both wastes a trial. We encode this by
**discretizing each continuous parameter onto a perceptually-spaced grid**:

| Grid type | For | Spacing |
|-----------|-----|---------|
| `weber_grid` (geometric) | Weber's-law params (amplitude, rate) | ×(1 + jnd) |
| `linear_grid` (additive) | absolute-JND params (seconds) | fixed step |
| `list_grid` (explicit) | naturally discrete params, **and integer JND knobs** | given levels |
| categorical | nominal params | `MixedSingleTaskGP` |

The current study uses the third row for everything: each optimized dimension is
an integer count of JNDs away from a baseline design. That has a second benefit
worth knowing — `ContinuousParam.normalize()` is **linear**, so a geometric grid
of raw values is *not* equally spaced in model coordinates and a stationary GP
kernel cannot treat "one JND" as a constant distance. On an integer JND axis it
can, by construction.

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

1. **Parameters** — edit `CONT_PARAMS` / `CAT_PARAMS`. For a FLAT contract each
   `name` is the Firestore field the app reads/writes; for a nested one (this
   study) the names are the optimizer's own dimensions and `space.expand()`
   maps them onto the document. Choose a grid builder per parameter:
   ```python
   CONT_PARAMS = [
       ContinuousParam("intensity", weber_grid(0.2, 1.0, 0.13)),   # Weber JND
       ContinuousParam("duration",  linear_grid(0.06, 1.0, 0.24)), # absolute JND
       ContinuousParam("pulseCount", list_grid([1, 2, 3, 4])),     # discrete
   ]
   CAT_PARAMS = [ CategoricalParam("pattern", ["a", "b", "c"]) ]   # or [] for none
   ```
   If the app's document is **nested** (as in the current study), the names above
   are the optimizer's dimensions rather than Firestore field names, and
   `space.expand()` maps a point in that space onto the document the app reads.
   `main.build_proposal_doc()` assembles the rest of the doc.
   `_validate()` runs on import and rejects duplicate names, empty/​non-ascending
   grids, and empty category lists — so a bad edit fails immediately with a clear
   message, not deep inside the GP at request time.
2. **Objectives** — `OBJECTIVE_FIELDS` (default `["subjectiveScore",
   "objectiveScore"]`, both maximized, in `[0,1]`). Add a third for 3 objectives;
   everything downstream adapts automatically.
3. **Trial budget** — auto-derived: `N_SOBOL = 1 + 2·(D+1)` (one anchor round plus
   the Sobol draws), `N_TOTAL = space.MAX_ROUND_NUMBER`. Override with the
   `N_SOBOL` / `N_TOTAL` env vars; `N_MOBO` is derived and read-only.
4. **Hard constraints** — implement `is_feasible(raw)` (returns `False` for
   configs that must never be tested). Applied to Sobol draws and MOBO candidates.

No other file needs editing to change the study. `main.py` and `optimizer_core.py`
read the space generically from `space.py`.

> **Finalizing real ranges:** run `inspect_db.py` against your existing data to
> read actual value ranges and category lists, then set them in `space.py`.

### Changing parameters after deploy

Editing `space.py` and redeploying is safe **between participants**. Doing it
while participants are mid-study is not: a knob vector is meaningful only
relative to the space that produced it, so a change retroactively reinterprets
— or invalidates — history that is already in Firestore. Every proposal is
stamped with `mobo.spaceVersion` and the service refuses to train on a round
from a different one.

| Edit | Effect on a participant who is already mid-study |
|------|--------------------------------------------------|
| **Add, rename, or remove a knob** | `D` changes, so `N_SOBOL = 1 + 2·(D+1)` moves mid-study. Bump `SPACE_VERSION` when you do: earlier rounds are then ignored for training (logged, not silent) rather than being reinterpreted under the new meaning. |
| **Change a knob's range** | Earlier knob values may fall outside the new grid. `on_grid()` rejects them and the round is dropped from training — loudly, in the logs — instead of being normalised to an out-of-bounds model input. |
| **Change a step size or `SEED_DESIGN`** | The same knob integer now renders a *different stimulus*, so history is silently wrong unless `SPACE_VERSION` is bumped. This is the one that looks harmless and is not. |
| **Change the wire format** (`expand()`, the cue list, the doc shape) | The app may no longer be able to render or echo the design. Coordinate with the app team first — this is their contract, not ours. |

> **⚠️ What this costs now.** The old permanent stall — where a collapsed
> history made the service re-propose a step whose doc already existed and
> answer `{"skipped": true}` forever — is gone: `next_phase_step` is derived
> from **rounds issued** (`max(phaseStep)`), which a broken history cannot move.
> The remaining cost is quieter and still matters: every unreadable round is
> excluded from training, so the participant keeps receiving stimuli but the GP
> never gets enough data to take over and the run silently degrades to pure
> exploration. Watch for `has N rounds but only M usable observations` in the
> Cloud Run logs — that line is the warning.

**Safe procedure**

1. Edit `space.py`, then run the full local suite (`make test` / `.\tasks.ps1
   test`). `_validate()` and `tests/test_space.py` catch bad grids at import.
2. Tell the app team **before** deploying if the WIRE FORMAT changed (a cue, a
   burst field, the doc shape) — that is their contract. Renaming a knob does
   not touch the wire; bumping `SPACE_VERSION` does not either, but it does
   exclude earlier rounds from training.
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
| `N_SOBOL` | `space.N_SOBOL_DEFAULT` = `1 + 2·(D+1)` | non-model rounds: 1 anchor + `2·(D+1)` Sobol draws. The GP takes over on *usable observations*, so `N_MOBO = N_TOTAL − N_SOBOL` is a maximum. |
| `N_TOTAL` | `space.MAX_ROUND_NUMBER` (18) | total **rounds** before `studyCompleted`. Refuses to start above the app's rules cap. |
| `NUM_RESTARTS` | `5` | acqf optimizer restarts |
| `RAW_SAMPLES` | `256` | acqf raw samples |
| `MC_SAMPLES` | `64` | MC samples for the hypervolume estimate |

The acqf knobs are deliberately light. With `CAT_PARAMS = []` the plain
`optimize_acqf` path runs (one inner optimisation, not one per categorical
combination), so a single `/updatePolicy` call stays well under the Cloud Run
300 s timeout at `D = 4` with ≤18 observations. Re-measure if `D` grows.

## Local development & validation

No cloud needed. Run these after any edit to `space.py`:

```bash
python space.py                # print the resolved grids — eyeball the level counts
python tests/test_space.py     # unit tests (~4 s): grids, snapping, encoding, and an
                               # EXHAUSTIVE pass over every reachable design vs the
                               # app's security rules
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

Concurrency & idempotency: `parameterValues` docs are written with `create()`,
which is **first-wins and the only real guard** — the per-user `threading.Lock`
lives in one Python process while Cloud Run autoscales, so it serializes
same-user requests only *within* an instance, and the read-then-create check is
a cross-instance TOCTOU window. Duplicate Cloud Function deliveries are detected
and skipped. Failed attention checks (`attentionCheckPassed == false`) are
excluded from the training data but **still consume their round**, because the
app's rules pin `phaseStep == roundNumber`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Cloud Function never fires | Trigger is on the wrong database. It must match the app's DB (default `(default)`); a **named** DB needs `database:` set on the trigger. |
| `/updatePolicy` 500s | Check Cloud Run logs. A single malformed doc is skipped with a warning, not fatal. |
| `/updatePolicy` answers **422** and the function logs `REFUSED (422, not retried)` | `space.expand()` rendered a design the rules mirror rejects — only possible after an edit to `space.py` that passed import. Fix the space; do not redeploy the trigger. `python tests/test_space.py` reproduces it. |
| Deploy/queries need a Firestore index | `firebase deploy --only firestore:indexes`. The file must also carry the **app's** indexes — including `parameterValues(pid, schemaVersion, createdAt DESC)`, which its listener needs — or the declarative deploy deletes them. |
| Participant receives nothing, and there is **no error anywhere on our side** | The app's write was rejected by its own security rules. Almost always an illegal burst (a float `pulseCount`, a value out of range, a rate above 120 Hz) or a `phase` other than `'exploration'`. Reproduce with `python inspect_db.py` — CHECK 3 runs the rules mirror over every burst in the database. |
| `firebase deploy` fails with `User code failed to load. Cannot determine backend specification. Timeout after 10000` | Not a code error — `firebase-tools` loads `functions/` in a subprocess to discover the triggers and gives it 10 s, which Windows/cold-cache machines routinely miss. Confirm the code is fine with `node -e "require('./index.js')"` in `functions/`, then re-run with `FUNCTIONS_DISCOVERY_TIMEOUT=120` set. Also check `functions/package.json`'s `engines.node` matches your local `node --version`. |
| Logs show `has N rounds but only M usable observations` | Results are arriving but are not trainable — a `spaceVersion` mismatch after a mid-study redeploy, or the app echoing designs off the knob grid. The participant still gets stimuli, but the model phase is shrinking. See [Changing parameters after deploy](#changing-parameters-after-deploy). |
| Logs show `echoed a design outside the knob grid` | The app clamped or substituted a value before rendering. The round is dropped from training (and from the analysis) because the participant felt something the optimizer cannot name. Ask the app team whether the echo is what was *delivered* or what was *received*. |
| Every Firestore call returns `403 This API method requires billing to be enabled` | Billing is disabled on that project. The data is retained but unreadable until billing is restored. |
| `--allow-unauthenticated` rejected on deploy | Org policy `iam.allowedPolicyMemberDomains` blocks `allUsers` — see [Org-policy gotchas](#org-policy-gotchas). |
| Slow MOBO step / timeout | Lower `RAW_SAMPLES`/`NUM_RESTARTS`/`MC_SAMPLES`; ensure the image built the fast EHVI kernel (needs `build-essential`+`ninja`, already in the Dockerfile). |

## Data contract with the app

The authoritative contract is the app team's Firestore **security rules** (ruleset
`42691aae`, released 2026-09-08). They validate the app's own writes, so a design
this service writes that violates them is accepted from us (the Admin SDK bypasses
rules) and then makes the *app's* result write illegal — which produces no error
on our side at all. `space.rules_valid_burst()` mirrors the predicate and
`write_next_params()` refuses to write a design that fails it.

- Service writes `parameterValues/{pid}_step_N`: `schemaVersion: 2`, `candidateId`,
  `phase: "exploration"`, `phaseStep`/`roundNumber`, `hapticMode: "burst"`,
  `isFinalRound`, `createdAt`, the 14-cue `haptics` map, and a `mobo` map holding
  the knob vector, the real optimizer phase and the `spaceVersion`.
- App writes `interventionResults/{id}` echoing the design it rendered, plus
  `pid`, `phaseStep` == `roundNumber`, both objective fields,
  `attentionCheckPassed`, `touchedTarget`, `parameterDocumentId`, `candidateId`,
  `sessionId`, `mapName`, `authUid`, `resultId`.
- Service writes the anytime hypervolume to `moboMetrics/{pid}_step_N` — **never**
  into the app's own doc, whose rules permit a client update only for a retry that
  changes nothing but `createdAt`.
- Study ends at `N_TOTAL` **rounds** → `users/{pid}.studyCompleted = true`. The app
  cannot read `users/`, so the end-of-study signal it *can* see is `isFinalRound`
  on the last proposal.

### The burst profile (per cue)

Each of the 14 cues carries exactly these five keys — no more, no fewer (the rule
uses `hasOnly`, so a sixth key is fatal):

| Field | Type | Meaning | Range (enforced by the rules) |
|-------|------|---------|-------------------------------|
| `intensity` | double | Core Haptics amplitude | 0 – 1 |
| `sharpness` | double | Core Haptics timbre | 0 – 1 |
| `pulseCount` | **int64** | pulses inside the burst | 1 – 120, **and** ≤ `onDuration × 120` |
| `onDuration` | double | length of the **whole burst**, seconds | 0.01 – 2.0 |
| `offDuration` | double | silence before the burst repeats, seconds | 0.01 – 2.0 |

So the pulse **rate** is `pulseCount / onDuration` Hz and the cross-constraint is a
120 Hz cap (confirmed by the app team, 2026-09-11 — every cue in their seed design
sits at exactly 40, 60 or 120 Hz). `pulseCount` must be a genuine integer: a
Python `float` lands in Firestore as a double and fails `pulseCount is int`.

### The 14 cues

`start`, `onRoute`, `offRoute`, `onRouteIntersection`, `offRouteIntersection`,
`landmark`, `end`, `street`, `onRouteSidewalk`, `offRouteSidewalk`,
`onRouteCrosswalk`, `offRouteCrosswalk`, `turn`, `intersectionCenter`.

### What is optimized

14 × 5 = 70 numbers against at most 18 session-level score pairs is
under-determined — a GP over 70 inputs fits 70 lengthscales from 18 points, so the
posterior is the prior and `qLogNEHVI` degenerates into quasi-random search. The
optimizer therefore searches four **integer JND knobs**, offsets from the app
team's seed design (`space.SEED_DESIGN`, copied verbatim from the live doc):

| Knob | Levels | One step | Applies to |
|------|--------|----------|------------|
| `gainIntensity` | 9 (−6…+2) | ×1.13 | every cue's `intensity` |
| `gainSharpness` | 7 (−3…+3) | +0.20 | every cue's `sharpness` (placeholder step) |
| `gainRate` | 11 (−8…+2) | ×1.20 | every cue's pulse rate |
| `gainBurstLength` | 8 (−4…+3) | ×1.25 | every cue's `onDuration` (placeholder step) |

`space.expand()` scales each cue **multiplicatively** from its seed value, so the
contrasts the app team designed between cues survive at every knob setting —
re-gridding absolute per-cue values does not (at low settings the 14 cues collapse
onto one intensity and the navigation signal is destroyed). Knobs all zero
reproduces the seed design exactly, so the expert baseline is reachable and is the
worst case, not a lucky draw. `offDuration` is held at its seed value pending an
answer on whether a cue loops.

The whole grid is 9 × 7 × 11 × 8 = **5,544 designs**; `tests/test_space.py` checks
exhaustively that every one of them satisfies the live rules and that the map is
injective (so a rendered design can be inverted back to its knobs, which is how
history is read back from what the app echoed).

## License & citing

MIT — see [LICENSE](LICENSE). If you use this in academic work, please cite it;
machine-readable metadata is in [CITATION.cff](CITATION.cff) (a paper reference
will be added there once published).
