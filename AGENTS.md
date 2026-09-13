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

A candidate is a whole **haptic design**: a nested `haptics` map of 14 navigation
cues × 5 burst values (70 numbers), written under `schemaVersion: 2`. The app's
security rules cap a participant at **18 rounds**, and one round yields a single
pair of scores for the entire session — so 70 free numbers would be
*unidentifiable*, not merely under-sampled.

The optimizer therefore searches **four integer JND knobs** (`gainIntensity`,
`gainSharpness`, `gainRate`, `gainBurstLength`) that are offsets from the app
team's seed design; `space.expand()` turns a knob vector into all 70 numbers.
Knobs all zero **is** the seed design. Round 1 is that anchor, rounds 2–11 are
Sobol (`2·(D+1)` draws), rounds 12–18 are GP `qLogNEHVI`.

## Ground rules

1. **`space.py` is the only file that *defines* the study.** Knobs, grids,
   `SEED_DESIGN`, `expand()`, the rules mirror, feasibility. `optimizer_core.py`
   reads the space generically and needs no edit to change parameters.
   `main.py` does know the *wire format* (it assembles the proposal doc), so a
   change to the app's document schema — as opposed to the search space —
   touches `space.expand()`, `main.build_proposal_doc()`, both test files, the
   exporter and the R script together. Changing what is *optimized* is still a
   `space.py`-only edit.
2. **Validate locally before claiming anything works.** No cloud needed:
   ```bash
   python space.py              # print resolved grids — eyeball the level counts
   python tests/test_space.py   # search-space unit tests (~5 s: exhaustive rules pass)
   python simulate.py           # offline loop: on-grid, unique, hypervolume rises
   python tests/test_service.py # full HTTP service vs an in-memory Firestore fake
   ```
   Or `make test` / `.\tasks.ps1 test` for all four. `_validate()` runs on import
   and rejects duplicate names, non-ascending grids and empty category lists, so
   a bad edit fails loudly at import, not deep inside the GP at request time.
3. **`2·(D+1)` Sobol draws is a deliberate decision**, one more than the study
   memo's `2n+1`. It is not an off-by-one bug — do not "fix" it. The constant
   `N_SOBOL` counts the anchor round too (`space.N_SOBOL_DEFAULT = N_ANCHOR +
   2·(D+1)` = 11): rounds 1–11 are not model-driven. `N_MOBO` is a *maximum* —
   the GP takes over on usable observations, so every discarded round shortens
   the model phase by one (logged as `has N rounds but only M usable observations`).
4. **Never deploy `firestore:rules`.** The app team owns the live security rules
   (ruleset `42691aae`, released 2026-09-08); a blind deploy overwrites them.
   Note this protection is **procedural, not structural**: there is no
   `firebase.json` in this repository at all, so the deploy runs from whatever
   external `firebase-deploy/` directory you built — check *that* copy before
   running anything. Reading the rules is fine and necessary; `space.py` mirrors
   them in `rules_valid_burst()`.
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

`GET <url>/health` echoes the live knob names, `spaceVersion`, `schemaVersion`,
cue count, database and round budget — the fastest check that the space you
think you deployed is the one running.

> The predecessor project `multinav-a6ade` is retired and its billing is
> disabled, which makes its Firestore unreadable (`403 … requires billing`). Do
> not point anything at it.

## Firestore contract

| Collection | Written by | Must contain |
|-----------|-----------|--------------|
| `users/{pid}` | **an admin** | clients cannot write here (rules deny it), so `registerUserOnCreate` only fires for an Admin SDK / console create |
| `interventionResults/{id}` | app | `pid`, `phaseStep` == `roundNumber` (1–18), `attentionCheckPassed`, `touchedTarget`, `parameterDocumentId`, `candidateId`, `sessionId`, `mapName`, `authUid`, `resultId`, `schemaVersion` 2, `hapticMode` `'burst'`, `phase` `'exploration'`, both `space.OBJECTIVE_FIELDS`, and the echoed 14-cue `haptics` map |
| `parameterValues/{pid}_step_N` | this service | `schemaVersion` 2, `candidateId`, `phase` `'exploration'`, `phaseStep`/`roundNumber`, `isFinalRound`, the `haptics` map, and a `mobo` map holding the knobs + the real phase |
| `moboMetrics/{pid}_step_N` | this service | the anytime hypervolume. **Never annotate the app's own docs** — their rules allow a client update only for a retry that changes nothing but `createdAt`. |

`users/{pid}.studyCompleted = true` is set once `N_TOTAL` **rounds** (not clean
observations) exist. The app cannot read `users/` — it learns the study is over
from `isFinalRound` on the last proposal.

## Traps

- **A space change mid-study degrades a participant silently.** It no longer
  STALLS them — `next_phase_step` comes from rounds issued (`max(phaseStep)`),
  which a broken history cannot move — but every round stamped with a different
  `mobo.spaceVersion`, or carrying knobs that are no longer on the grid, is
  excluded from training. The participant keeps receiving stimuli while the GP
  quietly starves and the run reverts to exploration. Log signature: `has N
  rounds but only M usable observations`. Change the space **between**
  participants, and bump `SPACE_VERSION` when you do. Full table in
  [README](README.md#changing-parameters-after-deploy).
- **Malformed docs fail silently by design.** `load_observations()` skips them
  with a warning so one bad doc can't take down a participant's whole update —
  which also means a broken contract looks like "the optimizer just isn't
  learning". Check the Cloud Run logs for `skipping malformed doc`.
- **`(default)` vs a database literally named `default`** are different stores.
  The service, the triggers and `inspect_db.py` all use `(default)`. A named
  database needs `database:` on each Firestore trigger in `index.js` and
  `FIRESTORE_DATABASE` on the service.
- **An illegal design is invisible to us.** The Admin SDK bypasses the security
  rules, so we can write a burst the rules forbid — but then the *app's* own
  result write is rejected, no Cloud Function fires, and the participant stalls
  with **no log line anywhere on our side**. `space.rules_valid_haptics()`
  mirrors the rule and `write_next_params()` refuses to write without it. Never
  loosen that mirror with an epsilon: the rule is a hard predicate on the
  written doubles.
- **`pulseCount` must be a real `int`.** The rules check `pulseCount is int`;
  Firestore stores a Python `float` as a double, so a `2.0` fails. `expand()`
  produces it with `int(round(...))`, never via `ContinuousParam.snap()`.
- **`phase` on the wire is always the literal `'exploration'`.** The rules pin
  the *result* doc to that string, so if the app echoes our label, writing
  `'optimization'` would make every GP round's result illegal. The real label
  lives in `mobo.phase` and that is what the analysis splits on.
- **Burst semantics:** `onDuration` is the length of the WHOLE burst and
  `pulseCount` is the number of pulses inside it, so the rate is
  `pulseCount / onDuration` Hz — capped at 120 Hz, which is exactly what the
  rules' `pulseCount <= onDuration * 120` encodes. Confirmed by the app team
  2026-09-11. `offDuration` is the silence before the burst repeats.
- **Rounds are counted, not observations.** The rules pin
  `phaseStep == roundNumber` and cap it at 18, so `load_observations()` returns
  `rounds_done` (max `phaseStep` seen, taken *before* any content filter)
  separately from the training-set size. Conflating them desynchronises the
  counter the moment a round is discarded, and the service then re-proposes a
  step whose doc already exists — `{"skipped": true}` forever.

## Open questions

Answered 2026-09-11: the burst semantics (see Traps). Still open with the app
team, in rough order of how badly they block:

1. Who creates `users/{pid}` and who writes round 1? The rules deny client writes
   to `users/`, so the registration trigger can only fire from an admin.
2. Does the app copy `phase` / `candidateId` / `schemaVersion` from our proposal
   onto its result doc?
3. Is `validResult` `hasAll` or `hasOnly`? We add root fields (`mobo`,
   `isFinalRound`) to the proposal doc that a key-exact rule would reject if
   the app echoed them. (`roundNumber` is required by the rules, not extra.)
4. What exactly does the app's listener query? (A live composite index on
   `pid + schemaVersion + createdAt DESC` implies it, but an index only proves
   some query existed.)
5. Does a cue loop, and what stops it? Decides whether `offDuration` is a live
   dimension or dead weight.
6. Is the echoed burst what was *proposed* or what was *delivered* (post-clamp)?
7. **After a failed attention check, does the app REPEAT round k (same
   `roundNumber`, same design) or advance to k+1?** `firebase/index.js` never
   calls the optimizer for a failed result ("step repeated"), so the service's
   round counter is only reached through the passing repeat. Consistent if the
   app repeats; a silent stall at round k if it advances. Neither is verified.
8. Can the questionnaire produce a per-cue rating? This is the highest-value
   question in the list — it would turn one round into up to 14 observations and
   make per-cue personalization statistically defensible instead of
   budget-forced.

Two knob step sizes are placeholders with no psychophysical basis:
`gainSharpness` (0.20 additive) and `gainBurstLength` (×1.25). The
`gainIntensity` Weber fraction of 1.13 is finer than the published vibrotactile
amplitude range (~0.20), i.e. conservative. The repo contains no literature
citations for any of these — fixing that is an open item.

Use `inspect_db.py` (read-only; needs `GOOGLE_CLOUD_PROJECT` and ADC) to check
what the app is actually writing before changing the contract.
