"""
Haptic-design MOBO optimizer service (Cloud Run).

Firestore-mediated, same as before:
  app -> interventionResults -> Cloud Function -> POST /updatePolicy -> here,
  here -> parameterValues -> app snapshot listener.

What a document looks like now (contract, ruleset 42691aae — see space.py):
a candidate is a nested `haptics` map of 14 navigation cues x 5 burst values,
carried on a doc that declares `schemaVersion: 2`, a `candidateId`, and the
literal `phase: "exploration"`. The optimizer's own state (the knob vector and
the REAL phase label) rides along in a `mobo` map the app ignores.

The search space lives entirely in space.py; the GP/acqf logic in
optimizer_core.py. Both are Firestore-free and unit-testable.

HTTP status semantics (the Cloud Function keys its retry policy on them):
  2xx  handled — including the benign "a concurrent delivery got there first"
  4xx  a DETERMINISTIC refusal (illegal design, bad request): logged loudly,
       NOT retried — retrying would only repeat the same refusal
  5xx  a TRANSIENT failure (Firestore outage, permission, timeout): retried
"""

import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone

import torch
from flask import Flask, request, jsonify
from google.api_core import exceptions as gapi_exceptions
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

import space
import optimizer_core as core

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Firestore database. The project's "(default)" database is already in nam5
# (US multi-region), so the default is correct — no named/secondary DB needed.
# Overridable via FIRESTORE_DATABASE only if you ever point at another database.
FIRESTORE_DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")
db = firestore.Client(database=FIRESTORE_DATABASE)

RESULTS = "interventionResults"
PROPOSALS = "parameterValues"
METRICS = "moboMetrics"   # our own bookkeeping; we never write to the app's docs

# ── Trial-budget constants ───────────────────────────────────────────────────
# Defined once in space.py (N_ANCHOR, N_SOBOL_DEFAULT) so simulate.py, the tests
# and this service cannot drift apart; overridable here via env vars. Round 1 is
# the anchor, rounds 2..N_SOBOL are Sobol draws, the rest are GP rounds. N_TOTAL
# is capped by the app's security rules, which reject roundNumber > 18. N_MOBO
# is a MAXIMUM: the GP takes over only once N_SOBOL usable observations exist,
# so every discarded round shortens the model-driven phase by one.
N_ANCHOR = space.N_ANCHOR
N_SOBOL = int(os.environ.get("N_SOBOL", str(space.N_SOBOL_DEFAULT)))
N_TOTAL = int(os.environ.get("N_TOTAL", str(space.MAX_ROUND_NUMBER)))
N_MOBO = max(0, N_TOTAL - N_SOBOL)

if N_TOTAL > space.MAX_ROUND_NUMBER:
    raise SystemExit(
        f"N_TOTAL={N_TOTAL} exceeds the app's rules cap of {space.MAX_ROUND_NUMBER} "
        "rounds — every result past the cap would be rejected by Firestore and the "
        "participant would stall with no error visible here."
    )
if N_SOBOL >= N_TOTAL:
    raise SystemExit(
        f"N_SOBOL={N_SOBOL} leaves no model-driven rounds inside N_TOTAL={N_TOTAL}. "
        "That is a random-search study wearing a GP label — lower N_SOBOL or drop a knob."
    )

REF_POINT_HV = torch.tensor(core.REF_POINT, dtype=torch.double)

# Per-user lock to serialise concurrent requests for the same user *within one
# Cloud Run instance*. Cloud Run autoscales, so this is NOT a global lock — the
# real cross-instance guard is create() being first-wins in write_next_params().
_user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


class ProposalRefused(RuntimeError):
    """The design we were about to write would be rejected by the app's rules.

    Deterministic: the same round would produce the same design again, so it is
    answered with HTTP 422 — loud in both logs, but NOT retried by the Cloud
    Function (which rethrows only on 5xx). A silent False here would be
    indistinguishable from the benign "a concurrent delivery got there first"
    case, and the participant would stall with nothing in any log.
    """


@app.errorhandler(ProposalRefused)
def _refused(e: ProposalRefused):
    return jsonify({"ok": False, "error": str(e), "retry": False}), 422


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── Hypervolume tracking ─────────────────────────────────────────────────────
def log_hypervolume(user_id: str, Y: torch.Tensor, phase_step: int, obs_count: int) -> float:
    """Record the anytime hypervolume in OUR collection, not the app's.

    The app's security rules allow a client to update its own interventionResults
    doc only for a retry that changes nothing but createdAt. Annotating that doc
    (as this service used to) makes a legitimate app retry illegal, so the
    metric goes to moboMetrics/{pid}_step_{n} instead.
    """
    pareto_Y = Y[is_non_dominated(Y)]
    hv = float(Hypervolume(ref_point=REF_POINT_HV).compute(pareto_Y))
    db.collection(METRICS).document(f"{user_id}_step_{phase_step}").set(
        {
            "pid": user_id,
            "phaseStep": phase_step,
            "obsCount": obs_count,
            "hypervolume": hv,
            "spaceVersion": space.SPACE_VERSION,
            "createdAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
    log.info("user=%s obs=%d phase_step=%d hypervolume=%.4f — saved",
             user_id, obs_count, phase_step, hv)
    return hv


# ── Firestore helpers ────────────────────────────────────────────────────────
class Proposals:
    """Every parameterValues doc for one participant, read in ONE query.

    Serves three consumers that used to each hit Firestore on their own (one
    document get() per result doc, then a second full stream): the
    result -> proposal join, the "does this step already exist" check, and the
    set of designs already ISSUED — which is the novelty set. Issued, not
    observed: a round whose result was unusable (an off-grid echo, a stale
    spaceVersion, a failed attention check) was still FELT by the participant,
    and re-issuing it is what turns one bad round into a run of identical stimuli.
    Decoded with space.decode_knobs, so the app team's hand-seeded auto-id doc
    (haptics only, no `mobo` map) counts too.
    """

    def __init__(self, user_id: str):
        self.by_id: dict[str, dict] = {}
        self.by_step: dict[int, dict] = {}
        self.issued: set = set()
        for snap in (db.collection(PROPOSALS)
                     .where(filter=FieldFilter("pid", "==", user_id))
                     .stream()):
            doc = snap.to_dict() or {}
            self.by_id[snap.id] = doc
            step = space.round_number(doc.get("phaseStep"))
            if step is not None:
                self.by_step.setdefault(step, doc)
            knobs, why = space.decode_knobs(doc)
            if knobs is not None:
                self.issued.add(space.obs_key(knobs))
            else:
                log.warning("Proposals: %s/%s has no decodable design (%s) — not in the "
                            "novelty set", user_id, snap.id, why)

    def for_result(self, user_id: str, result: dict, step: int) -> dict | None:
        """The proposal a result was rendered from: by parameterDocumentId (which
        the rules REQUIRE on every result), else by step."""
        pdid = result.get("parameterDocumentId")
        if pdid and str(pdid) in self.by_id:
            return self.by_id[str(pdid)]
        return self.by_step.get(step) or self.by_id.get(f"{user_id}_step_{step}")


def knobs_for_result(user_id: str, result: dict, step: int, proposals: Proposals) -> dict | None:
    """The knob vector to train on for one result doc, or None to skip it.

    The ECHOED design is authoritative: if the app clamped or substituted
    anything, the participant felt the echo, and training on the proposal would
    fit a stimulus nobody experienced. The proposal supplies the spaceVersion
    gate and a diagnostic when the two differ.
    """
    proposal = proposals.for_result(user_id, result, step)
    knobs, why = space.decode_knobs(result, proposal)
    if knobs is None:
        log.error("knobs_for_result: %s step=%s excluded from training — %s", user_id, step, why)
        return None
    if proposal is not None and why == "echo":
        stored = space.knobs_from_stored((proposal.get("mobo") or {}).get("knobs"))
        if stored is not None and stored != knobs:
            log.warning("knobs_for_result: %s step=%s echo != proposal (%s vs %s) — "
                        "training on the echo", user_id, step, knobs, stored)
    return knobs


def load_observations(user_id: str, proposals: Proposals):
    """Return (X_model [n,D], Y [n,m], last_phase_step, observed_keys, rounds_done).

    `rounds_done` counts ROUNDS ISSUED (max phaseStep seen, taken before any
    content filter); `len(X_model)` counts clean training observations. They
    differ whenever a round is discarded, and conflating them desynchronises the
    step counter from the app's roundNumber — which the rules pin to phaseStep.
    """
    docs = list(
        db.collection(RESULTS)
        .where(filter=FieldFilter("pid", "==", user_id))
        .order_by("phaseStep")
        .stream()
    )
    if not docs:
        return None, None, None, set(), 0

    rows_x, rows_y, observed_keys = [], [], set()
    last_phase_step = None
    rounds_done = 0
    seen_steps = set()  # Cloud Functions is at-least-once → deduplicate
    for doc in docs:
        d = doc.to_dict()
        raw_step = d.get("phaseStep")
        # A doc from another schema is not a round of THIS study. Without this
        # check one stray doc (a pilot, an app QA run, a legacy schema-1 result)
        # sets rounds_done and can end the study before it begins.
        version = d.get("schemaVersion")
        if version != space.SCHEMA_VERSION:
            log.warning("load_observations: %s step=%s has schemaVersion=%r "
                        "(expected %s) — ignored entirely, not counted as a round",
                        user_id, raw_step, version, space.SCHEMA_VERSION)
            continue
        # A step that is not a whole number is ignored ENTIRELY. Half-admitting it
        # (trained on, but not counted as a round) is what makes next_phase_step
        # point at a doc that already exists — the permanent skip.
        step = space.round_number(raw_step)
        if step is None:
            log.warning("load_observations: %s has phaseStep=%r, not a round number — ignored",
                        user_id, raw_step)
            continue
        # Count the round FIRST: a discarded round still consumed a roundNumber.
        rounds_done = max(rounds_done, step)
        if d.get("attentionCheckPassed") is False:
            continue
        if step in seen_steps:
            log.warning("load_observations: duplicate phaseStep=%s for %s — skipping", step, user_id)
            continue
        # One malformed doc (missing scores, an unreachable design, a stale
        # spaceVersion) must not take down the whole participant's update.
        try:
            knobs = knobs_for_result(user_id, d, step, proposals)
            if knobs is None:
                continue
            x_row = space.to_model_row(knobs)
            y_row = [float(d[f]) for f in space.OBJECTIVE_FIELDS]
            key = space.obs_key(knobs)
        except (KeyError, ValueError, TypeError) as e:
            log.warning("load_observations: skipping malformed doc for %s step=%s: %s", user_id, step, e)
            continue
        seen_steps.add(step)
        last_phase_step = step
        rows_x.append(x_row)
        rows_y.append(y_row)
        observed_keys.add(key)

    if not rows_x:
        return None, None, last_phase_step, observed_keys, rounds_done

    X = torch.tensor(rows_x, dtype=torch.double)
    Y = torch.tensor(rows_y, dtype=torch.double)
    return X, Y, last_phase_step, observed_keys, rounds_done


def build_proposal_doc(user_id: str, knobs: dict, optimizer_phase: str, phase_step: int) -> dict:
    """The parameterValues payload, in the shape the app's schemaVersion 2 expects."""
    haptics = space.expand(knobs)
    return {
        "pid": user_id,
        # schemaVersion is NOT cosmetic: the app's listener filters on it (there
        # is a live composite index on pid + schemaVersion + createdAt DESC), so
        # a doc without it is invisible to the app.
        "schemaVersion": space.SCHEMA_VERSION,
        "hapticMode": space.HAPTIC_MODE,
        "candidateId": f"{user_id}-round-{phase_step:02d}-{optimizer_phase}-{_utc_stamp()}",
        # The rules pin the RESULT doc's phase to the literal 'exploration'. If
        # the app echoes this field, anything else here would make every GP
        # round's result illegal, and the participant would stall silently. The
        # real optimizer phase lives under `mobo` instead.
        "phase": space.WIRE_PHASE,
        "phaseStep": phase_step,
        "roundNumber": phase_step,
        # The app cannot read users/{pid}.studyCompleted (the rules deny reads on
        # every collection except parameterValues), so the end-of-study signal
        # has to travel on this doc.
        "isFinalRound": phase_step >= N_TOTAL,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "haptics": haptics,
        "mobo": {
            "knobs": {k: int(round(float(v))) for k, v in knobs.items()},
            "phase": optimizer_phase,
            "spaceVersion": space.SPACE_VERSION,
        },
    }


def write_next_params(user_id: str, knobs: dict, optimizer_phase: str, phase_step: int) -> bool:
    """Write the next design as a new parameterValues doc (create() = first-wins).

    Returns False only for the benign duplicate. Raises ProposalRefused (-> 422,
    not retried) for a design the rules would reject, and lets transient
    Firestore errors propagate (-> 500, retried by the Cloud Function).
    """
    if phase_step > space.MAX_ROUND_NUMBER:
        raise ProposalRefused(
            f"{user_id} step={phase_step}: the app's rules cap roundNumber at "
            f"{space.MAX_ROUND_NUMBER} and would reject the result"
        )

    doc = build_proposal_doc(user_id, knobs, optimizer_phase, phase_step)
    # Last line of defence: an illegal design is accepted from us (the Admin SDK
    # bypasses the rules) but makes the APP's own write illegal, which produces
    # no error on our side at all — the participant just stops.
    if not space.rules_valid_haptics(doc["haptics"]):
        log.error("write_next_params: %s step=%d — expand(%s) violates the app's "
                  "validHaptics rule; NOT writing", user_id, phase_step, knobs)
        raise ProposalRefused(
            f"{user_id} step={phase_step}: rendered design violates validHaptics"
        )

    ref = db.collection(PROPOSALS).document(f"{user_id}_step_{phase_step}")
    try:
        ref.create(doc)
        return True
    except gapi_exceptions.AlreadyExists:
        # The only benign failure: a concurrent delivery got there first.
        log.warning("write_next_params: step_%d already exists for %s — duplicate suppressed",
                    phase_step, user_id)
        return False
    except Exception:
        # Anything else (permission, quota, deadline, an outage) used to be
        # logged as a duplicate and swallowed, which is indistinguishable from a
        # stalled participant. Let it surface as a 5xx so the function retries.
        log.exception("write_next_params: FAILED to write step_%d for %s", phase_step, user_id)
        raise


def study_completed(user_id: str) -> bool:
    snap = db.collection("users").document(user_id).get()
    return bool(snap.exists and (snap.to_dict() or {}).get("studyCompleted"))


def already_enrolled(user_id: str) -> bool:
    """True if ANY proposal exists for this pid, whoever wrote it.

    One limited query on pid alone — not by document id and not by phaseStep:
    the app team seeds round 1 with an auto-id doc, and nothing in the
    schemaVersion-2 contract obliges a parameterValues doc to carry phaseStep,
    so a narrower check can miss it and write a competing round-1 design that
    the app's newest-first listener would then pick up mid-round.
    """
    for _ in (db.collection(PROPOSALS)
              .where(filter=FieldFilter("pid", "==", user_id))
              .limit(1)
              .stream()):
        return True
    return False


def exploration_draw(next_phase_step: int, avoid: set) -> dict:
    """The Sobol design for a round, replaced by a fresh random point on collision.

    Indexed by the ROUND being filled, never by how many results were usable:
    indexing by the usable count hands a participant whose results stop parsing
    the SAME design every remaining round, because the count stops advancing.
    """
    raw = space.sobol_next(next_phase_step - N_ANCHOR - 1)
    if space.obs_key(raw) in avoid:  # keep exploration on-grid AND novel
        raw = space.random_feasible(seed=next_phase_step, exclude=avoid)
    return raw


def choose_proposal(obs_count: int, next_phase_step: int, avoid: set,
                    X_model, Y) -> tuple[dict, str]:
    """(knobs, optimizer_phase) for the next round.

    The GP takes over on USABLE observations (it needs the data), while the round
    budget and the Sobol pointer follow the ROUND. So a discarded round costs one
    model-driven round at the end — logged by update_policy — rather than fitting
    a GP on fewer points than the exploration phase was sized for.
    """
    if next_phase_step <= N_ANCHOR:
        return space.anchor(), "anchor"
    if obs_count < N_SOBOL or X_model is None:
        return exploration_draw(next_phase_step, avoid), "sobol"
    return core.choose_next(X_model, Y, avoid, dedup_seed=next_phase_step), "mobo"


# ── Registration endpoint ────────────────────────────────────────────────────
@app.post("/registerUser")
def register_user():
    payload = request.get_json(force=True)
    user_id = payload.get("userId")
    if not user_id:
        return jsonify({"ok": False, "error": "missing userId"}), 400

    with _user_locks[user_id]:
        # Completion writes users/{pid} with merge=True, which CREATES the doc
        # and re-fires registerUserOnCreate. Without this guard a finished
        # participant would be handed a fresh round-1 design.
        if study_completed(user_id):
            log.info("registerUser: %s already completed the study — skipping", user_id)
            return jsonify({"ok": True, "skipped": True, "studyCompleted": True})
        if already_enrolled(user_id):
            log.info("registerUser: %s already has a proposal — skipping", user_id)
            return jsonify({"ok": True, "skipped": True})

        knobs = space.anchor()
        written = write_next_params(user_id, knobs, "anchor", 1)
        log.info("registerUser: anchor design as step_1 for %s (written=%s)", user_id, written)
        return jsonify({"ok": True, "written": written, "knobs": knobs, "phase": "anchor"})


# ── Main endpoint ────────────────────────────────────────────────────────────
@app.post("/updatePolicy")
def update_policy():
    payload = request.get_json(force=True)
    user_id = payload.get("userId")
    if not user_id:
        return jsonify({"ok": False, "error": "missing userId"}), 400

    if payload.get("type") != "interventionResult":
        return jsonify({"ok": True, "ignored": True})

    with _user_locks[user_id]:
        # 1. Load history: every proposal in ONE query, then the results.
        proposals = Proposals(user_id)
        X_model, Y, last_phase_step, observed_keys, rounds_done = load_observations(user_id, proposals)
        obs_count = 0 if X_model is None else len(X_model)

        hv = None
        if Y is not None and obs_count >= 1 and last_phase_step is not None:
            hv = log_hypervolume(user_id, Y, last_phase_step, obs_count)

        # Study complete — counted in ROUNDS ISSUED, which is what the app's
        # roundNumber cap applies to, not in clean observations.
        if rounds_done >= N_TOTAL:
            db.collection("users").document(user_id).set(
                {"studyCompleted": True, "studyCompletedAt": firestore.SERVER_TIMESTAMP}, merge=True
            )
            log.info("user=%s study COMPLETED (%d rounds, %d usable observations)",
                     user_id, rounds_done, obs_count)
            return jsonify({"ok": True, "studyCompleted": True,
                            "roundsDone": rounds_done, "obsCount": obs_count})

        next_phase_step = rounds_done + 1
        # Rounds are the budget the app enforces; observations are what the GP
        # can learn from. When they diverge the model phase is quietly shrinking.
        if rounds_done > obs_count:
            log.warning("user=%s has %d rounds but only %d usable observations — "
                        "the model-driven phase is %d rounds shorter than planned",
                        user_id, rounds_done, obs_count, rounds_done - obs_count)

        # Idempotency: if a concurrent invocation already wrote this step, stop.
        # (create() below is the real cross-instance guard; this saves the GP fit.)
        if next_phase_step in proposals.by_step or f"{user_id}_step_{next_phase_step}" in proposals.by_id:
            log.info("user=%s step_%d already exists — skipping duplicate", user_id, next_phase_step)
            return jsonify({"ok": True, "skipped": True})

        # 2. Choose the next design. Avoid everything already ISSUED, not just
        #    what produced a usable observation.
        avoid = observed_keys | proposals.issued
        try:
            knobs, phase = choose_proposal(obs_count, next_phase_step, avoid, X_model, Y)
        except Exception:
            log.exception("MOBO failed for %s — falling back to Sobol", user_id)
            knobs, phase = exploration_draw(next_phase_step, avoid), "sobol-fallback"

        # 3. Write. ProposalRefused propagates to the 422 handler above.
        written = write_next_params(user_id, knobs, phase, next_phase_step)
        log.info("user=%s rounds=%d obs=%d phase=%s next_step=%d written=%s knobs=%s",
                 user_id, rounds_done, obs_count, phase, next_phase_step, written, knobs)

        return jsonify({
            "ok": True,
            "written": written,
            "phase": phase,
            "obsCount": obs_count,
            "roundsDone": rounds_done,
            "hypervolume": hv,
            "lastPhaseStep": last_phase_step,
            "nextPhaseStep": next_phase_step,
            "nextKnobs": knobs,
        })


@app.get("/health")
def health():
    return jsonify({"ok": True,
                    "space": space.PARAM_NAMES,
                    "spaceVersion": space.SPACE_VERSION,
                    "schemaVersion": space.SCHEMA_VERSION,
                    "cues": len(space.CUES),
                    "database": FIRESTORE_DATABASE,
                    "N_ANCHOR": N_ANCHOR, "N_SOBOL": N_SOBOL,
                    "N_MOBO": N_MOBO, "N_TOTAL": N_TOTAL,
                    "maxRoundNumber": space.MAX_ROUND_NUMBER})
