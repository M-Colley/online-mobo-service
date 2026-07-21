"""
VAM MOBO optimizer service (Cloud Run).

Same Firestore-mediated architecture as the original optimizer_service:
  app -> interventionResults -> Cloud Function -> POST /updatePolicy -> here,
  here -> parameterValues -> app snapshot listener.

Only the search space and the parameter field handling changed — the whole
communication layer (Cloud Functions, locks, idempotency, hypervolume logging)
is untouched. The search space lives entirely in space.py; the GP/acqf
logic lives in optimizer_core.py. Both are Firestore-free and unit-testable.
"""

import logging
import os
import threading
from collections import defaultdict

import torch
from flask import Flask, request, jsonify
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

# ── Trial-budget constants ───────────────────────────────────────────────────
# Rule from the study memo: "first 2n+1 trials are the same (Sobol), then the
# personalised optimisation happens in the subsequent trials." n = D here, so
# N_SOBOL adapts automatically if you add/remove parameters in space.py.
N_SOBOL = int(os.environ.get("N_SOBOL", str(2 * (space.D + 1))))
N_MOBO = int(os.environ.get("N_MOBO", "5"))
N_TOTAL = int(os.environ.get("N_TOTAL", str(N_SOBOL + N_MOBO)))

REF_POINT_HV = torch.tensor(core.REF_POINT, dtype=torch.double)

# Per-user lock to serialise concurrent requests for the same user.
_user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


# ── Hypervolume tracking ─────────────────────────────────────────────────────
def log_hypervolume(user_id: str, Y: torch.Tensor, phase_step: int, obs_count: int) -> float:
    pareto_Y = Y[is_non_dominated(Y)]
    hv = float(Hypervolume(ref_point=REF_POINT_HV).compute(pareto_Y))
    docs = (
        db.collection("interventionResults")
        .where(filter=FieldFilter("pid", "==", user_id))
        .where(filter=FieldFilter("phaseStep", "==", phase_step))
        .stream()
    )
    saved = False
    for doc in docs:
        if doc.to_dict().get("attentionCheckPassed") is not False:
            doc.reference.update({"hypervolume": hv})
            saved = True
            break
    if saved:
        log.info("user=%s obs=%d phase_step=%d hypervolume=%.4f — saved", user_id, obs_count, phase_step, hv)
    else:
        log.warning("user=%s phase_step=%d — no valid interventionResults doc found, hv not saved", user_id, phase_step)
    return hv


# ── Firestore helpers ────────────────────────────────────────────────────────
def load_observations(user_id: str):
    """Return (X_model [n,D], Y [n,m], last_phase_step, observed_keys) or Nones.

    Reads the VAM parameter fields (names come from space.PARAM_NAMES) and
    the objective fields (space.OBJECTIVE_FIELDS) straight from Firestore.
    """
    docs = list(
        db.collection("interventionResults")
        .where(filter=FieldFilter("pid", "==", user_id))
        .order_by("phaseStep")
        .stream()
    )
    if not docs:
        return None, None, None, set()

    rows_x, rows_y, observed_keys = [], [], set()
    last_phase_step = None
    seen_steps = set()  # Cloud Functions is at-least-once → deduplicate
    for doc in docs:
        d = doc.to_dict()
        if d.get("attentionCheckPassed") is False:
            continue
        step = d.get("phaseStep")
        if step in seen_steps:
            log.warning("load_observations: duplicate phaseStep=%s for %s — skipping", step, user_id)
            continue
        # One malformed doc (missing a param field, an unknown pattern category,
        # a non-numeric value) must not take down the whole participant's update.
        try:
            x_row = space.to_model_row(d)
            y_row = [float(d[f]) for f in space.OBJECTIVE_FIELDS]
            key = space.obs_key(d)
        except (KeyError, ValueError, TypeError) as e:
            log.warning("load_observations: skipping malformed doc for %s step=%s: %s", user_id, step, e)
            continue
        seen_steps.add(step)
        last_phase_step = step
        rows_x.append(x_row)
        rows_y.append(y_row)
        observed_keys.add(key)

    X = torch.tensor(rows_x, dtype=torch.double)
    Y = torch.tensor(rows_y, dtype=torch.double)
    return X, Y, last_phase_step, observed_keys


def write_next_params(user_id: str, raw: dict, phase: str, phase_step: int) -> bool:
    """Write next parameters as a new parameterValues doc (create() = first-wins)."""
    ref = db.collection("parameterValues").document(f"{user_id}_step_{phase_step}")
    doc = {
        "pid": user_id,
        "phase": phase,
        "phaseStep": phase_step,
        "createdAt": firestore.SERVER_TIMESTAMP,
        **raw,  # all VAM parameter fields
    }
    try:
        ref.create(doc)
        return True
    except Exception:
        log.warning("write_next_params: step_%d already exists for %s — duplicate suppressed", phase_step, user_id)
        return False


# ── Registration endpoint ────────────────────────────────────────────────────
@app.post("/registerUser")
def register_user():
    payload = request.get_json(force=True)
    user_id = payload.get("userId")
    if not user_id:
        return jsonify({"ok": False, "error": "missing userId"}), 400

    with _user_locks[user_id]:
        step_ref = db.collection("parameterValues").document(f"{user_id}_step_1")
        if step_ref.get().exists:
            log.info("registerUser: step_1 already exists for %s — skipping", user_id)
            return jsonify({"ok": True, "skipped": True})

        raw = space.sobol_next(0)
        write_next_params(user_id, raw, "exploration", 1)
        log.info("registerUser: wrote step_1 for %s %s", user_id, raw)
        return jsonify({"ok": True, "params": raw})


# ── Main endpoint ────────────────────────────────────────────────────────────
@app.post("/updatePolicy")
def update_policy():
    payload = request.get_json(force=True)
    user_id = payload.get("userId", "test_user_1")

    if payload.get("type") != "interventionResult":
        return jsonify({"ok": True, "ignored": True})

    with _user_locks[user_id]:
        # 1. Load history.
        X_model, Y, last_phase_step, observed_keys = load_observations(user_id)
        obs_count = 0 if X_model is None else len(X_model)

        hv = None
        if Y is not None and obs_count >= 1 and last_phase_step is not None:
            hv = log_hypervolume(user_id, Y, last_phase_step, obs_count)

        # Study complete.
        if obs_count >= N_TOTAL:
            db.collection("users").document(user_id).set(
                {"studyCompleted": True, "studyCompletedAt": firestore.SERVER_TIMESTAMP}, merge=True
            )
            log.info("user=%s study COMPLETED (%d observations)", user_id, obs_count)
            return jsonify({"ok": True, "studyCompleted": True})

        next_phase_step = obs_count + 1

        # Idempotency: if a concurrent invocation already wrote this step, stop.
        existing = db.collection("parameterValues").document(f"{user_id}_step_{next_phase_step}").get()
        if existing.exists:
            log.info("user=%s step_%d already exists — skipping duplicate", user_id, next_phase_step)
            return jsonify({"ok": True, "skipped": True})

        # 2. Choose next candidate.
        phase = "exploration"
        try:
            if obs_count < N_SOBOL or X_model is None:
                raw = space.sobol_next(obs_count)
                # keep Sobol on-grid AND novel
                if space.obs_key(raw) in observed_keys:
                    raw = space.random_feasible(seed=obs_count)
            else:
                raw = core.choose_next(X_model, Y, observed_keys, dedup_seed=obs_count)
                phase = "optimization"
        except Exception:
            log.exception("MOBO failed for %s — falling back to Sobol", user_id)
            raw = space.sobol_next(obs_count)
            phase = "exploration-fallback"

        # 3. Write.
        write_next_params(user_id, raw, phase, next_phase_step)
        log.info("user=%s obs=%d phase=%s next_step=%d next=%s",
                 user_id, obs_count, phase, next_phase_step, raw)

        return jsonify({
            "ok": True,
            "phase": phase,
            "obsCount": obs_count,
            "hypervolume": hv,
            "lastPhaseStep": last_phase_step,
            "nextPhaseStep": next_phase_step,
            "nextParams": raw,
        })


@app.get("/health")
def health():
    return jsonify({"ok": True, "space": space.PARAM_NAMES,
                    "database": FIRESTORE_DATABASE,
                    "N_SOBOL": N_SOBOL, "N_TOTAL": N_TOTAL})
