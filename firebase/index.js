/**
 * Cloud Functions glue between the app's Firestore writes and the optimizer
 * service on Cloud Run. Deploy from a standard Firebase `functions/` folder
 * (see README.md in this directory).
 *
 * Set CLOUD_RUN_URL in `functions/.env` to the optimizer's Cloud Run URL.
 */

const { setGlobalOptions } = require("firebase-functions/v2");
// Match your Firestore database's location (e.g. nam5 → us-central1).
setGlobalOptions({ region: "us-central1" });

const { onDocumentCreated } = require("firebase-functions/v2/firestore");
const { initializeApp } = require("firebase-admin/app");

initializeApp();

const CLOUD_RUN_URL = process.env.CLOUD_RUN_URL;

/**
 * When a new user document is created (participant registers), call /registerUser
 * so step_1 is written from the Sobol seed before the app even connects.
 */
exports.registerUserOnCreate = onDocumentCreated(
  // The optimizer image is large (CPU torch), so a cold start can take well over
  // the 60 s default. retry:true is safe because the service writes proposals
  // with create() (first-wins), so a redelivery cannot double-write.
  { document: "users/{userId}", timeoutSeconds: 300, retry: true },
  async (event) => {
    const userId = event.params.userId;

    if (!CLOUD_RUN_URL) {
      console.log("CLOUD_RUN_URL missing");
      return;
    }

    const res = await fetch(`${CLOUD_RUN_URL}/registerUser`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId }),
    });

    const txt = await res.text();
    console.log(`[${userId}] registerUser response:`, res.status, txt);
    // 4xx = the optimizer REFUSED deterministically (e.g. a design its rules
    // mirror rejects): retrying would only repeat the refusal, so log loudly
    // and stop. 5xx = transient; throw so the retry actually happens — without
    // it the participant silently never receives a round-1 design.
    if (res.status >= 400 && res.status < 500) {
      console.error(`[${userId}] registerUser REFUSED (${res.status}, not retried): ${txt}`);
      return;
    }
    if (res.status >= 500) {
      throw new Error(`registerUser failed with ${res.status}: ${txt}`);
    }
  }
);

/**
 * When the app writes a completed trial, ask the optimizer for the next config.
 */
exports.updatePolicyOnResult = onDocumentCreated(
  { document: "interventionResults/{resultId}", timeoutSeconds: 240, retry: true },
  async (event) => {
    const data = event.data.data();
    const userId = data.pid;

    if (!CLOUD_RUN_URL) {
      console.log("CLOUD_RUN_URL missing");
      return;
    }

    // Attention check failed → ignore this result, repeat the same step
    if (data.attentionCheckPassed === false) {
      console.log(`[${userId}] Attention check failed — skipping optimizer, step repeated.`);
      return;
    }

    // The optimizer reads all fields directly from Firestore (Admin SDK).
    // We only need to tell it which user triggered the update.
    const payload = {
      userId,
      type:      "interventionResult",
      phaseStep: data.phaseStep ?? null,
    };

    const res = await fetch(`${CLOUD_RUN_URL}/updatePolicy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const txt = await res.text();
    console.log("Optimizer response:", res.status, txt);
    // 4xx = a deterministic refusal (the optimizer answers 422 for a design its
    // rules mirror rejects). Retrying would re-run the whole GP fit for the
    // same answer, for up to 24 h — so log it as an error and stop here.
    if (res.status >= 400 && res.status < 500) {
      console.error(`[${userId}] updatePolicy REFUSED (${res.status}, not retried): ${txt}`);
      return;
    }
    // 5xx = transient: the optimizer never wrote the next design. Throwing is
    // the only way the participant gets another chance at it.
    if (res.status >= 500) {
      throw new Error(`updatePolicy failed with ${res.status}: ${txt}`);
    }
  }
);
