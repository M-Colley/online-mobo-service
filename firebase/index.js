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

async function callOptimizer(endpoint, payload) {
  if (!CLOUD_RUN_URL) {
    // A successful acknowledgement discards the event permanently. Leave the
    // event retryable while deployment configuration is being repaired.
    throw new Error("CLOUD_RUN_URL missing");
  }
  const res = await fetch(`${CLOUD_RUN_URL.replace(/\/$/, "")}/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(210_000),
  });
  const txt = await res.text();
  console.log(`[${payload.userId}] ${endpoint} response:`, res.status, txt);
  if (res.status === 408 || res.status === 429 || res.status >= 500) {
    throw new Error(`${endpoint} failed with ${res.status}: ${txt}`);
  }
  if (!res.ok) {
    console.error(`[${payload.userId}] ${endpoint} REFUSED (${res.status}, not retried): ${txt}`);
  }
}

/**
 * When a new user document is created (participant registers), call /registerUser
 * so step_1 is written from the anchor before the app connects.
 */
exports.registerUserOnCreate = onDocumentCreated(
  // The optimizer image is large (CPU torch), so a cold start can take well over
  // the 60 s default. retry:true is safe because the service writes proposals
  // with create() (first-wins), so a redelivery cannot double-write.
  { document: "users/{userId}", timeoutSeconds: 300, retry: true },
  async (event) => {
    const userId = event.params.userId;
    await callOptimizer("registerUser", { userId });
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

    // No attention-check short-circuit any more: the app dropped attention
    // checks (ruleset 6700114d, 2026-09-17) and never repeats a round, so EVERY
    // result must reach the optimizer or the participant waits forever for the
    // next design. (A legacy result explicitly marked attentionCheckPassed:false
    // is still forwarded; the service excludes it from training itself.)

    // The optimizer reads all fields directly from Firestore (Admin SDK).
    // We only need to tell it which user triggered the update.
    const payload = {
      userId,
      type:      "interventionResult",
      phaseStep: data.phaseStep ?? null,
    };

    await callOptimizer("updatePolicy", payload);
  }
);
