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
  "users/{userId}",
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
  }
);

/**
 * When the app writes a completed trial, ask the optimizer for the next config.
 */
exports.updatePolicyOnResult = onDocumentCreated(
  { document: "interventionResults/{resultId}", timeoutSeconds: 300 },
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
  }
);
