# Firebase glue (reference copies)

The optimizer never talks to the app directly — a Cloud Function relays
Firestore writes to the Cloud Run service. This folder holds reference copies of
that glue so the repo is usable end-to-end:

| File | What it is |
|------|-----------|
| `index.js` | The two v2 Firestore triggers: `registerUserOnCreate` → `POST /registerUser`, `updatePolicyOnResult` → `POST /updatePolicy`. |
| `package.json` | Node 24, firebase-functions v7, firebase-admin v13. |
| `app-rules-local.txt` | Byte-exact copy of the current app repository's rules (17-round cap), used by offline contract tests. Deployment status is unverified. Never deploy from this service repository. |
| `app-rules-snapshot.txt` | Historical, **read-only** copy of ruleset `6700114d` (2026-09-17, 18-round cap). `inspect_db.py` CHECK 0 compares it with the live rules. Keep it until live rules are inspected via the Firebase Rules REST API (`releases` → `rulesets/<id>`). Never deploy it. |
| `test_index.cjs` | Offline trigger routing/retry regressions, run with `node --test firebase/test_index.cjs` from the repository root. |
| `firestore.indexes.json` | **All four** live composite indexes — one the optimizer needs (`interventionResults` `pid`+`phaseStep`) and three the app needs, including `parameterValues(pid, schemaVersion, createdAt DESC)` for its listener. The deploy is declarative: anything missing from this file is **deleted** from the database. |

## Wiring it up

From a standard Firebase project directory (`firebase init functions` if you
don't have one):

```bash
cp index.js package.json <your-project>/functions/
cp firestore.indexes.json <your-project>/

# point the functions at your deployed optimizer
echo "CLOUD_RUN_URL=https://<your-cloud-run-url>" > <your-project>/functions/.env

firebase deploy --only functions,firestore:indexes --project <PROJECT_ID>
```

Notes:
- List and merge the current live indexes before deploying indexes: deployment
  deletes composite indexes omitted from the manifest. The checked-in list is a
  historical snapshot, not proof of the current database's contents.
- Missing `CLOUD_RUN_URL`, network failures, HTTP 408/429 and 5xx cause a retry.
  Deterministic HTTP refusals (for example 422) are logged without retrying.
- The triggers fire on the `(default)` Firestore database; a **named** database
  needs `database: "<name>"` in each trigger's options.
- Keep the functions' region consistent with your database location
  (e.g. `nam5` → `us-central1`); set it in `setGlobalOptions` in `index.js`.
- `functions/.env` holds `CLOUD_RUN_URL` (firebase-functions v7 removed
  `functions.config()`). Never commit `.env`.
