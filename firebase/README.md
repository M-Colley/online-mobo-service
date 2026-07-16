# Firebase glue (reference copies)

The optimizer never talks to the app directly — a Cloud Function relays
Firestore writes to the Cloud Run service. This folder holds reference copies of
that glue so the repo is usable end-to-end:

| File | What it is |
|------|-----------|
| `index.js` | The two v2 Firestore triggers: `registerUserOnCreate` → `POST /registerUser`, `updatePolicyOnResult` → `POST /updatePolicy`. |
| `package.json` | Node 24, firebase-functions v7, firebase-admin v13. |
| `firestore.indexes.json` | The `pid`+`phaseStep` composite indexes the optimizer's queries need. |

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
- The triggers fire on the `(default)` Firestore database; a **named** database
  needs `database: "<name>"` in each trigger's options.
- Keep the functions' region consistent with your database location
  (e.g. `nam5` → `us-central1`); set it in `setGlobalOptions` in `index.js`.
- `functions/.env` holds `CLOUD_RUN_URL` (firebase-functions v7 removed
  `functions.config()`). Never commit `.env`.
