const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function handlers({ status = 200, url = "https://optimizer.example/", fetchError } = {}) {
  const calls = [];
  const context = {
    exports: {}, process: { env: { CLOUD_RUN_URL: url } },
    AbortSignal,
    console: { log() {}, error() {} },
    require(name) {
      if (name === "firebase-functions/v2") return { setGlobalOptions() {} };
      if (name === "firebase-functions/v2/firestore") {
        return { onDocumentCreated: (options, handler) => handler };
      }
      if (name === "firebase-admin/app") return { initializeApp() {} };
      throw new Error(`Unexpected dependency ${name}`);
    },
    async fetch(url, options) {
      calls.push({ url, options });
      if (fetchError) throw fetchError;
      return { status, ok: status >= 200 && status < 300, text: async () => "response" };
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "index.js"), "utf8"), context);
  return { ...context.exports, calls };
}

const result = { params: { resultId: "r1" }, data: { data: () => ({ pid: "101", phaseStep: 1 }) } };

test("both triggers forward participant and round to the right endpoint", async () => {
  const h = handlers();
  await h.registerUserOnCreate({ params: { userId: "101" } });
  await h.updatePolicyOnResult(result);
  assert.equal(h.calls[0].url, "https://optimizer.example/registerUser");
  assert.deepEqual(JSON.parse(h.calls[0].options.body), { userId: "101" });
  assert.equal(h.calls[1].url, "https://optimizer.example/updatePolicy");
  assert.deepEqual(JSON.parse(h.calls[1].options.body), {
    userId: "101", type: "interventionResult", phaseStep: 1,
  });
  assert.ok(h.calls[1].options.signal instanceof AbortSignal);
});

test("temporary HTTP/network failures and missing configuration remain retryable", async () => {
  for (const status of [408, 429, 500, 503]) {
    const h = handlers({ status });
    await assert.rejects(h.updatePolicyOnResult(result), new RegExp(String(status)));
    await assert.rejects(h.registerUserOnCreate({ params: { userId: "101" } }));
  }
  await assert.rejects(handlers({ url: "" }).updatePolicyOnResult(result), /CLOUD_RUN_URL missing/);
  await assert.rejects(handlers({ fetchError: new Error("network") }).updatePolicyOnResult(result), /network/);
});

test("deterministic refusals are acknowledged without a retry loop", async () => {
  const h = handlers({ status: 422 });
  await h.updatePolicyOnResult(result);
  await h.registerUserOnCreate({ params: { userId: "101" } });
});
