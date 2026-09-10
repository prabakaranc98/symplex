// Offline DOM-free checks for operator gate rendering and explicit API writes.
// This does not exercise browser layout, native form validation, or a live API.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const listeners = new Map(), requests = [], toasts = [];
const sandbox = {
  console, Date, Intl, URL,
  setTimeout() {},
  localStorage: { getItem() { return null; }, setItem() {} },
  document: {
    addEventListener(name, callback) {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(callback);
    },
    querySelector() { return { textContent: "", classList: { add() {}, remove() {} } }; },
  },
  FormData: class {
    constructor(form) { this.fields = form.fields; }
    get(name) { return this.getAll(name)[0] ?? null; }
    getAll(name) {
      const value = this.fields[name];
      return value == null ? [] : Array.isArray(value) ? value : [value];
    }
  },
  mockApi: async (route, body) => {
    requests.push({ route, ...(body === undefined ? {} : { body: JSON.parse(JSON.stringify(body)) }) });
    return body === undefined ? sandbox.gate : { id: "record_written", status: "approved" };
  },
  recordToast: value => toasts.push(value),
};
vm.createContext(sandbox);
for (const file of ["common.js", "views/evolution.js", "views/methodlab.js"])
  vm.runInContext(fs.readFileSync("symplex/interfaces/web/" + file, "utf8"), sandbox);
vm.runInContext("api=mockApi; refresh=async()=>{}; toast=recordToast;", sandbox);
const problem = { id: "problem_one", kind: "workspace_problem", stale: false, data: { question: "Test problem" } };
const candidate = { id: "candidate_one", kind: "method_candidate", parent: problem.id, data: {
  name: "Architect instruction", changes: [{ role: "complexity_architect", current_failure: "Missing references", proposed_instruction: "Check evidence references", expected_effect: "Contract pass", regression_risk: "No scientific gain established" }],
} };
sandbox.fixture = { records: [problem, candidate] };
vm.runInContext("state=fixture; selected='problem_one';", sandbox);
const stageNames = ["development", "regression", "holdout"];
const missing = {
  candidate_id: candidate.id, state: "blocked", eligible: false,
  blockers: ["Missing holdout report <script>unsafe</script>"],
  stages: Object.fromEntries(stageNames.map(stage => [stage, { status: "missing", protocol_id: stage + "_protocol", report_id: null }])),
  report_ids: {}, reviews: [], deployments: [], rollbacks: [], observations: [],
};
function render(gate) {
  sandbox.gate = gate;
  vm.runInContext("methodLabCache.set('candidate_one',{result:gate,fetched:'2026-09-10T12:00:00Z'});", sandbox);
  return vm.runInContext("methodLabView()", sandbox);
}
let html = render(missing);
assert.equal(requests.length, 0, "Rendering must never issue requests");
assert.match(html, /value="approve" disabled/);
assert.equal((html.match(/No report<\/span>/g) || []).length, 3);
assert.match(html, /data-id="holdout_protocol"/);
assert.doesNotMatch(html, /data-method-form="canary"/);
assert.doesNotMatch(html, /Host gate requirements met/);
assert.match(html, /&lt;script&gt;unsafe&lt;\/script&gt;/);
assert.doesNotMatch(html, /<script>unsafe/);
assert.match(html, /absence of observations is not evidence/);
await vm.runInContext("methodLabRefresh()", sandbox);
assert.equal(requests.length, 1);
assert.equal(requests[0].route, "/method-lab/candidate_one");
assert.equal(requests[0].body, undefined, "Gate refresh is read-only");

const ready = { ...missing, state: "ready_for_review", eligible: true, blockers: [],
  stages: Object.fromEntries(stageNames.map(stage => [stage, {
    status: "verified", protocol_id: stage + "_protocol", report_id: stage + "_report",
    task_count: 4, parent_passes: 2, child_passes: 3, paired_regressions: 0, matched_resources_verified: true,
  }])),
  report_ids: Object.fromEntries(stageNames.map(stage => [stage, stage + "_report"])),
};
html = render(ready);
assert.match(html, /value="approve" >/);
assert.equal((html.match(/Verified by host/g) || []).length, 3);
assert.equal((html.match(/2 \/ 3/g) || []).length, 3);
assert.doesNotMatch(html, /data-method-form="canary"/);
const review = { id: "review_one", kind: "method_operator_review", data: { status: "approved", actor: "A < B", reason: "Review \"evidence\"", problem_ids: [problem.id] } };
html = render({ ...ready, reviews: [review] });
assert.match(html, /data-method-form="canary"/);
assert.match(html, /value="review_one"/);
assert.match(html, /A &lt; B/);
assert.match(html, /Review &quot;evidence&quot;/);
html = render({ ...ready, reviews: [review, { id: "review_two", data: { status: "rejected" } }] });
assert.doesNotMatch(html, /data-method-form="canary"/, "A rejected latest review cannot expose an older approval for activation");
const deployment = { id: "deployment_one", kind: "method_deployment", data: { mode: "canary", status: "active", problem_ids: [problem.id] } };
html = render({ ...ready, deployments: [deployment] });
assert.match(html, /data-method-form="rollback"/);
html = render({ ...ready, deployments: [deployment], rollbacks: [{ id: "rollback_one", data: { deployment_id: deployment.id } }] });
assert.doesNotMatch(html, /data-method-form="rollback"/);
html = render({ ...ready, deployments: [{ ...deployment, data: { ...deployment.data, mode: "shadow" } }] });
assert.doesNotMatch(html, /data-method-form="rollback"/);
assert.match(html, /does not execute a replay/);

const handler = listeners.get("submit").at(-1);
const baseFields = { actor: "Operator", reason: "Reviewed limitations", problem_ids: [problem.id] };
async function submit(operation, fields) {
  const message = { textContent: "" }, submitter = { disabled: false, isConnected: true };
  const event = {
    target: { dataset: { methodForm: operation, candidate: candidate.id }, fields, querySelector() { return message; } },
    submitter, prevented: false, preventDefault() { this.prevented = true; },
  };
  requests.length = 0;
  await handler(event);
  assert.equal(event.prevented, true);
  assert.equal(submitter.disabled, false);
  assert.equal(vm.runInContext("busy", sandbox), false);
  return { writes: requests.filter(r => r.body !== undefined), message: message.textContent };
}
const reviewFields = { ...baseFields, decision: "approve", execution_failure_limit: "2", numerical_failure_limit: "3",
  ...Object.fromEntries(stageNames.map(stage => ["report_" + stage, stage + "_report"])),
};
let result = await submit("review", reviewFields);
assert.deepEqual(result.writes, [{ route: "/method-lab/review", body: {
  ...baseFields, candidate_id: candidate.id, decision: "approve", report_ids: ready.report_ids,
  execution_failure_limit: 2, numerical_failure_limit: 3,
} }], "An explicit review writes only a review, never activates a canary");
result = await submit("review", { ...reviewFields, decision: "reject", problem_ids: [] });
assert.equal(result.writes.length, 1);
assert.equal(result.writes[0].body.decision, "reject");
assert.deepEqual(result.writes[0].body.problem_ids, []);
result = await submit("review", { ...reviewFields, problem_ids: [] });
assert.equal(result.writes.length, 0);
assert.match(result.message, /Select at least one problem/);
result = await submit("shadow", baseFields);
assert.deepEqual(result.writes, [{ route: "/method-lab/shadow", body: { ...baseFields, candidate_id: candidate.id } }]);
result = await submit("canary", { ...baseFields, review_id: review.id });
assert.deepEqual(result.writes, [{ route: "/method-lab/canary", body: { actor: baseFields.actor, reason: baseFields.reason, review_id: review.id } }]);
result = await submit("rollback", { ...baseFields, deployment_id: deployment.id });
assert.deepEqual(result.writes, [{ route: "/method-lab/rollback", body: { actor: baseFields.actor, reason: baseFields.reason, deployment_id: deployment.id } }]);
result = await submit("canary", { ...baseFields, review_id: "" });
assert.equal(result.writes.length, 0);
assert.match(result.message, /Select an approved review/);
result = await submit("shadow", { ...baseFields, actor: "  " });
assert.equal(result.writes.length, 0);
assert.match(result.message, /operator name/);
console.log("Passed Method Lab gate rendering, escaping, read-only refresh, and explicit review/shadow/canary/rollback action checks. No live calls.");
