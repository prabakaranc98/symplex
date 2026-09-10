// Render-function smoke check against a real API snapshot; not a browser layout test.
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
const state = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const root = path.resolve("symplex/interfaces/web");
const element = () => ({
  innerHTML: "",
  textContent: "",
  value: "",
  classList: { add() {}, remove() {}, toggle() {} },
  focus() {},
  showModal() {},
  close() {},
  setAttribute() {},
  removeAttribute() {},
});
const elements = new Map();
const sandbox = {
  console,
  URL,
  Date,
  Intl,
  setTimeout() {},
  clearTimeout() {},
  document: {
    querySelector(s) {
      if (!elements.has(s)) elements.set(s, element());
      return elements.get(s);
    },
    addEventListener() {},
  },
  localStorage: {
    getItem() {
      return null;
    },
    setItem() {},
  },
  innerWidth: 1440,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root, "common.js"), "utf8"), sandbox);
for (const file of fs
  .readdirSync(path.join(root, "views"))
  .filter((f) => f.endsWith(".js"))) {
  vm.runInContext(
    fs.readFileSync(path.join(root, "views", file), "utf8"),
    sandbox,
  );
}
sandbox.snapshot = state;
vm.runInContext("state=snapshot;", sandbox);
const problems = [
  null,
  ...state.records
    .filter((r) => r.kind === "workspace_problem" && !r.stale)
    .map((r) => r.id),
];
let checks = 0;
for (const problem of problems) {
  sandbox.problemId = problem;
  vm.runInContext("selected=problemId;", sandbox);
  for (const view of [
    "overview",
    "problemView",
    "evidenceView",
    "systemGraphView",
    "complexityView",
    "evolutionView",
    "methodLabView",
    "modelsView",
    "scenariosView",
    "runsView",
    "deliverablesView",
    "harnessView",
    "workflowView",
    "connectorsView",
    "labView",
    "sceneView",
    "architectureView",
  ]) {
    const html = vm.runInContext(view + "()", sandbox);
    if (typeof html !== "string" || html.includes("NaN"))
      throw Error("Invalid rendered " + view);
    checks++;
  }
}
console.log(
  `Passed ${checks} view renders across ${problems.length} problem states. Browser layout and interaction still require separate QA.`,
);
