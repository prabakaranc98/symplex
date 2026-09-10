let sceneMountRevision = 0;
function sceneView() {
  const scenes = related("file_blob").filter(r => r.data.filename === "symplex_scene.json"),
    csvs = related("file_blob").filter(r => r.data.filename.endsWith(".csv") && r.data.basis === "generated");
  return `<div class="scene-page">` + head("SIMULATION EXPLORER", "Explore recorded dynamics.", "Replay model outputs, compare trajectories and inspect the coordinates behind each view.") +
    `<div class="card scene-workspace">${scenes.length ? `<div class="scene-picker"><div><label for="scene-select">Recorded scene</label><p id="scene-facts" class="tiny" role="status">Loading recorded state…</p></div><select id="scene-select">${scenes.map(r=>`<option value="${esc(r.id)}">${esc(r.data.filename)} · ${esc(new Date(r.created * 1000).toLocaleString())}</option>`).join("")}</select></div><div id="scene-canvas" class="scene-canvas" aria-label="Interactive simulation replay" aria-busy="true"><p class="scene-loading">Loading the recorded scene…</p></div><div id="scene-metadata"></div>` : empty("No recorded 3D state yet", "Astra can produce a scene from its simulation. Once computed outputs are available, you can replay trajectories here or map a CSV to phase space below.", current() ? btn("Open modeling studio", "nav", "lab") : btn("Create problem", "new"))}</div>` +
    `<details class="card scene-build"><summary>Create a view from computed data<span>Optional · map recorded CSV columns</span></summary><div class="scene-build-body"><p>Choose which recorded quantities define time and the three axes. This creates a view of existing computation without a new model run. A phase-space projection needs a coordinate model before it can be interpreted as physical geometry.</p>${csvs.length ? `<form id="scene-form"><label for="scene-csv">Computed data file</label><select id="scene-csv">${csvs.map(r=>`<option value="${esc(r.id)}">${esc(r.data.filename)} · ${esc(r.id.slice(-8))}</option>`).join("")}</select><div id="scene-columns" class="scene-fields"></div><label for="scene-filters">Select a scenario or run (optional JSON filter)</label><input id="scene-filters" value="{}" placeholder='{"scenario":"baseline"}' aria-describedby="scene-filter-help"/><p id="scene-filter-help" class="tiny">Use exact row values when a table contains repeated clocks. Recorded frames keep their original times; missing samples remain absent.</p><button type="submit" class="primary">Create recorded view ↗</button></form><details class="scene-sample-details"><summary>Inspect first recorded rows</summary><div id="scene-samples" class="tiny"></div></details>` : '<p class="muted">Run the modeling agent to produce numerical CSV outputs first.</p>'}</div></details></div>`;
}
async function loadScenePlayer() {
  const container = $("#scene-canvas"), selector = $("#scene-select");
  if (!container || !selector) return;
  const revision = ++sceneMountRevision;
  container.setAttribute("aria-busy", "true");
  try {
    const result = await api("/scenes/" + selector.value);
    if (revision !== sceneMountRevision || !container.isConnected) return;
    if (!window.SymplexScene) throw Error("The 3D renderer is not available in this build.");
    window.SymplexScene.dispose();
    const scene = result.scene;
    $("#scene-facts").textContent = `${scene.frames.length} recorded frames · ${scene.agents.length} trajectories`;
    $("#scene-metadata").innerHTML = `<details class="scene-provenance"><summary>Inspect source data and execution</summary><p>Follow this view back to the recorded data and the run that produced it. The scene’s declared basis and interpretation limits remain visible in the replay.</p><div class="actions">${btn("Source data", "artifact", result.source_id)}${btn("Executed run", "artifact", result.run_id)}</div></details>`;
    const controller = window.SymplexScene.mount(container, scene);
    controller.setFrame(scene.frames.length - 1);
  } catch (error) {
    if (revision === sceneMountRevision && container.isConnected) {
      container.textContent = error.message;
      $("#scene-facts").textContent = "Recorded scene unavailable";
    }
  } finally { if (revision === sceneMountRevision && container.isConnected) container.setAttribute("aria-busy", "false"); }
}
async function loadSceneColumns() {
  const selector = $("#scene-csv");
  if (!selector) return;
  try {
    const value = selector.value, result = await api("/scenes/columns/" + value);
    if (!$("#scene-csv") || $("#scene-csv").value !== value) return;
    const fields = [["time", "Time"], ["x", "X quantity"], ["y", "Y quantity"], ["z", "Z quantity"], ["group", "Trajectory identity"]];
    $("#scene-columns").innerHTML = fields.map(([id,label]) => `<div><label for="scene-${id}">${label}</label><select id="scene-${id}" ${id === "group" ? '' : 'required'}>${id === "group" ? '<option value="">Single trajectory</option>' : '<option value="" disabled selected>Choose a column</option>'}${result.columns.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join("")}</select></div>`).join("");
    $("#scene-samples").textContent = JSON.stringify(result.samples, null, 2);
  } catch(error) { toast(error.message); }
}
function mountSceneView() { loadScenePlayer(); loadSceneColumns(); }
document.addEventListener("change", e => {
  if (e.target.id === "scene-select") loadScenePlayer();
  if (e.target.id === "scene-csv") loadSceneColumns();
});
document.addEventListener("submit", async e => {
  if (e.target.id !== "scene-form") return;
  e.preventDefault();
  const button = e.submitter; button.disabled = true;
  try {
    const filters = JSON.parse($("#scene-filters").value || "{}");
    await api("/scenes", { id: selected, blob_id: $("#scene-csv").value, time_column: $("#scene-time").value, x_column: $("#scene-x").value, y_column: $("#scene-y").value, z_column: $("#scene-z").value, group_column: $("#scene-group").value || null, filters });
    await refresh(); toast("3D view derived from recorded coordinates.");
  } catch(error) { toast(error.message); }
  finally { if (button.isConnected) button.disabled = false; }
});
