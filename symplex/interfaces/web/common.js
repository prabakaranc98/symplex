"use strict";
const $ = (s) => document.querySelector(s),
  esc = (v) =>
    String(v ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
const pretty = (v) => String(v ?? "").replaceAll("_", " "),
  money = (n) => "$" + Number(n || 0).toFixed(3);
let state = null,
  page = "overview",
  selected = localStorage.getItem("symplex.problem") || null,
  busy = false,
  chatBusy = false,
  filter = "all",
  messageSignature = "";
const nav = [
  ["overview", "◈", "Overview"],
  ["problem", "◎", "Problem & hypotheses"],
  ["evidence", "▤", "Evidence"],
  ["system", "⌘", "Solution outline"],
  ["complexity", "⧉", "Complex system"],
  ["evolution", "⇄", "Design & improvement"],
  ["methodlab", "⇌", "Method Lab"],
  ["models", "◇", "Solution components"],
  ["scenarios", "⌁", "Scenarios"],
  ["lab", "◉", "Modeling studio"],
  ["scene", "◒", "3D trajectories"],
  ["runs", "↗", "Runs & evaluation"],
  ["deliverables", "▧", "Deliverables"],
  ["workflow", "⤴", "Agent workflow"],
  ["connectors", "⊞", "Connectors"],
  ["architecture", "⤧", "Engine map"],
  ["harness", "⚙", "Runtime settings"],
];
const records = (kind) => (state?.records || []).filter((r) => r.kind === kind),
  related = (kind) =>
    records(kind).filter((r) => r.parent === selected && !r.stale),
  current = () => records("workspace_problem").find((r) => r.id === selected);
const latest = (kind) => related(kind).at(-1),
  allLatest = (kind) =>
    records(kind)
      .filter((r) => !r.stale)
      .at(-1);
const tag = (text, warn = false) =>
  `<span class="badge ${warn ? "warn" : ""}">${esc(pretty(text))}</span>`;
const btn = (label, action, id = "", cls = "secondary") =>
  `<button class="${cls}" data-action="${action}" data-id="${esc(id)}">${label}</button>`;
const list = (items) =>
  `<ul class="detail-list">${(items || []).map((t) => `<li>${esc(t)}</li>`).join("")}</ul>`;
const head = (eyebrow, title, desc, action = "") =>
  `<div class="page-title"><div class="intro"><span class="eyebrow">${eyebrow}</span><h1>${esc(title)}</h1><p>${esc(desc)}</p></div>${action}</div>`;
const empty = (title, text, action = "") =>
  `<div class="empty"><h3>${esc(title)}</h3><p>${esc(text)}</p>${action}</div>`;
const artifact = (r, label = "Inspect artifact ↗") =>
  r ? btn(label, "artifact", r.id) : "";
function urlLink(url, label) {
  try {
    const u = new URL(url);
    if (!["http:", "https:"].includes(u.protocol)) return esc(label);
    return `<a href="${esc(u.href)}" target="_blank" rel="noreferrer">${esc(label)}</a>`;
  } catch {
    return esc(label);
  }
}
// A deliberately small formatter: raw HTML is always text; links resolve only
// to HTTP(S) sources or actual recorded files from the displayed computation.
function resultText(text, fileIds = []) {
  const files = records("file_blob").filter(r => fileIds.includes(r.id));
  function inline(value) {
    return String(value).split(/(\[[^\]\n]+\]\([^)\n]+\)|\*\*[^*\n]+\*\*|`[^`\n]+`)/g).map(part => {
      const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (link) {
        const file = files.find(r => link[2] === "sandbox:/mnt/data/" + r.data.filename);
        if (file) return `<a href="/api/files/${esc(file.id)}?download=true">${esc(link[1])}</a>`;
        return urlLink(link[2], link[1]);
      }
      if (part.startsWith("**") && part.endsWith("**")) return `<strong>${esc(part.slice(2, -2))}</strong>`;
      if (part.startsWith("`") && part.endsWith("`")) return `<code>${esc(part.slice(1, -1))}</code>`;
      return esc(part);
    }).join("");
  }
  const lines = String(text || "").replace(/([^\n#])(?=#{2,6} )/g, "$1\n").split("\n");
  const output = [];
  const cells = line => line.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
  for (let i = 0; i < lines.length;) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (/^```/.test(line)) {
      const code = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]);
      i++;
      output.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
    } else if (/^#{1,6} /.test(line)) {
      output.push(`<h4>${inline(line.replace(/^#{1,6} /, ""))}</h4>`); i++;
    } else if (line.includes("|") && i + 1 < lines.length && cells(lines[i + 1]).every(c => /^:?-{3,}:?$/.test(c))) {
      const headings = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
      output.push(`<div class="result-table-scroll" tabindex="0" role="region" aria-label="Reported model metrics"><table><thead><tr>${headings.map(c => `<th scope="col">${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(c => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
    } else if (/^[-*] /.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*] /.test(lines[i])) items.push(lines[i++].slice(2));
      output.push(`<ul class="detail-list">${items.map(item => `<li>${inline(item)}</li>`).join("")}</ul>`);
    } else {
      output.push(`<p>${inline(line)}</p>`); i++;
    }
  }
  return output.join("");
}

function toast(text) {
  $("#toast").textContent = text;
  $("#toast").classList.add("visible");
  setTimeout(() => $("#toast").classList.remove("visible"), 6500);
}
async function api(path, body) {
  const options =
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Symplex-Token": state?.token || "",
          },
          body: JSON.stringify(body),
        };
  const r = await fetch("/api" + path, options);
  let data;
  try {
    data = await r.json();
  } catch {
    throw Error("The server returned an unreadable response.");
  }
  if (!r.ok)
    throw Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || data),
    );
  return data;
}
async function refresh(render = true) {
  try {
    state = await api("/state");
    if (selected && !current()) selected = null;
    renderSidebar();
    if (render) renderMain();
    renderChat();
    $("#connection").textContent = state.key_configured
      ? "API key configured"
      : "API key required";
    $("#chat-model").textContent = state.models.chat;
  } catch (e) {
    toast(e.message);
  }
}
function problemLabel(problem) {
  if (!problem) return "Choose a problem";
  const representation = (state?.records || []).filter(r => r.parent === problem.id && !r.stale && ["complex_system", "solution"].includes(r.kind)).at(-1);
  const text = problem.data.title || representation?.data.title || problem.data.question;
  return text.length > 100 ? text.slice(0, 97) + "…" : text;
}
const stableHTML = new WeakMap();
function replaceChangedHTML(element, html) {
  if (!element || stableHTML.get(element) === html) return;
  element.innerHTML = html;
  stableHTML.set(element, html);
}
function renderSidebar() {
  const picker = $("#problem-picker");
  replaceChangedHTML(picker, '<option value="">Choose a problem</option>' + records("workspace_problem").filter(r => !r.stale).map(r => `<option value="${esc(r.id)}" ${r.id === selected ? "selected" : ""}>${esc(problemLabel(r))}</option>`).join(""));
  const counts = {
    evidence: records("evidence").length + records("evidence_note").length + records("evidence_synthesis").length,
    complexity: records("complex_system").length,
    evolution: records("candidate_design").length + records("method_candidate").length + records("experiment_comparison").length,
    methodlab: records("method_candidate").filter(r => !r.stale).length,
    models: records("program").length + records("code_proposal").length,
    runs: state.jobs.length,
    deliverables: records("decision").length + records("deliverable").length + records("decision_brief").length,
  };
  const navButton = ([id, icon, name]) => `<button class="nav-item ${page === id ? "active" : ""}" data-action="nav" data-id="${id}" aria-label="${esc(name)}" ${page === id ? 'aria-current="page"' : ""}><span class="nav-icon" aria-hidden="true">${icon}</span><span>${name}</span>${counts[id] ? `<span class="nav-count">${counts[id]}</span>` : ""}</button>`;
  const groups = [
    ["Understand", ["overview", "problem", "evidence", "complexity"]],
    ["Model & test", ["lab", "scene", "evolution", "runs"]],
    ["Decide", ["deliverables", "methodlab", "architecture"]],
  ];
  const primary = new Set(groups.flatMap(g => g[1]));
  const advanced = nav.filter(n => !primary.has(n[0]));
  replaceChangedHTML($("#navigation"), groups.map(([label, ids]) => `<div class="nav-section-label">${label}</div>${ids.map(id => navButton(nav.find(n => n[0] === id))).join("")}`).join("") + `<details class="nav-details" ${advanced.some(n => n[0] === page) ? "open" : ""}><summary>Engine details</summary>${advanced.map(navButton).join("")}</details>`);
  replaceChangedHTML($("#problem-list"),
    records("workspace_problem")
      .filter((r) => !r.stale)
      .slice()
      .reverse()
      .map(
        (r) =>
          `<button class="problem-link ${selected === r.id ? "selected" : ""}" data-action="select" data-id="${r.id}">· ${esc(problemLabel(r))}</button>`,
      )
      .join("") || '<span class="tiny">No problems saved yet.</span>');
  $("#breadcrumb").textContent =
    nav.find((n) => n[0] === page)?.[2] || "Overview";
}
function datasetCard(d) {
  const loaded = records("research_dataset").find(
    (r) => r.data.name === d.name,
  );
  return `<article class="card"><div class="card-top"><span class="eyebrow">${esc(d.publisher)}</span>${tag(loaded ? "Dataset acquired" : d.status, !loaded && d.status === "specification_only")}</div><h3>${esc(d.name)}</h3><p>${esc(d.use_case)}</p><p>${esc(d.description)}</p><div class="tiny">${esc(d.license)} · ${esc(d.size)}</div><div class="card-footer">${urlLink(d.url, "Source ↗")}${d.id === "bamtwoogle" ? (loaded ? artifact(loaded, "Inspect metadata ↗") : btn("Acquire dataset", "acquire", "", "quiet")) : d.id === "p1" ? btn("Import slice", "import", "", "quiet") : btn("Create problem", "dataset-problem", d.id, "quiet")}</div></article>`;
}
function renderMain() {
  if (typeof window !== "undefined" && window.SymplexScene) window.SymplexScene.dispose();
  if (!state) return;
  const views = {
    overview,
    problem: problemView,
    evidence: evidenceView,
    system: systemGraphView,
    complexity: complexityView,
    evolution: evolutionView,
    methodlab: methodLabView,
    models: modelsView,
    scenarios: scenariosView,
    lab: labView,
    scene: sceneView,
    runs: runsView,
    deliverables: deliverablesView,
    harness: harnessView,
    architecture: architectureView,
    workflow: workflowView,
    connectors: connectorsView,
  };
  const active = state.active.filter((a) => a.status === "running");
  $("#main").innerHTML =
    active
      .map(
        (a) =>
          `<div class="progress-notice"><span><span class="model-dot pulse"></span> ${esc(a.label)} is working…</span>${btn("Cancel", "cancel", a.id, "quiet")}</div>`,
      )
      .join("") + (views[page] || overview)();
  if (page === "scene") mountSceneView();
  renderSidebar();
}
function renderChat() {
  const messages = records("message").filter((r) => r.parent === selected),
    sig = JSON.stringify([selected, messages.map((r) => r.id), chatBusy]);
  if (messageSignature === sig) return;
  messageSignature = sig;
  $("#chat-messages").innerHTML = messages.length
    ? messages
        .map(
          (r) =>
            `<div class="chat-message ${r.data.role === "user" ? "user" : ""}"><span class="message-role">${r.data.role === "user" ? "YOU" : esc(r.data.model || "COLLABORATOR")}</span>${esc(r.data.text)}${r.data.citations?.length ? `<div class="citations">${r.data.citations.map((c, i) => urlLink(c.url, c.title || "Source " + (i + 1))).join("")}</div>` : ""}</div>`,
        )
        .join("")
    : `<div class="chat-welcome">Describe what you need to understand or decide.<br><br>Use chat to clarify the question or find evidence. Enable “Steer agent” to add direction to the investigation.</div><button class="suggestion" data-action="suggest" data-id="Help me turn this problem into a measurable decision.">Make the problem measurable ↗</button><button class="suggestion" data-action="suggest" data-id="Which assumptions could change the best solution to this problem?">Challenge the assumptions ↗</button><button class="suggestion" data-action="suggest" data-id="What evidence would be most useful next, and why?">Find the next useful evidence ↗</button>`;
  if (chatBusy)
    $("#chat-messages").innerHTML +=
      '<div class="chat-message muted pulse">Working with the available context…</div>';
  $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
}
function openNew(dataset) {
  $("#problem-dataset").innerHTML =
    '<option value="general">Start with my problem · no dataset required</option>' +
    state.datasets
      .map(
        (d) =>
          `<option value="${d.id}">${esc(d.name)} · ${esc(d.domain)}</option>`,
      )
      .join("");
  if (dataset) {
    $("#problem-dataset").value = dataset;
    $("#problem-question").value =
      state.datasets.find((d) => d.id === dataset)?.use_case || "";
  }
  $("#problem-dialog").showModal();
  $("#problem-question").focus();
}
async function inspect(id) {
  const r = await api("/artifacts/" + id);
  $("#artifact-title").textContent = pretty(r.kind) + " · " + id;
  $("#artifact-content").textContent = JSON.stringify(r.data, null, 2);
  $("#artifact-export").href = "/api/export/" + id;
  $("#artifact-dialog").showModal();
}
async function act(action, id) {
  if (action === "nav") {
    page = id;
    renderMain();
    $("#main").focus({ preventScroll: true });
    $("#main").scrollTop = 0;
    window.scrollTo({ top: 0, behavior: "instant" });
    if (id === "methodlab") await methodLabRefresh();
    return;
  }
  if (action === "method-lab-refresh") {
    await methodLabRefresh(id || null);
    return;
  }
  if (action === "select") {
    selected = id;
    localStorage.setItem("symplex.problem", id);
    page = "overview";
    renderMain();
    renderChat();
    $("#main").focus({ preventScroll: true });
    $("#main").scrollTop = 0;
    window.scrollTo({ top: 0, behavior: "instant" });
    return;
  }
  if (action === "new") {
    openNew();
    return;
  }
  if (action === "dataset-problem") {
    openNew(id);
    return;
  }
  if (action === "suggest" || action === "steer-suggest") {
    document.body.classList.remove("chat-hidden");
    if (innerWidth <= 1050) document.body.classList.add("chat-open");
    if (action === "steer-suggest") $("#steering-mode").checked = true;
    $("#chat-input").value = id;
    $("#chat-input").focus();
    return;
  }
  if (action === "artifact") {
    await inspect(id);
    return;
  }
  if (action === "domain") {
    const d = state.domains.find((d) => d.id === id);
    $("#artifact-title").textContent = d.name;
    $("#artifact-content").textContent = d.cases
      .map((c) => `${c.id} · ${c.title}\n\n${c.full_specification}`)
      .join("\n\n────────────────\n\n");
    $("#artifact-export").removeAttribute("href");
    $("#artifact-dialog").showModal();
    return;
  }
  if (action === "import") {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json,application/json";
    input.onchange = async () => {
      try {
        const file = input.files[0];
        if (!file) return;
        if (file.size > 4 * 1024 * 1024)
          throw Error("Prepared slice must be under 4 MiB");
        const r = await api("/datasets/import", JSON.parse(await file.text()));
        await refresh();
        toast("P1 slice admitted. Run it from Evidence.");
        const label = btn("Run imported P1 slice", "p1-run", r.id, "primary");
        $("#main").insertAdjacentHTML(
          "afterbegin",
          `<div class="progress-notice"><span>Prepared slice admitted</span>${label}</div>`,
        );
      } catch (e) {
        toast(e.message);
      }
    };
    input.click();
    return;
  }
  if (busy) return;
  busy = true;
  try {
    let r;
    if (action === "use-case") {
      const brief = (state.use_cases || []).find((item) => item.id === id);
      if (!brief || typeof brief.problem !== "string" || brief.problem.length < 5 || brief.problem.length > 4000)
        throw Error("This starter brief does not contain a supported problem statement.");
      const content = JSON.stringify({
        kind: "starter_brief",
        evidence_status: "Planning context, not observed data or validated findings.",
        brief,
      }, null, 2);
      if (content.length > 1000000) throw Error("This starter brief exceeds the context size limit.");
      r = await api("/problems", { question: brief.problem, dataset: "general" });
      selected = r.id;
      localStorage.setItem("symplex.problem", selected);
      page = "problem";
      try {
        await api("/context", {
          problem_id: selected,
          title: ("Starter brief · " + brief.title).slice(0, 200),
          content,
          format: "json",
          basis: "user_context",
        });
      } catch (error) {
        await refresh();
        throw Error("Problem created, but the starter context was not attached: " + error.message);
      }
      toast("Problem and starter context saved. Add your evidence before launching the agent.");
    }
    if (action === "plan") r = await api("/plan", { id });
    if (action === "design") r = await api("/solutions", { id });
    if (action === "complex-system") {
      r = await api("/complex-system", { id });
      page = "complexity";
      toast("Astra is constructing the semantic system and validation requirements.");
    }
    if (action === "solve") {
      r = await api("/solve", {
        id,
        depth: $("#solver-depth")?.value || "balanced",
      });
      page = "runs";
      toast("The agent is choosing and executing the investigation.");
    }
    if (action === "scenarios") {
      r = await api("/scenarios", { id });
      page = "scenarios";
    }
    if (action === "compute") {
      r = await api("/compute", { id });
      page = "lab";
    }
    if (action === "code") {
      r = await api("/code", { id });
      page = "models";
    }
    if (action === "validate") {
      r = await api("/validate", { id });
      toast("Separate model critique saved.");
    }
    if (action === "acquire") {
      r = await api("/datasets/bamtwoogle", {});
      toast("BamTwoogle downloaded and Croissant metadata validated.");
    }
    if (action === "fixture" || action === "fixture-live") {
      r = await api("/run", {
        kind: "fixture",
        live: action === "fixture-live",
      });
      page = "runs";
    }
    if (action === "p1-run") {
      r = await api("/run", { kind: "p1", dataset_id: id, live: true });
      page = "runs";
    }
    if (action === "research-run") {
      r = await api("/run", { kind: "research", live: true });
      page = "runs";
    }
    if (action === "cancel") {
      await api("/cancel", { id });
      toast("Cancellation requested. The current API request may finish.");
    }
    if (action === "capabilities") {
      r = await api("/capabilities", {});
      toast("Model access checks saved.");
    }
    await refresh();
    if (action === "complex-system" || action === "use-case") $("#main").focus();
    if (action === "plan" || action === "design") {
      page = action === "plan" ? "problem" : "system";
      renderMain();
    }
  } finally {
    busy = false;
  }
}
