document.addEventListener("click", (e) => {
  const button = e.target.closest("[data-action]");
  if (button) {
    const old = button.textContent;
    if (
      ![
        "nav",
        "select",
        "artifact",
        "new",
        "suggest",
        "domain",
        "dataset-problem",
        "import",
      ].includes(button.dataset.action)
    ) {
      button.disabled = true;
      button.textContent = "Working…";
    }
    act(button.dataset.action, button.dataset.id)
      .catch((e) => toast(e.message))
      .finally(() => {
        if (button.isConnected) {
          button.disabled = false;
          button.textContent = old;
        }
      });
  }
});
$("#new-problem").onclick = () => openNew();
$("#close-dialog").onclick = () => $("#problem-dialog").close();
$("#close-artifact").onclick = () => $("#artifact-dialog").close();
$("#system-shortcut").onclick = () => act("nav", "harness");
if (localStorage.getItem("symplex.chat-visible") !== "true") document.body.classList.add("chat-hidden");
$("#chat-toggle").onclick = () => {
  if (innerWidth <= 1050) document.body.classList.toggle("chat-open");
  else document.body.classList.toggle("chat-hidden");
  localStorage.setItem("symplex.chat-visible", String(innerWidth <= 1050 ? document.body.classList.contains("chat-open") : !document.body.classList.contains("chat-hidden")));
};
$("#problem-form").onsubmit = async (e) => {
  e.preventDefault();
  const submit = e.submitter;
  submit.disabled = true;
  try {
    const r = await api("/problems", {
      question: $("#problem-question").value,
      dataset: $("#problem-dataset").value,
    });
    selected = r.id;
    localStorage.setItem("symplex.problem", selected);
    page = "problem";
    $("#problem-dialog").close();
    $("#problem-question").value = "";
    await refresh();
  } catch (e) {
    toast(e.message);
  } finally {
    submit.disabled = false;
  }
};
$("#chat-form").onsubmit = async (e) => {
  e.preventDefault();
  if (chatBusy) return;
  const message = $("#chat-input").value.trim();
  if (!message) return;
  chatBusy = true;
  $("#send-chat").disabled = true;
  renderChat();
  try {
    if ($("#steering-mode").checked) {
      if (!selected) throw Error("Select a problem before steering its agent.");
      await api("/steer", {
        problem_id: selected,
        instruction: message,
        kind: "direction",
      });
    } else {
      await api("/chat", {
        message,
        research: $("#research-mode").checked,
        problem_id: selected,
      });
    }
    $("#chat-input").value = "";
    await refresh(false);
  } catch (e) {
    toast(e.message);
  } finally {
    chatBusy = false;
    $("#send-chat").disabled = false;
    await refresh(false);
    renderChat();
  }
};
document.addEventListener("submit", async (e) => {
  if (e.target.id !== "revision-form") return;
  e.preventDefault();
  try {
    const r = await api("/revise", {
      id: selected,
      assumption: $("#revision-input").value,
    });
    selected = r.id;
    localStorage.setItem("symplex.problem", selected);
    await refresh();
    toast("New version created. Prior dependent artifacts are stale.");
  } catch (e) {
    toast(e.message);
  }
});
refresh();
setInterval(() => {
  if (state && !busy && !chatBusy) {
    const running =
      state.active.some((a) => a.status === "running") ||
      state.jobs.some((a) => a.status === "running");
    refresh(running && page !== "scene" && page !== "methodlab");
  }
}, 4000);
document.addEventListener("input", (e) => {
  if (e.target.id === "whatif-value")
    $("#whatif-output").textContent = Number(e.target.value).toFixed(2);
});
document.addEventListener("submit", async (e) => {
  if (e.target.id === "context-form") {
    e.preventDefault();
    try {
      await api("/context", {
        problem_id: selected,
        title: $("#context-title").value,
        content: $("#context-text").value,
        format: "text",
        basis: $("#context-basis").value,
      });
      await refresh();
      toast("Context saved for the agent.");
    } catch (e) {
      toast(e.message);
    }
  }
  if (e.target.id === "whatif-form") {
    e.preventDefault();
    e.submitter.disabled = true;
    try {
      await api("/what-if", {
        model_id: e.target.dataset.model,
        node_id: $("#whatif-node").value,
        value: Number($("#whatif-value").value),
      });
      await refresh();
      toast("New scenario version computed.");
    } catch (e) {
      toast(e.message);
      e.submitter.disabled = false;
    }
  }
});
document.addEventListener("click", (e) => {
  if (e.target.id !== "context-file") return;
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".csv,.json,.geojson,.txt";
  input.onchange = async () => {
    try {
      const f = input.files[0];
      if (!f) return;
      if (f.size > 1000000)
        throw Error("Keep attached artifacts under 1 MB for this prototype.");
      const extension = f.name.split(".").at(-1).toLowerCase();
      await api("/context", {
        problem_id: selected,
        title: f.name,
        content: await f.text(),
        format: ["csv", "json", "geojson"].includes(extension)
          ? extension
          : "text",
        basis: "user_context",
      });
      await refresh();
      toast("Artifact attached with its schema and provenance status.");
    } catch (e) {
      toast(e.message);
    }
  };
  input.click();
});

document.addEventListener("change", e => {
  if (e.target.id !== "problem-picker") return;
  selected = e.target.value || null;
  localStorage.setItem("symplex.problem", selected || "");
  page = "problem";
  renderSidebar(); renderMain(); renderChat();
  $("#main").focus({ preventScroll: true });
  $("#main").scrollTop = 0;
  window.scrollTo({ top: 0, behavior: "instant" });
});
