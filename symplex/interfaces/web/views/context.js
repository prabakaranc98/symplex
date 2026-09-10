function contextPanel() {
  if (!current()) return "";
  const attached = related("context");
  return `<div class="section-heading"><h3>Your context & artifacts</h3><span>Observations, assumptions, documents and data</span></div><form id="context-form" class="card"><label for="context-title">Context title</label><input id="context-title" required maxlength="200" placeholder="Local constraints, candidate sites, or source notes"><label for="context-text">What should the agent know?</label><textarea id="context-text" rows="3" required maxlength="1000000" placeholder="Add constraints, observations, definitions or assumptions…"></textarea><div class="actions"><select id="context-basis" aria-label="Evidence basis"><option value="user_context">User-provided context</option><option value="assumption">Assumption to test</option><option value="observed_data">Observed data · not yet verified</option></select><button class="primary">Add context</button><button type="button" class="secondary" id="context-file">Attach CSV, JSON or GeoJSON</button></div></form>${attached.map((r) => `<div class="list-row"><div class="row-main"><strong>${esc(r.data.title)}</strong><small>${esc(r.data.format)} · ${esc(pretty(r.data.basis))}</small></div>${artifact(r)}</div>${r.data.format === "geojson" ? spatialPlot(r.data.inspection.parsed) : ""}`).join("")}`;
}
function spatialPlot(geo) {
  const points = geo.features.flatMap((f) =>
    f.geometry.type === "Point"
      ? [f.geometry.coordinates]
      : f.geometry.coordinates.flat(),
  );
  const xs = points.map((p) => p[0]),
    ys = points.map((p) => p[1]),
    xmin = Math.min(...xs),
    xmax = Math.max(...xs),
    ymin = Math.min(...ys),
    ymax = Math.max(...ys),
    project = (p) => [
      35 + ((p[0] - xmin) / (xmax - xmin || 1)) * 530,
      245 - ((p[1] - ymin) / (ymax - ymin || 1)) * 220,
    ];
  return `<div class="card"><span class="eyebrow">UPLOADED SITE GEOGRAPHY</span><svg viewBox="0 0 600 285" role="img" aria-label="User-provided site coordinates">${geo.features
    .map((f, i) => {
      if (f.geometry.type === "Point") {
        const [x, y] = project(f.geometry.coordinates);
        return `<circle cx="${x}" cy="${y}" r="5" fill="#32654e"/><text x="${Math.min(500, x + 10)}" y="${y - 8}" font-size="11" fill="#32654e">${esc(f.properties?.name || "Site " + (i + 1))}</text>`;
      }
      return `<path d="${f.geometry.coordinates.map((r) => "M" + r.map((p) => project(p).join(",")).join("L") + "Z").join(" ")}" fill="#b9cda8" fill-rule="evenodd" stroke="#32654e"/>`;
    })
    .join(
      "",
    )}<text x="35" y="275" font-size="9" fill="#809278">Longitude ${xmin.toFixed(4)}–${xmax.toFixed(4)} · Latitude ${ymin.toFixed(4)}–${ymax.toFixed(4)}</text></svg><p class="tiny">Uploaded coordinates in longitude and latitude. This view does not include terrain or distance analysis.</p></div>`;
}
