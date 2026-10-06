/* Supercell Tracks dashboard — reads data.json (built nightly by scripts/build_data.py). */
(async function () {
  "use strict";

  // ---------------------------------------------------------------- helpers
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const pad = (n) => String(n).padStart(2, "0");
  const fmtT = (t) => { const d = new Date(t * 1000); return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}Z`; };
  const fmtHM = (t) => { const d = new Date(t * 1000); return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}Z`; };
  const fmtDay = (t) => new Date(t * 1000).toISOString().slice(0, 10);
  const toInput = (t) => new Date(t * 1000).toISOString().slice(0, 16);
  const fromInput = (v) => Date.parse(v + ":00Z") / 1000;
  const dur = (min) => { min = Math.round(min); const h = Math.floor(Math.abs(min) / 60), m = Math.abs(min) % 60; return (min < 0 ? "−" : "") + (h ? `${h} h ${m} m` : `${m} min`); };
  const median = (a) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y), k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };
  const pct = (a, b) => (b ? Math.round((100 * a) / b) : 0);
  const efNum = (ef) => (ef === "U" || ef == null ? -1 : +ef);
  const efLabel = (ef) => (ef === "U" || ef == null ? "EFU" : "EF" + ef);
  const nceiLink = (id) => `https://www.ncdc.noaa.gov/stormevents/eventdetails.jsp?id=${id}`;

  // ---------------------------------------------------------------- data
  let D;
  try {
    D = await fetch("data.json", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
  } catch (e) {
    $("updated").textContent = "Couldn't load data.json — has the nightly build run yet?";
    return;
  }
  const S = D.supercells, M = D.mesos, T = D.tornadoes, C = D.cases;

  M.forEach((m, i) => {
    m.i = i; m.t0 = m.pts[0][0]; m.t1 = m.pts[m.pts.length - 1][0];
    m.life = (m.t1 - m.t0) / 60; m.tornadic = m.tor.length > 0;
  });
  S.forEach((s, i) => {
    s.i = i;
    s.tornadic = s.mesos.some((mi) => M[mi].tornadic);
    s.tors = s.mesos.flatMap((mi) => M[mi].tor);
    const starts = s.tors.map((ti) => T[ti]).filter((t) => t.ncei).map((t) => t.t0);
    s.firstTor = starts.length ? Math.min(...starts) : null;
    s.tt = s.firstTor != null ? (s.firstTor - s.t0) / 60 : null;
    s.date = C[s.case].date;
  });
  T.forEach((t, i) => { t.i = i; t.sc = M[t.meso].sc; });

  const tMin = Math.min(...S.map((s) => s.t0)), tMax = Math.max(...S.map((s) => s.t1));
  const H = 3600, lo = Math.floor(tMin / H) * H, hi = Math.ceil(tMax / H) * H;
  let win = [lo, hi];

  // header + notices
  const gen = new Date(D.generated);
  $("updated").textContent = `${C.length} day${C.length === 1 ? "" : "s"} · updated ${gen.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}`;
  $("foot").textContent = `Built from ${C.length} MesoTrack file${C.length === 1 ? "" : "s"}` + (D.problems.length ? ` · ${D.problems.length} parse warning(s), see console` : "");
  if (D.problems.length) console.warn("Data build warnings:\n" + D.problems.join("\n"));
  if (D.ncei_missing) {
    $("ncei-warn").textContent = `${D.ncei_missing} linked tornado${D.ncei_missing === 1 ? " isn't" : "es aren't"} in the NCEI Storm Events database yet (NCEI typically publishes ~2–3 months after an event). They still count toward tornadic totals but have no path on the map or start time for the timing chart until NCEI posts them.`;
    $("ncei-warn").classList.add("show");
  }

  // ---------------------------------------------------------------- map
  const map = L.map("map", { preferCanvas: true, zoomSnap: 0.25, worldCopyJump: true }).setView([36, -92], 5);
  const renderer = L.canvas({ tolerance: 6 });
  let tiles;
  const isDark = () => css("--surface").toLowerCase() === "#1a1a19";
  function setTiles() {
    if (tiles) map.removeLayer(tiles);
    const style = isDark() ? "dark_all" : "light_all";
    tiles = L.tileLayer(`https://{s}.basemaps.cartocdn.com/${style}/{z}/{x}/{y}{r}.png`, {
      subdomains: "abcd", maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }).addTo(map);
  }
  setTiles();
  L.control.scale({ imperial: true, metric: true }).addTo(map);

  const layer = L.layerGroup().addTo(map);
  let scLines = {}; // supercell idx -> [polyline]

  function mesoPopup(m) {
    const s = S[m.sc];
    const tors = m.tor.map((ti) => T[ti]);
    const torRows = tors.length
      ? tors.map((t) => t.ncei
        ? `<tr><td><span class="ef-chip">${efLabel(t.ef)}</span></td><td>${fmtHM(t.t0)}–${fmtHM(t.t1)} · ${t.where}${t.len != null ? ` · ${t.len} mi` : ""}${t.dth ? ` · ${t.dth} death${t.dth > 1 ? "s" : ""}` : ""} · <a href="${nceiLink(t.event)}" target="_blank" rel="noopener">NCEI ${t.event}</a></td></tr>`
        : `<tr><td><span class="ef-chip">?</span></td><td>Event ${t.event ?? "—"} not in NCEI yet</td></tr>`).join("")
      : `<tr><td>Tornadoes</td><td>None</td></tr>`;
    return `<div class="pop">
      <h3>Meso ${m.label} · supercell ${s.label}</h3>
      <div class="meta">${s.date} · ${C[s.case].file}</div>
      <table>
        <tr><td>Tracked</td><td>${fmtT(m.t0)} → ${fmtHM(m.t1)} (${dur(m.life)})</td></tr>
        <tr><td>Scans</td><td>${m.pts.length}</td></tr>
        <tr><td>Ended</td><td>${m.end || "—"}${s.merged_into ? ` (merged into ${s.merged_into})` : ""}</td></tr>
        <tr><td>Supercell</td><td>${s.mesos.length} meso${s.mesos.length > 1 ? "s" : ""}, ${dur((s.t1 - s.t0) / 60)}${s.tt != null ? `, 1st tornado +${dur(s.tt)}` : ""}</td></tr>
        ${torRows}
      </table></div>`;
  }

  function highlight(si, on) {
    (scLines[si] || []).forEach((l) => l.setStyle({ weight: on ? l.options._w + 2 : l.options._w, opacity: on ? 1 : l.options._o }));
  }

  function drawMap(sel) {
    layer.clearLayers(); scLines = {};
    const [a, b] = win, clip = $("clipTracks").checked;
    const showNon = $("showNon").checked, showTor = $("showTorM").checked, showPaths = $("showPaths").checked;
    const surface = css("--surface"), nonC = css("--nontor"), torC = css("--tor"), pathC = css("--torpath");
    const starts = [];
    const order = sel.mesos.slice().sort((x, y) => M[x].tornadic - M[y].tornadic); // tornadic on top
    for (const mi of order) {
      const m = M[mi];
      if (m.tornadic ? !showTor : !showNon) continue;
      let pts = m.pts;
      if (clip) pts = pts.filter((p) => p[0] >= a && p[0] <= b);
      if (!pts.length) continue;
      const latlngs = pts.map((p) => [p[1], p[2]]);
      const w = m.tornadic ? 3 : 2.25, o = m.tornadic ? 0.95 : 0.8;
      const line = (latlngs.length > 1 ? L.polyline(latlngs, { renderer, color: m.tornadic ? torC : nonC, weight: w, opacity: o, lineCap: "round", lineJoin: "round", _w: w, _o: o })
        : L.circleMarker(latlngs[0], { renderer, radius: 3, color: m.tornadic ? torC : nonC, weight: 2, _w: 2, _o: 1 }));
      line.bindTooltip(`<b>${S[m.sc].date} · ${m.label}</b><br>${fmtHM(m.t0)}–${fmtHM(m.t1)} · ${dur(m.life)}${m.tornadic ? ` · ${m.tor.length} tornado${m.tor.length > 1 ? "es" : ""}` : ""}`, { className: "mt-tip", sticky: true, direction: "top", offset: [0, -8] });
      line.bindPopup(() => mesoPopup(m), { maxWidth: 420 });
      line.on("mouseover", () => highlight(m.sc, true)).on("mouseout", () => highlight(m.sc, false));
      line.addTo(layer);
      (scLines[m.sc] ||= []).push(line);
      if (pts[0][0] === m.t0) starts.push([latlngs[0], m.tornadic ? torC : nonC]);
    }
    for (const [ll, c] of starts) L.circleMarker(ll, { renderer, radius: 3.5, color: c, weight: 2, fillColor: surface, fillOpacity: 1, interactive: false }).addTo(layer);

    if (showPaths) {
      for (const ti of sel.paths) {
        const t = T[ti];
        const w = [3.5, 4.5, 5.5, 7, 8.5, 10][efNum(t.ef)] ?? 3.5;
        const same = t.path[0][0] === t.path[1][0] && t.path[0][1] === t.path[1][1];
        const tip = `<b>${efLabel(t.ef)} tornado</b><br>${fmtT(t.t0)}–${fmtHM(t.t1)}<br>${t.where}${t.len != null ? ` · ${t.len} mi` : ""}${t.wid ? ` · ${t.wid} yd` : ""}`;
        const pop = `<div class="pop"><h3>${efLabel(t.ef)} tornado</h3><div class="meta">Meso ${M[t.meso].label} · supercell ${S[t.sc].label} · ${S[t.sc].date}</div><table>
            <tr><td>Time</td><td>${fmtT(t.t0)} → ${fmtHM(t.t1)}</td></tr><tr><td>Location</td><td>${t.where}</td></tr>
            <tr><td>Path</td><td>${t.len ?? "—"} mi × ${t.wid ?? "—"} yd</td></tr><tr><td>Casualties</td><td>${t.inj} injuries, ${t.dth} deaths</td></tr>
            <tr><td>NCEI</td><td><a href="${nceiLink(t.event)}" target="_blank" rel="noopener">Event ${t.event}</a></td></tr></table></div>`;
        let shape;
        if (same) shape = L.circleMarker(t.path[0], { renderer, radius: w / 1.4 + 1, color: surface, weight: 2, fillColor: pathC, fillOpacity: 1 });
        else {
          L.polyline(t.path, { renderer, color: surface, weight: w + 3, opacity: 0.9, lineCap: "round", interactive: false }).addTo(layer); // halo
          shape = L.polyline(t.path, { renderer, color: pathC, weight: w, opacity: 1, lineCap: "round" });
        }
        shape.bindTooltip(tip, { className: "mt-tip", sticky: true, direction: "top", offset: [0, -8] }).bindPopup(pop, { maxWidth: 380 }).addTo(layer);
      }
    }
  }

  function fitToSelection(sel) {
    const ll = [];
    sel.mesos.forEach((mi) => M[mi].pts.forEach((p) => { if (!$("clipTracks").checked || (p[0] >= win[0] && p[0] <= win[1])) ll.push([p[1], p[2]]); }));
    sel.paths.forEach((ti) => ll.push(...T[ti].path));
    if (ll.length) map.fitBounds(L.latLngBounds(ll), { padding: [30, 30], maxZoom: 10 });
  }

  // ---------------------------------------------------------------- charts
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.animation = { duration: 250 };
  const charts = {};

  function baseOpts(xTitle, yTitle) {
    const grid = css("--grid"), muted = css("--muted"), ink2 = css("--ink-2"), surface = css("--surface"), ink = css("--ink");
    return {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false, labels: { color: ink2, boxWidth: 10, boxHeight: 10, useBorderRadius: true, borderRadius: 2 } },
        tooltip: { backgroundColor: surface, titleColor: ink, bodyColor: ink2, borderColor: css("--axis"), borderWidth: 1, padding: 10, cornerRadius: 8, displayColors: true, boxPadding: 4 },
      },
      scales: {
        x: { grid: { display: false }, border: { color: css("--axis") }, ticks: { color: muted, maxRotation: 0, autoSkipPadding: 8 }, title: { display: !!xTitle, text: xTitle, color: muted, font: { size: 11 } } },
        y: { beginAtZero: true, grid: { color: grid }, border: { display: false }, ticks: { color: muted, precision: 0 }, title: { display: !!yTitle, text: yTitle, color: muted, font: { size: 11 } } },
      },
    };
  }
  const barDs = (data, color, label) => ({ label, data, backgroundColor: color, hoverBackgroundColor: color, borderRadius: { topLeft: 4, topRight: 4 }, borderSkipped: "start", categoryPercentage: 0.92, barPercentage: 0.94 });

  function upsert(id, type, data, options) {
    if (charts[id]) { charts[id].data = data; charts[id].options = options; charts[id].update(); return; }
    charts[id] = new Chart($(id), { type, data, options });
  }

  function hist(values, width, { minEdge } = {}) {
    if (!values.length) return { labels: [], counts: [] };
    const lo = minEdge ?? Math.floor(Math.min(...values) / width) * width;
    const hiE = Math.max(lo + width, Math.ceil((Math.max(...values) + 1e-9) / width) * width);
    const n = Math.round((hiE - lo) / width), counts = Array(n).fill(0);
    values.forEach((v) => { counts[Math.min(n - 1, Math.floor((v - lo) / width))]++; });
    const labels = counts.map((_, k) => `${lo + k * width}–${lo + (k + 1) * width}`);
    return { labels, counts };
  }
  const niceBin = (vals) => {
    const r = Math.max(...vals, 0) - Math.min(...vals, 0);
    return [5, 10, 15, 20, 30, 60, 120, 180, 360, 720, 1440].find((w) => r / w <= 16) || 2880;
  };

  function donut(id, tor, non, title, labels) {
    const torC = css("--tor"), nonC = css("--nontor"), surface = css("--surface");
    const total = tor + non;
    upsert(id, "doughnut", {
      labels,
      datasets: [{ data: total ? [tor, non] : [0, 1], backgroundColor: total ? [torC, nonC] : [css("--grid"), css("--grid")], borderColor: surface, borderWidth: 2, hoverOffset: 3 }],
    }, {
      responsive: true, maintainAspectRatio: false, cutout: "66%",
      plugins: { legend: { display: false }, tooltip: { enabled: !!total, backgroundColor: surface, titleColor: css("--ink"), bodyColor: css("--ink-2"), borderColor: css("--axis"), borderWidth: 1, cornerRadius: 8,
        callbacks: { label: (c) => ` ${c.label}: ${c.raw} (${pct(c.raw, total)}%)` } } },
    });
    $(id + "C").innerHTML = total ? `<div><b>${pct(tor, total)}%</b><span>tornadic</span></div>` : `<div><span>none</span></div>`;
    $(id + "T").innerHTML = `<h3>${title}</h3>
      <div class="row"><span><i style="background:${torC}"></i>${labels[0]}</span><b>${tor}</b></div>
      <div class="row"><span><i style="background:${nonC}"></i>${labels[1]}</span><b>${non}</b></div>
      <div class="row" style="color:var(--muted)"><span>Total</span><span>${total}</span></div>`;
  }

  // ---------------------------------------------------------------- selection + render
  function select() {
    const [a, b] = win;
    const scs = S.filter((s) => s.t0 <= b && s.t1 >= a);
    const mesos = scs.flatMap((s) => s.mesos).filter((mi) => M[mi].t0 <= b && M[mi].t1 >= a);
    const paths = T.filter((t) => t.ncei && t.path && t.t0 <= b && t.t1 >= a && (!$("clipTracks").checked || true)).map((t) => t.i);
    return { scs, mesos, paths };
  }

  function render({ fit = false } = {}) {
    const sel = select();
    drawMap(sel);
    if (fit) fitToSelection(sel);

    const scs = sel.scs;
    const allMesos = scs.flatMap((s) => s.mesos).map((mi) => M[mi]);
    const torSc = scs.filter((s) => s.tornadic);
    const torScMesos = torSc.flatMap((s) => s.mesos).map((mi) => M[mi]);
    const tors = scs.flatMap((s) => s.tors).map((ti) => T[ti]);
    const tts = torSc.map((s) => s.tt).filter((v) => v != null);
    const lifes = allMesos.map((m) => m.life);
    const days = new Set(scs.map((s) => s.case)).size;

    // KPIs
    const sig = tors.filter((t) => t.ncei && efNum(t.ef) >= 2).length;
    const kpi = (v, k, s) => `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div><div class="s">${s}</div></div>`;
    $("kpis").innerHTML =
      kpi(scs.length, "Supercells", `${days} day${days === 1 ? "" : "s"} · ${torSc.length} tornadic`) +
      kpi(allMesos.length, "Mesocyclones", scs.length ? `${(allMesos.length / scs.length).toFixed(2)} per supercell` : "—") +
      kpi(tors.length, "Linked tornadoes", `${sig} rated EF2+`) +
      kpi(tts.length ? dur(median(tts)) : "—", "Median time to 1st tornado", tts.length ? `n = ${tts.length} tornadic supercells` : "no timed tornadoes");

    // donuts
    donut("pieSc", torSc.length, scs.length - torSc.length, "Supercells", ["Tornadic", "Non-tornadic"]);
    const tm = allMesos.filter((m) => m.tornadic).length;
    donut("pieMeso", tm, allMesos.length - tm, "Mesocyclones (all supercells)", ["Tornadic", "Non-tornadic"]);
    const tmt = torScMesos.filter((m) => m.tornadic).length;
    donut("pieMesoTs", tmt, torScMesos.length - tmt, "Mesocyclones in tornadic supercells", ["Tornadic", "Non-tornadic"]);

    // histogram: mesos per supercell (integer bins)
    const counts = scs.map((s) => s.mesos.length), maxN = Math.max(1, ...counts);
    const mpc = Array.from({ length: maxN }, (_, k) => counts.filter((c) => c === k + 1).length);
    const o1 = baseOpts("Mesocyclones in supercell", "Supercells");
    o1.plugins.tooltip.callbacks = { title: (c) => `${c[0].label} meso${c[0].label === "1" ? "" : "s"}`, label: (c) => ` ${c.raw} supercell${c.raw === 1 ? "" : "s"} (${pct(c.raw, scs.length)}%)` };
    upsert("histMeso", "bar", { labels: mpc.map((_, k) => String(k + 1)), datasets: [barDs(mpc, css("--nontor"), "Supercells")] }, o1);

    // histogram: time to first tornado
    const ttB = hist(tts, niceBin(tts), { minEdge: tts.length && Math.min(...tts) >= 0 ? 0 : undefined });
    const o2 = baseOpts("Minutes after track start", "Supercells");
    o2.plugins.tooltip.callbacks = { title: (c) => `${c[0].label} min`, label: (c) => ` ${c.raw} supercell${c.raw === 1 ? "" : "s"}` };
    upsert("histTT", "bar", { labels: ttB.labels, datasets: [barDs(ttB.counts, css("--tor"), "Tornadic supercells")] }, o2);
    const neg = tts.filter((v) => v < 0).length;
    $("ttHint").textContent = `Tornadic supercells · minutes after first meso detection` + (tts.length ? ` · median ${dur(median(tts))}` : "") + (neg ? ` · ${neg} tornado before tracking began` : "");

    // histogram: meso lifetime
    const lB = hist(lifes, niceBin(lifes), { minEdge: 0 });
    const o3 = baseOpts("Minutes", "Mesocyclones");
    o3.plugins.tooltip.callbacks = { title: (c) => `${c[0].label} min`, label: (c) => ` ${c.raw} meso${c.raw === 1 ? "" : "s"}` };
    upsert("histLife", "bar", { labels: lB.labels, datasets: [barDs(lB.counts, css("--nontor"), "Mesocyclones")] }, o3);

    // ending modes, tornadic vs not (2 series)
    const modes = [...new Set(allMesos.map((m) => m.end || "unknown"))].sort();
    const o4 = baseOpts("", "Mesocyclones");
    o4.plugins.legend.display = true; o4.plugins.legend.position = "top"; o4.plugins.legend.align = "end";
    upsert("barEnd", "bar", {
      labels: modes,
      datasets: [
        barDs(modes.map((md) => allMesos.filter((m) => (m.end || "unknown") === md && m.tornadic).length), css("--tor"), "Tornadic"),
        barDs(modes.map((md) => allMesos.filter((m) => (m.end || "unknown") === md && !m.tornadic).length), css("--nontor"), "Non-tornadic"),
      ],
    }, o4);

    // EF ratings
    const efCats = ["0", "1", "2", "3", "4", "5", "U"];
    const efCounts = efCats.map((e) => tors.filter((t) => t.ncei && String(t.ef) === e).length);
    const pending = tors.filter((t) => !t.ncei).length;
    const efLabels = efCats.map((e) => "EF" + e);
    if (pending) { efLabels.push("Pending"); efCounts.push(pending); }
    const o5 = baseOpts("", "Tornadoes");
    o5.scales.x.ticks.autoSkip = false;
    o5.plugins.tooltip.callbacks = { label: (c) => ` ${c.raw} tornado${c.raw === 1 ? "" : "es"}` };
    upsert("barEF", "bar", { labels: efLabels, datasets: [barDs(efCounts, efLabels.map((l) => (l === "Pending" ? css("--muted") : css("--tor"))), "Tornadoes")] }, o5);

    // window note
    $("windowNote").textContent = `${fmtT(win[0])} → ${fmtT(win[1])} · ${scs.length} supercell${scs.length === 1 ? "" : "s"} active in window. Charts summarize those whole storms; the map shows track segments ${$("clipTracks").checked ? "inside the window only" : "for the full life of each meso"}.`;
    location.replace(`#t=${win[0]},${win[1]}`);
    renderDays();
  }

  // ---------------------------------------------------------------- days table
  const dayStats = C.map((c, ci) => {
    const scs = c.sc.map((i) => S[i]);
    const tors = scs.flatMap((s) => s.tors).map((ti) => T[ti]);
    const efs = tors.filter((t) => t.ncei).map((t) => efNum(t.ef));
    return { ci, date: c.date, sc: scs.length, tsc: scs.filter((s) => s.tornadic).length, m: scs.reduce((n, s) => n + s.mesos.length, 0),
      tor: tors.length, maxef: efs.length ? Math.max(...efs) : -2, t0: Math.min(...scs.map((s) => s.t0)), t1: Math.max(...scs.map((s) => s.t1)) };
  });
  let sortKey = "date", sortDir = -1;
  function renderDays() {
    const rows = [...dayStats].sort((x, y) => (x[sortKey] > y[sortKey] ? 1 : x[sortKey] < y[sortKey] ? -1 : 0) * sortDir);
    $("dayRows").innerHTML = rows.map((r) => {
      const active = win[0] <= r.t0 && win[1] >= r.t1 && (win[1] - win[0]) < 2 * 86400;
      return `<tr data-ci="${r.ci}" class="${active ? "active" : ""}"><td>${r.date}</td><td>${r.sc}</td><td>${r.tsc}</td><td>${r.m}</td><td>${r.tor}</td><td>${r.maxef === -2 ? "—" : r.maxef === -1 ? "EFU" : "EF" + r.maxef}</td></tr>`;
    }).join("");
  }
  document.querySelector("table.days thead").addEventListener("click", (e) => {
    const k = e.target.closest("th")?.dataset.k; if (!k) return;
    sortDir = k === sortKey ? -sortDir : -1; sortKey = k; renderDays();
  });
  $("dayRows").addEventListener("click", (e) => {
    const tr = e.target.closest("tr"); if (!tr) return;
    const r = dayStats.find((d) => d.ci === +tr.dataset.ci);
    setWindow(r.t0 - 900, r.t1 + 900, { fit: true });
    $("daySelect").value = r.ci;
  });

  // ---------------------------------------------------------------- time controls
  const slider = $("slider");
  noUiSlider.create(slider, { start: win, connect: true, step: 60, range: { min: lo, max: hi }, behaviour: "tap-drag" });
  $("scaleMin").textContent = fmtT(lo); $("scaleMax").textContent = fmtT(hi);
  let raf = 0, fitNext = false;
  function setWindow(a, b, { fit = false, fromSlider = false } = {}) {
    a = Math.max(lo, Math.min(a, hi)); b = Math.max(lo, Math.min(b, hi));
    if (b < a) [a, b] = [b, a];
    win = [a, b];
    $("tStart").value = toInput(a); $("tEnd").value = toInput(b);
    if (!fromSlider) slider.noUiSlider.set([a, b], false);
    fitNext = fitNext || fit;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => { render({ fit: fitNext }); fitNext = false; });
  }
  slider.noUiSlider.on("slide", (v) => { setWindow(+v[0], +v[1], { fromSlider: true }); $("daySelect").value = ""; $("yearSelect").value = ""; });
  [$("tStart"), $("tEnd")].forEach((el) => el.min = toInput(lo));
  [$("tStart"), $("tEnd")].forEach((el) => el.max = toInput(hi));
  $("tStart").addEventListener("change", () => setWindow(fromInput($("tStart").value), win[1]));
  $("tEnd").addEventListener("change", () => setWindow(win[0], fromInput($("tEnd").value)));
  $("allBtn").addEventListener("click", () => { $("daySelect").value = ""; $("yearSelect").value = ""; setWindow(lo, hi, { fit: true }); });

  dayStats.slice().sort((x, y) => (x.date < y.date ? 1 : -1)).forEach((d) => {
    $("daySelect").insertAdjacentHTML("beforeend", `<option value="${d.ci}">${d.date} — ${d.sc} supercells, ${d.tor} tornadoes</option>`);
  });
  $("daySelect").addEventListener("change", (e) => {
    if (e.target.value === "") return;
    const d = dayStats.find((x) => x.ci === +e.target.value);
    $("yearSelect").value = "";
    setWindow(d.t0 - 900, d.t1 + 900, { fit: true });
  });
  [...new Set(C.map((c) => c.date.slice(0, 4)))].sort().reverse().forEach((y) => $("yearSelect").insertAdjacentHTML("beforeend", `<option>${y}</option>`));
  $("yearSelect").addEventListener("change", (e) => {
    if (!e.target.value) return;
    $("daySelect").value = "";
    setWindow(Date.UTC(+e.target.value, 0, 1) / 1000, Date.UTC(+e.target.value + 1, 0, 1) / 1000 - 60, { fit: true });
  });
  ["showNon", "showTorM", "showPaths", "clipTracks"].forEach((id) => $(id).addEventListener("change", () => render()));
  $("fitBtn").addEventListener("click", () => fitToSelection(select()));

  // theme
  $("themeBtn").addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (_) {}
    Object.values(charts).forEach((c) => c.destroy()); for (const k in charts) delete charts[k];
    setTiles(); render();
  });
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; setTiles(); } catch (_) {}

  // initial window: from URL hash, else everything
  const m = location.hash.match(/t=(\d+),(\d+)/);
  if (m) setWindow(+m[1], +m[2], { fit: true }); else setWindow(lo, hi, { fit: true });
})();
