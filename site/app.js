/* Observed Database of Supercells dashboard — reads data.json (built nightly by scripts/build_data.py). */
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
  const kmBetween = (la1, lo1, la2, lo2) => {
    const r = Math.PI / 180, dLa = (la2 - la1) * r, dLo = (lo2 - lo1) * r;
    const h = Math.sin(dLa / 2) ** 2 + Math.cos(la1 * r) * Math.cos(la2 * r) * Math.sin(dLo / 2) ** 2;
    return 12742 * Math.asin(Math.sqrt(h));
  };
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  const nf = (v, d = 0) => v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
  const DOT = " · ";
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
    m.km = 0;
    for (let k = 1; k < m.pts.length; k++) m.km += kmBetween(m.pts[k - 1][1], m.pts[k - 1][2], m.pts[k][1], m.pts[k][2]);
  });
  S.forEach((s, i) => {
    s.i = i;
    s.tornadic = s.mesos.some((mi) => M[mi].tornadic);
    s.tors = [...new Set(s.mesos.flatMap((mi) => M[mi].tor))];
    const starts = s.tors.map((ti) => T[ti]).filter((t) => t.ncei).map((t) => t.t0);
    s.firstTor = starts.length ? Math.min(...starts) : null;
    s.tt = s.firstTor != null ? (s.firstTor - s.t0) / 60 : null;
    s.date = C[s.case].date;
    const ends = s.mesos.map((mi) => M[mi].end).filter(Boolean);
    s.end = ends.length ? ends.sort((x, y) => ends.filter((e) => e === y).length - ends.filter((e) => e === x).length)[0] : "unknown";
  });
  T.forEach((t, i) => {
    t.i = i; t.sc = M[t.mesos[0]].sc;
    t.drawn = t.segs.filter((g) => g.path);
    t.mlabels = t.mesos.map((mi) => M[mi].label).join(", ");
  });
  const uniq = (ids) => [...new Set(ids)];

  const tMin = Math.min(...S.map((s) => s.t0)), tMax = Math.max(...S.map((s) => s.t1));
  const H = 3600, lo = Math.floor(tMin / H) * H, hi = Math.ceil(tMax / H) * H;
  let win = [lo, hi];

  // header + notices
  const gen = new Date(D.generated);
  $("updated").textContent = `${C.length} day${C.length === 1 ? "" : "s"} · updated ${gen.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}`;
  $("foot").textContent = `Built from ${C.length} MesoTrack file${C.length === 1 ? "" : "s"}` + (D.problems.length ? ` · ${D.problems.length} parse warning(s), see console` : "");
  if (D.problems.length) console.warn("Data build warnings:\n" + D.problems.join("\n"));
  if (D.ncei_missing) {
    $("ncei-warn").textContent = `${D.ncei_missing} linked tornado${D.ncei_missing === 1 ? " wasn't" : "es weren't"} found in the tornado database or in NCEI Storm Events (NCEI typically publishes ~2–3 months after an event). They still count toward tornadic totals but have no path on the map or start time for the timing chart until the record exists.`;
    $("ncei-warn").classList.add("show");
  }

  // ---------------------------------------------------------------- map
  const map = L.map("map", { preferCanvas: true, zoomSnap: 0.25, worldCopyJump: true }).setView([36, -92], 5);
  const renderer = L.canvas({ tolerance: 6 });
  map.createPane("refPane").style.zIndex = 350; // labels/boundaries: above basemap, below tracks
  map.getPane("refPane").style.pointerEvents = "none";
  // Basemap: OpenFreeMap vector tiles (no API key) rendered by MapLibre, so labels and
  // boundaries stay sharp on high-DPI screens. Falls back to Esri raster tiles if WebGL
  // or OpenFreeMap is unavailable.
  const isDark = () => css("--surface").toLowerCase() === "#1a1a19";
  const OFM = "https://tiles.openfreemap.org/styles/";
  const OFM_ATTR = '<a href="https://openfreemap.org" target="_blank" rel="noopener">OpenFreeMap</a> &copy; <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors';
  const styleCache = {};
  async function vectorStyle(dark) {
    const name = dark ? "dark" : "positron";
    if (styleCache[name]) return styleCache[name];
    const st = await fetch(OFM + name).then((r) => { if (!r.ok) throw new Error("style " + r.status); return r.json(); });
    // Replace sub-national boundary layers with explicit state + county lines.
    const isSub = (l) => l["source-layer"] === "boundary" && /state|boundary_3/.test(l.id);
    const at = Math.max(0, st.layers.findIndex(isSub));
    st.layers = st.layers.filter((l) => !isSub(l));
    const land = ["!=", ["get", "maritime"], 1];
    const mk = (id, lvl, color, w, minzoom) => ({
      id, type: "line", source: "openmaptiles", "source-layer": "boundary", minzoom,
      filter: ["all", ["==", ["get", "admin_level"], lvl], land],
      layout: { "line-join": "round", "line-cap": "round" },
      paint: { "line-color": color, "line-width": ["interpolate", ["linear"], ["zoom"], ...w] },
    });
    st.layers.splice(at, 0,
      mk("mt-county", 6, dark ? "#3b3b38" : "#d2d0c8", [5, 0.3, 8, 0.7, 12, 1.2], 5),
      mk("mt-state", 4, dark ? "#64635e" : "#a8a69e", [3, 0.6, 6, 1.1, 10, 1.8], 2));
    return (styleCache[name] = st);
  }
  const webgl = (() => { try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch (_) { return false; } })();
  let gl = null, vectorOK = !!(window.maplibregl && L.maplibreGL && webgl), tiles = [];
  async function setTiles() {
    const dark = isDark();
    if (vectorOK) {
      try {
        const style = await vectorStyle(dark);
        if (!gl) {
          gl = L.maplibreGL({ style, interactive: false }).addTo(map);
          map.attributionControl.addAttribution(OFM_ATTR);
        } else {
          gl.getMaplibreMap().setStyle(style);
        }
        return;
      } catch (e) {
        console.warn("Vector basemap unavailable; using raster tiles.", e);
        vectorOK = false;
        if (gl) { map.removeLayer(gl); map.attributionControl.removeAttribution(OFM_ATTR); gl = null; }
      }
    }
    setRasterTiles();
  }
  const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/";
  function setRasterTiles() {
    tiles.forEach((t) => map.removeLayer(t));
    const v = isDark() ? "Dark" : "Light";
    const opts = { maxZoom: 16, maxNativeZoom: 16, attribution: "Basemap &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors" };
    const base = L.tileLayer(`${ESRI}World_${v}_Gray_Base/MapServer/tile/{z}/{y}/{x}`, opts);
    const ref = L.tileLayer(`${ESRI}World_${v}_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, { ...opts, attribution: "", pane: "refPane", opacity: 0.9 });
    let failed = 0;
    base.on("tileerror", () => {
      if (++failed === 6) {
        tiles.forEach((t) => map.removeLayer(t));
        tiles = [L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map)];
        tiles[0].bringToBack();
      }
    });
    tiles = [base.addTo(map), ref.addTo(map)];
    base.bringToBack();
  }
  L.control.scale({ imperial: true, metric: true }).addTo(map);

  const layer = L.layerGroup().addTo(map);
  let scLines = {}; // supercell idx -> [meso polylines]
  let scTors = {};  // supercell idx -> [tornado lines/halos/dots]

  function mesoPopup(m) {
    const s = S[m.sc];
    const tors = m.tor.map((ti) => T[ti]);
    const torRows = tors.length
      ? tors.map((t) => t.ncei
        ? `<tr><td><span class="ef-chip">${efLabel(t.ef)}</span></td><td>${fmtHM(t.t0)}–${fmtHM(t.t1)} · ${t.where}${t.len != null ? ` · ${t.len} mi` : ""}${t.segs.length > 1 ? ` · ${t.segs.length} segments` : ""}${t.dth ? ` · ${t.dth} death${t.dth > 1 ? "s" : ""}` : ""}</td></tr>`
        : `<tr><td><span class="ef-chip">?</span></td><td>Event ${t.id ?? "—"} not found yet</td></tr>`).join("")
      : `<tr><td>Tornadoes</td><td>None</td></tr>`;
    return `<div class="pop">
      <h3>Meso ${m.label} · supercell ${s.label}</h3>
      <div class="meta">${s.date} · ${C[s.case].file}</div>
      <table>
        <tr><td>Tracked</td><td>${fmtT(m.t0)} → ${fmtHM(m.t1)} (${dur(m.life)})</td></tr>
        <tr><td>Scans</td><td>${m.pts.length}</td></tr>
        <tr><td>Supercell end</td><td>${s.end}${s.merged_into ? ` (merged into ${s.merged_into})` : ""}</td></tr>
        <tr><td>Supercell</td><td>${s.mesos.length} meso${s.mesos.length > 1 ? "s" : ""}, ${dur((s.t1 - s.t0) / 60)}${s.tt != null ? `, 1st tornado +${dur(s.tt)}` : ""}</td></tr>
        ${torRows}
      </table></div>`;
  }

  function highlight(si, on) {
    (scLines[si] || []).forEach((l) => l.setStyle({ weight: on ? l.options._w + 2 : l.options._w, opacity: on ? 1 : l.options._o }));
    (scTors[si] || []).forEach((l) => {
      if (l.options._r != null) l.setRadius(on ? l.options._r + 1.5 : l.options._r);
      else l.setStyle({ weight: on ? l.options._w + 3 : l.options._w });
    });
  }

  function drawMap(sel) {
    layer.clearLayers(); scLines = {}; scTors = {};
    const [a, b] = win, clip = $("clipTracks").checked;
    const showNon = $("showNon").checked, showTor = $("showTorM").checked, showPaths = $("showPaths").checked;
    const surface = css("--surface"), nonC = css("--nontor"), torC = css("--tor"), pathC = css("--torpath");
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
    }

    if (showPaths) {
      for (const ti of sel.paths) {
        const t = T[ti];
        const tip = `<b>${efLabel(t.ef)} tornado</b>${t.segs.length > 1 ? ` · ${t.segs.length} segments` : ""}<br>${fmtT(t.t0)}–${fmtHM(t.t1)}<br>${t.where}${t.len != null ? ` · ${t.len} mi` : ""}${t.wid ? ` · max ${t.wid} yd` : ""}`;
        const segRows = t.segs.map((g) => `<tr><td><span class="ef-chip">${efLabel(g.ef)}</span></td><td>${fmtHM(g.t0)}–${fmtHM(g.t1)} · ${g.where}${g.len != null ? ` · ${g.len} mi` : ""} · <a href="${nceiLink(g.event)}" target="_blank" rel="noopener">${g.event}</a>${t.linked.includes(g.event) ? " ◂ linked" : ""}${g.path ? "" : " (no location)"}</td></tr>`).join("");
        const pop = `<div class="pop"><h3>${efLabel(t.ef)} tornado${t.segs.length > 1 ? ` · ${t.segs.length} segments` : ""}</h3><div class="meta">Meso ${t.mlabels} · supercell ${S[t.sc].label} · ${S[t.sc].date}</div><table>
            <tr><td>Time</td><td>${fmtT(t.t0)} → ${fmtHM(t.t1)} (${dur((t.t1 - t.t0) / 60)})</td></tr><tr><td>Location</td><td>${t.where}</td></tr>
            <tr><td>Path</td><td>${t.len ?? "—"} mi total · max width ${t.wid ?? "—"} yd</td></tr><tr><td>Casualties</td><td>${t.inj} injuries, ${t.dth} deaths</td></tr>
            ${t.partial ? `<tr><td>Note</td><td>From NCEI only; other segments may exist</td></tr>` : ""}
            </table><table style="margin-top:8px">${segRows}</table></div>`;
        // 1) all segment lines (halo, then line); 2) endpoint dots, smallest first so the
        //    wider segment's dot sits on top where two segments meet.
        const shapes = [], dots = [];
        const scList = uniq(t.mesos.map((mi) => M[mi].sc));
        const reg = (l) => { scList.forEach((si) => (scTors[si] ||= []).push(l)); return l; };
        for (const g of t.drawn) {
          const w = [3.5, 4.5, 5.5, 7, 8.5, 10][efNum(g.ef)] ?? 3.5;
          const same = g.path[0][0] === g.path[1][0] && g.path[0][1] === g.path[1][1];
          if (!same) {
            reg(L.polyline(g.path, { renderer, color: surface, weight: w + 3, opacity: 0.9, lineCap: "round", interactive: false, _w: w + 3 })).addTo(layer); // halo
            shapes.push(reg(L.polyline(g.path, { renderer, color: pathC, weight: w, opacity: 1, lineCap: "round", _w: w })));
            dots.push([g.path[0], w], [g.path[1], w]);
          } else {
            dots.push([g.path[0], w]);
          }
        }
        dots.sort((x, y) => x[1] - y[1]);
        for (const [ll, w] of dots) {
          shapes.push(reg(L.circleMarker(ll, { renderer, radius: w / 2 + 1.5, color: surface, weight: 1.75, fillColor: pathC, fillOpacity: 1, _r: w / 2 + 1.5 })));
        }
        shapes.forEach((sh) => sh.bindTooltip(tip, { className: "mt-tip", sticky: true, direction: "top", offset: [0, -8] }).bindPopup(pop, { maxWidth: 440 })
          .on("mouseover", () => scList.forEach((si) => highlight(si, true))).on("mouseout", () => scList.forEach((si) => highlight(si, false))).addTo(layer));
      }
    }
  }

  function fitToSelection(sel) {
    const ll = [];
    sel.mesos.forEach((mi) => M[mi].pts.forEach((p) => { if (!$("clipTracks").checked || (p[0] >= win[0] && p[0] <= win[1])) ll.push([p[1], p[2]]); }));
    sel.paths.forEach((ti) => T[ti].drawn.forEach((g) => ll.push(...g.path)));
    if (ll.length) map.fitBounds(L.latLngBounds(ll), { padding: [30, 30], maxZoom: 10 });
  }

  // ---------------------------------------------------------------- charts
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.animation.duration = 250;
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

  // Hover window for bar charts: x value, count, and % of everything in the chart.
  // opts.withinColumn adds % of that bar's column (all series at that x value).
  const pctStr = (v, tot) => (tot ? `${(Math.round((1000 * v) / tot) / 10).toFixed(1)}%` : "—");
  function barTip(titleFn, noun, plural, { withinColumn = false } = {}) {
    return {
      title: (items) => titleFn(items[0].label),
      label: (c) => {
        const sets = c.chart.data.datasets;
        const all = sets.reduce((n, ds) => n + ds.data.reduce((x, y) => x + y, 0), 0);
        const name = sets.length > 1 ? `${c.dataset.label}: ` : "";
        let out = ` ${name}${c.raw} ${c.raw === 1 ? noun : plural} · ${pctStr(c.raw, all)}${withinColumn ? " of all" : ""}`;
        if (withinColumn) {
          const col = sets.reduce((n, ds) => n + (ds.data[c.dataIndex] || 0), 0);
          out += ` · ${pctStr(c.raw, col)} of ${c.label}`;
        }
        return out;
      },
    };
  }

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

  // Floating HTML tooltip drawn above the whole page (not clipped by small canvases).
  const tipEl = document.createElement("div");
  tipEl.className = "chart-tip";
  document.body.appendChild(tipEl);
  function floatingTip({ chart, tooltip }) {
    if (!tooltip || tooltip.opacity === 0) { tipEl.style.opacity = 0; return; }
    const body = (tooltip.dataPoints || []).map((dp) => `<div><i style="background:${dp.dataset.backgroundColor[dp.dataIndex]}"></i>${chart.options.plugins.tooltip.callbacks.label(dp)}</div>`).join("");
    tipEl.innerHTML = `<b>${(tooltip.title || []).join(" ")}</b>${body}`;
    const r = chart.canvas.getBoundingClientRect();
    const half = tipEl.offsetWidth / 2, vw = document.documentElement.clientWidth;
    const x = Math.min(Math.max(r.left + tooltip.caretX, half + 8), vw - half - 8);
    tipEl.style.left = `${x + window.scrollX}px`;
    tipEl.style.top = `${r.top + window.scrollY + tooltip.caretY}px`;
    tipEl.style.opacity = 1;
  }

  // ---------------------------------------------------------------- tornado chronology
  // Intensity tab: one donut per EF rating, split by order within the supercell.
  // Order tab: one donut per order position, split by EF rating. Same counts, transposed.
  const ORD = ["Only", "1st", "2nd", "3rd", "4th", "5th+"];
  const ORD_VARS = ["--ord-only", "--ord-1", "--ord-2", "--ord-3", "--ord-4", "--ord-5"];
  const EF_BINS = ["U", "0", "1", "2", "3", "4", "5"];
  const EF_VARS = ["--ef-u", "--ef-0", "--ef-1", "--ef-2", "--ef-3", "--ef-4", "--ef-5"];
  const EF_SHOW = ["0", "1", "2", "3", "4", "5", "U"]; // grid order for the Intensity tab
  let chronoView = "intensity";
  const cell = (id, label, cls = "") => `
    <div class="order-cell ${cls}">
      <div class="order-donut"><canvas id="${id}"></canvas><div class="order-center">${label}</div></div>
      <div class="order-n" id="${id}-n"></div>
    </div>`;
  $("orderGrid").innerHTML = EF_SHOW.map((e) => cell(`ord-${e}`, efLabel(e), `ef${e}`)).join("");
  $("orderGrid2").innerHTML = ORD.map((o, k) => cell(`ordp-${k}`, o)).join("");
  let lastChronoScs = [];

  function smallDonut(id, title, labels, data, colors) {
    const n = data.reduce((a, b) => a + b, 0), surface = css("--surface");
    $(id + "-n").textContent = `n = ${n}`;
    upsert(id, "doughnut", {
      labels,
      datasets: [{ data: n ? data : [1], backgroundColor: n ? colors : [css("--grid")], borderColor: surface, borderWidth: n ? 1.5 : 0, hoverOffset: 2 }],
    }, {
      responsive: true, maintainAspectRatio: false, cutout: "62%", animation: { duration: 250 },
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: false, external: n ? floatingTip : () => {},
          filter: (c) => c.raw > 0,
          callbacks: { title: () => title, label: (c) => `${c.label}: ${c.raw} · ${pctStr(c.raw, n)}` },
        },
      },
    });
  }

  function renderOrderDonuts(scs) {
    lastChronoScs = scs;
    const counts = Object.fromEntries(EF_BINS.map((e) => [e, Array(ORD.length).fill(0)]));
    for (const s of scs) {
      // whole tornadoes with a known start time, in the order they began
      const seq = s.tors.map((ti) => T[ti]).filter((t) => t.ncei && t.t0 != null).sort((a, b) => a.t0 - b.t0);
      seq.forEach((t, k) => {
        const e = EF_BINS.includes(String(t.ef)) ? String(t.ef) : "U";
        counts[e][seq.length === 1 ? 0 : Math.min(k + 1, ORD.length - 1)]++;
      });
    }
    const ordColors = ORD_VARS.map(css), efColors = EF_VARS.map(css);
    const byIntensity = chronoView === "intensity";
    $("orderGrid").hidden = !byIntensity;
    $("orderGrid2").hidden = byIntensity;
    if (byIntensity) {
      for (const e of EF_BINS) smallDonut("ord-" + e, `${efLabel(e)} tornadoes`, ORD, counts[e], ordColors);
      $("chronoHint").textContent = "Where each tornado fell in its supercell's sequence of tornadoes, by rating";
      $("orderLegend").innerHTML = ORD.map((o, k) => `<span><i style="background:${ordColors[k]}"></i>${o === "Only" ? "Only tornado" : o}</span>`).join("");
    } else {
      ORD.forEach((o, k) => smallDonut("ordp-" + k, o === "Only" ? "Only tornado in supercell" : `${o} tornado in supercell`,
        EF_BINS.map(efLabel), EF_BINS.map((e) => counts[e][k]), efColors));
      $("chronoHint").textContent = "Ratings of the tornadoes at each position in their supercell's sequence";
      $("orderLegend").innerHTML = EF_BINS.map((e, k) => `<span><i style="background:${efColors[k]}"></i>${efLabel(e)}</span>`).join("");
    }
  }
  const setChronoView = (v) => {
    chronoView = v;
    $("tabIntensity").setAttribute("aria-selected", v === "intensity");
    $("tabOrder").setAttribute("aria-selected", v === "order");
    tipEl.style.opacity = 0;
    renderOrderDonuts(lastChronoScs);
  };
  $("tabIntensity").addEventListener("click", () => setChronoView("intensity"));
  $("tabOrder").addEventListener("click", () => setChronoView("order"));

  // ---------------------------------------------------------------- selection + render
  function select() {
    const [a, b] = win;
    const scs = S.filter((s) => s.t0 <= b && s.t1 >= a);
    const mesos = scs.flatMap((s) => s.mesos).filter((mi) => M[mi].t0 <= b && M[mi].t1 >= a);
    const paths = T.filter((t) => t.ncei && t.drawn.length && t.t0 <= b && t.t1 >= a).map((t) => t.i);
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
    const tors = uniq(scs.flatMap((s) => s.tors)).map((ti) => T[ti]);
    const tts = torSc.map((s) => s.tt).filter((v) => v != null);
    const lifes = allMesos.map((m) => m.life);
    const days = new Set(scs.map((s) => s.case)).size;
    const totKm = allMesos.reduce((n, m) => n + m.km, 0), totMin = lifes.reduce((n, v) => n + v, 0);

    // KPIs
    const sig = tors.filter((t) => t.ncei && efNum(t.ef) >= 2).length;
    const kpi = (v, k, s) => `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div><div class="s">${s}</div></div>`;
    $("kpis").innerHTML =
      kpi(scs.length, "Supercells", `${days} day${days === 1 ? "" : "s"} · ${torSc.length} tornadic`) +
      kpi(allMesos.length, "Mesocyclones", scs.length ? `${(allMesos.length / scs.length).toFixed(2)} per supercell` : "—") +
      kpi(tors.length, "Linked tornadoes", `${sig} rated EF2+`) +
      kpi(tts.length ? dur(median(tts)) : "—", "Median time to 1st tornado", tts.length ? `${tts.length} tornadic supercell${tts.length === 1 ? "" : "s"}` : "no timed tornadoes") +
      kpi(`${nf(totKm)} km`, "Total meso track length", allMesos.length ? `${nf(totKm * 0.621371)} mi${DOT}${nf(totKm / allMesos.length)} km per meso` : "—") +
      kpi(`${nf(totMin / 60, totMin < 600 ? 1 : 0)} h`, "Total meso duration", allMesos.length ? `${dur(totMin / allMesos.length)} per meso` : "—");

    // donuts
    donut("pieSc", torSc.length, scs.length - torSc.length, "Supercell count", ["Tornadic", "Nontornadic"]);
    const tm = allMesos.filter((m) => m.tornadic).length;
    donut("pieMeso", tm, allMesos.length - tm, "Mesocyclone count", ["Tornadic", "Nontornadic"]);
    const tmt = torScMesos.filter((m) => m.tornadic).length;
    donut("pieMesoTs", tmt, torScMesos.length - tmt, "Mesocyclones in tornadic supercells", ["Tornadic", "Nontornadic"]);

    // histogram: mesos per supercell (integer bins)
    const counts = scs.map((s) => s.mesos.length), maxN = Math.max(1, ...counts);
    const mpc = Array.from({ length: maxN }, (_, k) => counts.filter((c) => c === k + 1).length);
    const o1 = baseOpts("Mesocyclone count", "Supercells");
    o1.plugins.tooltip.callbacks = barTip((x) => `${x} meso${x === "1" ? "" : "s"} per supercell`, "supercell", "supercells");
    $("mpcHint").textContent = counts.length ? `Mean: ${nf(mean(counts), 2)}${DOT}Max: ${Math.max(...counts)}` : "No supercells in window";
    upsert("histMeso", "bar", { labels: mpc.map((_, k) => String(k + 1)), datasets: [barDs(mpc, css("--nontor"), "Supercells")] }, o1);

    // histograms: tornadoes per tornadic supercell / per tornadic mesocyclone (bins from 1)
    const countHist = (id, hintId, allVals, xTitle, yTitle, noun, plural, unit) => {
      const vals = allVals.filter((v) => v > 0);
      const maxN = Math.max(1, ...vals);
      const cnt = Array.from({ length: maxN }, (_, k) => vals.filter((v) => v === k + 1).length);
      const o = baseOpts(xTitle, yTitle);
      o.scales.x.ticks.autoSkip = false;
      o.plugins.tooltip.callbacks = barTip((x) => `${x} tornado${x === "1" ? "" : "es"} per ${unit}`, noun, plural);
      upsert(id, "bar", { labels: cnt.map((_, k) => String(k + 1)), datasets: [barDs(cnt, css("--tor"), plural)] }, o);
      $(hintId).textContent = vals.length ? `Mean: ${nf(mean(vals), 2)}${DOT}Max: ${Math.max(...vals)}` : `No tornadic ${plural} in window`;
    };
    countHist("histTPS", "tpsHint", scs.map((x) => x.tors.length), "Tornado count", "Tornadic supercells", "supercell", "supercells", "supercell");
    countHist("histTPM", "tpmHint", allMesos.map((m) => m.tor.length), "Tornado count", "Tornadic mesocyclones", "mesocyclone", "mesocyclones", "mesocyclone");

    // donuts: order of each tornado within its supercell, one donut per EF rating
    renderOrderDonuts(scs);

    // histogram: time to first tornado
    // 20-min bins; label the bin that starts each hour (0–20, 60–80, ...), tilted 30°
    const ttB = hist(tts, 20, { minEdge: tts.length && Math.min(...tts) >= 0 ? 0 : undefined });
    const ttHour = (i) => { const lab = ttB.labels[i]; return lab != null && Number(lab.split("–")[0]) % 60 === 0; };
    const o2 = baseOpts("Minutes", "Supercells");
    Object.assign(o2.scales.x.ticks, {
      autoSkip: false, minRotation: 30, maxRotation: 30,
      callback: (v, i) => (ttHour(i) ? ttB.labels[i] : ""),
    });
    // short, wide gray tick marks below the axis at the labeled (hourly) bars only
    o2.scales.x.grid = {
      display: true, drawOnChartArea: false, drawTicks: true, offset: false,
      tickLength: 7, tickWidth: 2, tickColor: (c) => (ttHour(c.index) ? css("--axis") : "transparent"),
    };
    o2.plugins.tooltip.callbacks = barTip((x) => `${x} min to first tornado`, "supercell", "supercells");
    upsert("histTT", "bar", { labels: ttB.labels, datasets: [barDs(ttB.counts, css("--tor"), "Tornadic supercells")] }, o2);
    $("ttHint").textContent = tts.length ? `Min: ${dur(Math.min(...tts))}${DOT}Mean: ${dur(mean(tts))}${DOT}Max: ${dur(Math.max(...tts))}` : "No tornadic supercells in window";

    // histogram: meso lifetime
    const lB = hist(lifes, niceBin(lifes), { minEdge: 0 });
    const o3 = baseOpts("Minutes", "Mesocyclones");
    o3.plugins.tooltip.callbacks = barTip((x) => `${x} min lifetime`, "mesocyclone", "mesocyclones");
    $("lifeHint").textContent = lifes.length ? `Min: ${dur(Math.min(...lifes))}${DOT}Mean: ${dur(mean(lifes))}${DOT}Max: ${dur(Math.max(...lifes))}` : "No mesocyclones in window";
    upsert("histLife", "bar", { labels: lB.labels, datasets: [barDs(lB.counts, css("--nontor"), "Mesocyclones")] }, o3);

    // supercell dissipation mode, tornadic vs not (2 series), fixed order
    const ORDER = ["dissipate", "merge-sup", "merge-qlcs", "upscale", "unknown"];
    const modes = [...ORDER, ...uniq(scs.map((x) => x.end)).filter((md) => !ORDER.includes(md)).sort()];
    const o4 = baseOpts("", "Supercells");
    o4.scales.x.ticks.autoSkip = false;
    o4.plugins.tooltip.callbacks = barTip((x) => `Dissipation mode: ${x}`, "supercell", "supercells", { withinColumn: true });
    o4.plugins.legend.display = true; o4.plugins.legend.position = "top"; o4.plugins.legend.align = "end";
    upsert("barEnd", "bar", {
      labels: modes,
      datasets: [
        barDs(modes.map((md) => scs.filter((x) => x.end === md && x.tornadic).length), css("--tor"), "Tornadic"),
        barDs(modes.map((md) => scs.filter((x) => x.end === md && !x.tornadic).length), css("--nontor"), "Nontornadic"),
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
    o5.plugins.tooltip.callbacks = barTip((x) => (x === "Pending" ? "Not yet in a tornado record" : `${x} (max segment rating)`), "tornado", "tornadoes");
    upsert("barEF", "bar", { labels: efLabels, datasets: [barDs(efCounts, efLabels.map((l) => (l === "Pending" ? css("--muted") : css("--tor"))), "Tornadoes")] }, o5);

    // window note
    $("windowNote").textContent = `${fmtT(win[0])} → ${fmtT(win[1])} · ${scs.length} supercell${scs.length === 1 ? "" : "s"} active in window. Charts summarize those whole storms; the map shows track segments ${$("clipTracks").checked ? "inside the window only" : "for the full life of each meso"}.`;
    location.replace(`#t=${win[0]},${win[1]}` + (range[0] !== lo || range[1] !== hi ? `&r=${range[0]},${range[1]}` : ""));
    renderDays();
  }

  // ---------------------------------------------------------------- days table
  const dayStats = C.map((c, ci) => {
    const scs = c.sc.map((i) => S[i]);
    const tors = uniq(scs.flatMap((s) => s.tors)).map((ti) => T[ti]);
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
    zoomToEvent(r.t0, r.t1);
    $("daySelect").value = r.ci;
  });

  // ---------------------------------------------------------------- time controls
  const slider = $("slider");
  noUiSlider.create(slider, { start: win, connect: true, step: 60, range: { min: lo, max: hi }, behaviour: "tap-drag" });
  // The slider's own span ("range") can zoom to one event for finer scrubbing;
  // "All days" (either button) restores the full span.
  let range = [lo, hi];
  function setRange(a, b) {
    range = [Math.max(lo, a), Math.min(hi, b)];
    slider.noUiSlider.updateOptions({ range: { min: range[0], max: range[1] } }, false);
    $("scaleMin").textContent = fmtT(range[0]); $("scaleMax").textContent = fmtT(range[1]);
  }
  const Q = 900; // 15-min padding/rounding for an event's span
  function zoomToEvent(t0, t1) {
    const a = Math.floor((t0 - Q) / Q) * Q, b = Math.ceil((t1 + Q) / Q) * Q;
    setRange(a, b);
    setWindow(a, b, { fit: true });
  }
  $("scaleMin").textContent = fmtT(lo); $("scaleMax").textContent = fmtT(hi);
  let raf = 0, fitNext = false;
  function setWindow(a, b, { fit = false, fromSlider = false } = {}) {
    a = Math.max(lo, Math.min(a, hi)); b = Math.max(lo, Math.min(b, hi));
    if (b < a) [a, b] = [b, a];
    if (a < range[0] || b > range[1]) setRange(lo, hi); // typed/hash window outside the zoomed span
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
  $("allBtn2").addEventListener("click", () => $("allBtn").click());
  $("allBtn").addEventListener("click", () => { $("daySelect").value = ""; $("yearSelect").value = ""; setRange(lo, hi); setWindow(lo, hi, { fit: true }); });

  dayStats.slice().sort((x, y) => (x.date < y.date ? 1 : -1)).forEach((d) => {
    $("daySelect").insertAdjacentHTML("beforeend", `<option value="${d.ci}">${d.date} — ${d.sc} supercells, ${d.tor} tornadoes</option>`);
  });
  $("daySelect").addEventListener("change", (e) => {
    if (e.target.value === "") return;
    const d = dayStats.find((x) => x.ci === +e.target.value);
    $("yearSelect").value = "";
    zoomToEvent(d.t0, d.t1);
  });
  [...new Set(C.map((c) => c.date.slice(0, 4)))].sort().reverse().forEach((y) => $("yearSelect").insertAdjacentHTML("beforeend", `<option>${y}</option>`));
  $("yearSelect").addEventListener("change", (e) => {
    if (!e.target.value) return;
    $("daySelect").value = "";
    setRange(lo, hi);
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
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (_) {}
  setTiles();

  // initial window: from URL hash, else everything
  const m = location.hash.match(/t=(\d+),(\d+)/), mr = location.hash.match(/r=(\d+),(\d+)/);
  if (mr) setRange(+mr[1], +mr[2]);
  if (m) setWindow(+m[1], +m[2], { fit: true }); else setWindow(lo, hi, { fit: true });
})();
