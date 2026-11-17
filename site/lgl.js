/* lgl.js — a small Leaflet-shaped adapter over MapLibre GL, so the map viewer (app.js) can draw on a
   globe without changing its drawing code. It implements only what app.js uses: L.map, L.layerGroup,
   L.polyline, L.polygon, L.circleMarker, L.marker + L.divIcon, L.imageOverlay, L.control, L.Control.extend,
   L.control.scale, L.DomUtil.create, L.DomEvent, L.latLngBounds, L.canvas. Vector shapes are batched into
   three GeoJSON sources (fills, lines, circles); draw order follows the order shapes are added, as on
   Leaflet's canvas; hover styling uses MapLibre feature-state. Used by globe.html only. */
(function () {
  "use strict";
  const ml = window.maplibregl;
  const TOL = 6; // px, hit tolerance (as L.canvas({ tolerance: 6 }))
  const SRC = { fill: "lgl-fill", line: "lgl-line", circle: "lgl-circle" };
  let seq = 1;

  // ---------------------------------------------------------------- small helpers
  const ll2 = (ll) => (Array.isArray(ll) ? [ll[1], ll[0]] : [ll.lng, ll.lat]); // [lat, lon] -> [lon, lat]
  class Evented {
    on(t, f) { ((this._h ||= {})[t] ||= []).push(f); return this; }
    fire(t, e) { (this._h?.[t] || []).forEach((f) => f(e)); }
    has(t) { return !!(this._h?.[t]?.length); }
  }
  class Bounds {
    constructor(pts) { this.s = 90; this.n = -90; this.w = 180; this.e = -180; (pts || []).forEach((p) => this.extend(p)); }
    extend(p) { const [lat, lon] = Array.isArray(p) ? p : [p.lat, p.lng]; this.s = Math.min(this.s, lat); this.n = Math.max(this.n, lat); this.w = Math.min(this.w, lon); this.e = Math.max(this.e, lon); return this; }
    toML() { return [[this.w, this.s], [this.e, this.n]]; }
    corners() { return [[this.w, this.n], [this.e, this.n], [this.e, this.s], [this.w, this.s]]; } // tl tr br bl
    isValid() { return this.s <= this.n && this.w <= this.e; }
  }

  // ---------------------------------------------------------------- shapes
  class Shape extends Evented {
    constructor(kind, latlngs, opts) {
      super();
      this.kind = kind; this.latlngs = latlngs; this.options = { ...opts }; this.id = seq++; this._state = {};
      if (this.options.interactive === undefined) this.options.interactive = true;
    }
    addTo(target) { (target._isMap ? target._root : target).addLayer(this); return this; }
    remove() { this._group?.removeLayer(this); return this; }
    bindTooltip(html) { this._tip = html; return this; }
    bindPopup(html, opts) { this._pop = html; this._popOpts = opts || {}; return this; }
    setStyle(st) {
      const o = this.options, map = this._map;
      if (st.weight != null) this._state.w = st.weight;
      if (st.opacity != null) this._state.o = st.opacity;
      if (st.fillOpacity != null) this._state.fo = st.fillOpacity;
      map?._state(this);
      return this;
    }
    setRadius(r) { this._state.r = r; this._map?._state(this); return this; }
    features(order) {
      const o = this.options, out = [];
      const base = { id: this.id, sort: order, i: o.interactive ? 1 : 0 };
      if (this.kind === "circle") {
        out.push(["circle", { type: "Feature", id: this.id, geometry: { type: "Point", coordinates: ll2(this.latlngs) },
          properties: { ...base, r: o.radius ?? 10, sc: o.color ?? "#3388ff", sw: o.weight ?? 3, so: o.opacity ?? 1,
            fc: o.fillColor ?? o.color ?? "#3388ff", fo: o.fillOpacity ?? 0.2 } }]);
        return out;
      }
      const dash = o.dashArray ? (o.dashArray.startsWith("6") ? 2 : 1) : 0;
      const line = { ...base, c: o.color ?? "#3388ff", w: o.weight ?? 3, o: o.opacity ?? 1, d: dash };
      // Leaflet accepts one ring ([[lat, lon], ...]) or several ([[[lat, lon], ...], ...]); each ring here is its own part
      const nested = Array.isArray(this.latlngs[0]) && Array.isArray(this.latlngs[0][0]);
      const parts = (nested ? this.latlngs : [this.latlngs]).map((r) => r.map(ll2)).filter((r) => r.length);
      if (this.kind === "polygon") {
        const rings = parts.map((r) => (r[0][0] !== r[r.length - 1][0] || r[0][1] !== r[r.length - 1][1] ? [...r, r[0]] : r)).filter((r) => r.length >= 4);
        if (!rings.length) return out;
        out.push(["fill", { type: "Feature", id: this.id, geometry: { type: "MultiPolygon", coordinates: rings.map((r) => [r]) },
          properties: { ...base, fc: o.fillColor ?? o.color ?? "#3388ff", fo: o.fillOpacity ?? 0.2 } }]);
        out.push(["line", { type: "Feature", id: this.id, geometry: { type: "MultiLineString", coordinates: rings }, properties: line }]);
      } else if (parts.length) {
        out.push(["line", { type: "Feature", id: this.id, geometry: { type: "MultiLineString", coordinates: parts }, properties: line }]);
      }
      return out;
    }
  }

  class Marker {
    constructor(latlng, opts) { this.latlng = latlng; this.options = opts || {}; }
    addTo(target) { (target._isMap ? target._root : target).addLayer(this); return this; }
    _add(map) {
      const ic = this.options.icon || {};
      const el = document.createElement("div");
      el.className = ic.className || ""; el.innerHTML = ic.html || "";
      el.style.pointerEvents = "none";
      const a = ic.iconAnchor || [0, 0];
      this._m = new ml.Marker({ element: el, anchor: "top-left", offset: [-a[0], -a[1]] }).setLngLat(ll2(this.latlng)).addTo(map._gl);
    }
    _remove() { this._m?.remove(); this._m = null; }
  }

  class ImageOverlay {
    constructor(url, bounds, opts) { this._url = url; this._b = bounds instanceof Bounds ? bounds : new Bounds(bounds); this.options = opts || {}; this.id = "lgl-img-" + seq++; }
    addTo(target) { const map = target._isMap ? target : target._map; this._map = map; map._images.add(this); this._add(); return this; }
    _add() {
      const g = this._map._gl;
      if (!g.getSource("lgl-line") || g.getSource(this.id)) return;   // style not ready: added on style.load
      g.addSource(this.id, { type: "image", url: this._url, coordinates: this._b.corners() });
      g.addLayer({ id: this.id, type: "raster", source: this.id,
        paint: { "raster-opacity": this.options.opacity ?? 1, "raster-resampling": "nearest", "raster-fade-duration": 0 } }, this._map._below());
    }
    setUrl(url) { this._url = url; this._upd(); return this; }
    setBounds(b) { this._b = b instanceof Bounds ? b : new Bounds(b); this._upd(); return this; }
    _upd() { const s = this._map?._gl.getSource(this.id); if (s) s.updateImage({ url: this._url, coordinates: this._b.corners() }); }
    remove() {
      const g = this._map?._gl;
      if (g) { if (g.getLayer(this.id)) g.removeLayer(this.id); if (g.getSource(this.id)) g.removeSource(this.id); this._map._images.delete(this); }
      return this;
    }
  }

  class LayerGroup {
    constructor() { this._items = new Set(); this._map = null; }
    addTo(target) { const map = target._isMap ? target : target._map; this._map = map; map._groups.add(this); this._items.forEach((it) => this._attach(it)); return this; }
    addLayer(it) { this._items.add(it); it._group = this; if (this._map) this._attach(it); return this; }
    _attach(it) {
      if (it instanceof Shape) { it._map = this._map; this._map._dirty(); }
      else if (it instanceof Marker) it._add(this._map);
      else if (it instanceof ImageOverlay) it.addTo(this._map);
    }
    removeLayer(it) {
      this._items.delete(it);
      if (it instanceof Shape) { this._map?._dirty(); if (this._map?._hover === it) this._map._hover = null; }
      else if (it instanceof Marker) it._remove();
      else if (it instanceof ImageOverlay) it.remove();
      return this;
    }
    clearLayers() { [...this._items].forEach((it) => this.removeLayer(it)); return this; }
    shapes() { return [...this._items].filter((it) => it instanceof Shape); }
  }

  // ---------------------------------------------------------------- the map
  class GLMap extends Evented {
    constructor(id, opts) {
      super();
      this._isMap = true; this._groups = new Set(); this._images = new Set(); this._panes = {};
      this._container = typeof id === "string" ? document.getElementById(id) : id;
      this._root = new LayerGroup(); this._root.addTo(this);
      this._gl = new ml.Map({
        container: this._container, style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#e9e8e4" } }] },
        center: [-92, 36], zoom: 4, attributionControl: false, renderWorldCopies: false, dragRotate: false, pitchWithRotate: false,
        maxPitch: 0,
      });
      this._gl.touchZoomRotate.disableRotation();
      // keep the canvas the size of its box (the map panel grows after load as the page lays out)
      let rz = 0;
      new ResizeObserver(() => { cancelAnimationFrame(rz); rz = requestAnimationFrame(() => this._gl.resize()); }).observe(this._container);
      this.attributionControl = { addAttribution: () => {}, removeAttribution: () => {} };
      this._gl.on("style.load", () => this._styleLoaded());
      this._wireEvents();
    }
    // --- Leaflet-shaped API used by app.js
    setView(ll, z) { this._gl.jumpTo({ center: ll2(ll), zoom: Math.max(0, z - 1) }); return this; }
    fitBounds(b, o = {}) {
      const bb = b instanceof Bounds ? b : new Bounds(b);
      if (!bb.isValid()) return this;
      const pad = Array.isArray(o.padding) ? o.padding[0] : (o.padding ?? 0);
      this._gl.fitBounds(bb.toML(), { padding: pad, maxZoom: o.maxZoom != null ? o.maxZoom - 1 : 22, duration: 600 });
      return this;
    }
    createPane(name) { return (this._panes[name] ||= document.createElement("div")); }
    getPane(name) { return this.createPane(name); }
    getContainer() { return this._container; }
    removeLayer(l) { l?.remove?.(); return this; }
    getZoom() { return this._gl.getZoom() + 1; }
    // --- basemap: a MapLibre style (vector or raster); our layers are re-added on top after each load
    setBasemap(style, attribution) {
      this._attrHTML = attribution || "";
      this._gl.setStyle(style, { diff: false });
    }
    _styleLoaded() {
      const g = this._gl;
      try { g.setProjection({ type: "globe" }); } catch (_) { /* older MapLibre: flat */ }
      try {
        g.setSky({ "sky-color": "#0b1324", "horizon-color": "#1d2b45", "fog-color": "#1d2b45", "sky-horizon-blend": 0.6, "horizon-fog-blend": 0.6, "fog-ground-blend": 0.9, "atmosphere-blend": ["interpolate", ["linear"], ["zoom"], 0, 1, 6, 0.4, 8, 0] });
      } catch (_) { /* no sky support */ }
      if (this._attr) g.removeControl(this._attr);
      this._attr = new ml.AttributionControl({ compact: false, customAttribution: this._attrHTML || undefined });
      g.addControl(this._attr, "bottom-right");
      this._addVectorLayers();
      this._images.forEach((im) => im._add());
      this._flush();
    }
    _below() {
      // radar images sit above the basemap's land/water/lines but below its place labels and our shapes
      const ls = this._gl.getStyle()?.layers || [];
      const sym = ls.find((l) => l.type === "symbol" && !l.id.startsWith("lgl-"));
      return sym ? sym.id : (this._gl.getLayer("lgl-fill") ? "lgl-fill" : undefined);
    }
    _addVectorLayers() {
      const g = this._gl, empty = { type: "FeatureCollection", features: [] };
      for (const s of Object.values(SRC)) if (!g.getSource(s)) g.addSource(s, { type: "geojson", data: empty });
      const st = (k, prop) => ["coalesce", ["feature-state", k], ["get", prop]];
      if (!g.getLayer("lgl-fill")) g.addLayer({ id: "lgl-fill", type: "fill", source: SRC.fill, layout: { "fill-sort-key": ["get", "sort"] },
        paint: { "fill-color": ["get", "fc"], "fill-opacity": st("fo", "fo") } });
      const lineLayer = (id, filter, dash) => g.getLayer(id) || g.addLayer({ id, type: "line", source: SRC.line, filter,
        layout: { "line-cap": "round", "line-join": "round", "line-sort-key": ["get", "sort"] },
        paint: { "line-color": ["get", "c"], "line-width": st("w", "w"), "line-opacity": st("o", "o"), ...(dash ? { "line-dasharray": dash } : {}) } });
      lineLayer("lgl-line", ["==", ["get", "d"], 0]);
      lineLayer("lgl-line-d1", ["==", ["get", "d"], 1], [3.3, 2.7]);   // "5 4" at weight 1.5
      lineLayer("lgl-line-d2", ["==", ["get", "d"], 2], [3, 2.5]);     // "6 5" at weight 2
      if (!g.getLayer("lgl-circle")) g.addLayer({ id: "lgl-circle", type: "circle", source: SRC.circle, layout: { "circle-sort-key": ["get", "sort"] },
        paint: { "circle-radius": st("r", "r"), "circle-color": ["get", "fc"], "circle-opacity": ["get", "fo"], "circle-stroke-color": ["get", "sc"],
          "circle-stroke-width": st("w", "sw"), "circle-stroke-opacity": ["get", "so"], "circle-pitch-alignment": "map" } });
    }
    _dirty() { if (!this._pending) { this._pending = true; requestAnimationFrame(() => { this._pending = false; this._flush(); }); } }
    _flush() {
      const g = this._gl;
      if (!g.getSource(SRC.line)) return;
      const fc = { fill: [], line: [], circle: [] };
      let order = 0;
      this._byId = new Map();
      for (const grp of this._groups) for (const sh of grp.shapes()) {
        order++;
        this._byId.set(sh.id, sh);
        for (const [k, f] of sh.features(order)) fc[k].push(f);
      }
      for (const k of Object.keys(SRC)) g.getSource(SRC[k]).setData({ type: "FeatureCollection", features: fc[k] });
      for (const sh of this._byId.values()) if (Object.keys(sh._state).length) this._state(sh);
    }
    _state(sh) {
      const g = this._gl;
      if (!g.getSource(SRC.line)) return;
      const s = sh._state;
      if (sh.kind === "circle") g.setFeatureState({ source: SRC.circle, id: sh.id }, { r: s.r ?? null, w: s.w ?? null });
      else {
        g.setFeatureState({ source: SRC.line, id: sh.id }, { w: s.w ?? null, o: s.o ?? null });
        if (sh.kind === "polygon") g.setFeatureState({ source: SRC.fill, id: sh.id }, { fo: s.fo ?? null });
      }
    }
    // --- pointer: hover (mouseover/mousemove/mouseout + sticky tooltip) and click (popup or handler)
    _pick(pt) {
      const g = this._gl;
      const layers = ["lgl-circle", "lgl-line", "lgl-line-d1", "lgl-line-d2", "lgl-fill"].filter((l) => g.getLayer(l));
      if (!layers.length) return null;
      const fs = g.queryRenderedFeatures([[pt.x - TOL, pt.y - TOL], [pt.x + TOL, pt.y + TOL]], { layers });
      let best = null, rank = -1;
      for (const f of fs) {
        if (!f.properties.i) continue;
        const sh = this._byId?.get(f.id ?? f.properties.id);
        if (!sh) continue;
        // circles and lines above fills (as their layers are), then the latest added
        const r = (f.layer.id === "lgl-fill" ? 0 : f.layer.id === "lgl-circle" ? 2e9 : 1e9) + f.properties.sort;
        if (r > rank) { rank = r; best = sh; }
      }
      return best;
    }
    _wireEvents() {
      const g = this._gl, tip = document.createElement("div");
      tip.className = "leaflet-tooltip mt-tip lgl-tip"; tip.hidden = true;
      this._container.appendChild(tip);
      const placeTip = (e) => {
        const r = this._container.getBoundingClientRect();
        tip.style.left = `${e.originalEvent.clientX - r.left}px`; tip.style.top = `${e.originalEvent.clientY - r.top - 14}px`;
      };
      const out = () => {
        if (this._hover) { this._hover.fire("mouseout", {}); this._hover = null; }
        tip.hidden = true; g.getCanvas().style.cursor = "";
      };
      g.on("mousemove", (e) => {
        const sh = this._pick(e.point);
        if (sh !== this._hover) {
          out();
          if (sh) {
            this._hover = sh;
            const ev = { originalEvent: e.originalEvent, latlng: { lat: e.lngLat.lat, lng: e.lngLat.lng }, target: sh };
            sh.fire("mouseover", ev);
            if (sh._tip) { tip.innerHTML = typeof sh._tip === "function" ? sh._tip() : sh._tip; tip.hidden = false; }
            if (sh._pop || sh.has("click")) g.getCanvas().style.cursor = "pointer";
          }
        }
        if (this._hover) {
          this._hover.fire("mousemove", { originalEvent: e.originalEvent });
          if (!tip.hidden) placeTip(e);
        }
      });
      g.getCanvas().addEventListener("mouseleave", out);
      g.on("movestart", () => { tip.hidden = true; });
      g.on("click", (e) => {
        const sh = this._pick(e.point);
        if (!sh) return;
        const ev = { originalEvent: e.originalEvent, latlng: { lat: e.lngLat.lat, lng: e.lngLat.lng }, target: sh };
        if (sh.has("click")) { sh.fire("click", ev); return; }
        if (sh._pop) {
          tip.hidden = true;
          const html = typeof sh._pop === "function" ? sh._pop() : sh._pop;
          new ml.Popup({ maxWidth: `${sh._popOpts.maxWidth || 420}px`, className: "lgl-popup", focusAfterOpen: false })
            .setLngLat(e.lngLat).setHTML(html).addTo(g);
        }
      });
    }
  }

  // ---------------------------------------------------------------- controls and DOM helpers
  class Control {
    constructor(opts) { this.options = { position: "topright", ...(this.options || {}), ...(opts || {}) }; }
    addTo(map) {
      const el = this.onAdd(map);
      el.classList.add("maplibregl-ctrl", "lgl-ctrl");
      this._el = el;
      const pos = { topright: "top-right", topleft: "top-left", bottomright: "bottom-right", bottomleft: "bottom-left" }[this.options.position] || "top-right";
      map._gl.addControl({ onAdd: () => el, onRemove: () => el.remove() }, pos);
      return this;
    }
    getContainer() { return this._el; }
  }
  Control.extend = (proto) => {
    const C = class extends Control {};
    const { options, ...rest } = proto;
    C.prototype.options = options; Object.assign(C.prototype, rest);
    return C;
  };
  const control = (opts) => new Control(opts);
  control.scale = () => ({
    addTo(map) {
      map._gl.addControl(new ml.ScaleControl({ unit: "metric", maxWidth: 100 }), "bottom-left");
      map._gl.addControl(new ml.ScaleControl({ unit: "imperial", maxWidth: 100 }), "bottom-left");
      return this;
    },
  });

  window.L = {
    isGL: true,
    map: (id, opts) => new GLMap(id, opts),
    canvas: () => ({}),
    layerGroup: () => new LayerGroup(),
    polyline: (ll, o) => new Shape("line", ll, o),
    polygon: (ll, o) => new Shape("polygon", ll, o),
    circleMarker: (ll, o) => new Shape("circle", ll, o),
    marker: (ll, o) => new Marker(ll, o),
    divIcon: (o) => o,
    imageOverlay: (url, b, o) => new ImageOverlay(url, b, o),
    latLngBounds: (pts) => new Bounds(pts),
    control, Control,
    DomUtil: { create(tag, cls, parent) { const el = document.createElement(tag); if (cls) el.className = cls; parent?.appendChild(el); return el; } },
    DomEvent: {
      stop(ev) { const oe = ev?.originalEvent || ev; oe?.stopPropagation?.(); oe?.preventDefault?.(); },
      disableClickPropagation(el) { ["click", "dblclick", "mousedown", "pointerdown", "touchstart"].forEach((t) => el.addEventListener(t, (e) => e.stopPropagation())); },
      disableScrollPropagation(el) { el.addEventListener("wheel", (e) => e.stopPropagation()); },
    },
  };
})();
