// Timeline of the labeling windows (Streamlit component v2, plain SVG, no build step).
//
// Main panel: x = hours relative to an issue time t_X, y = GOES peak flux (log scale).
//   - flare window [t_X, t_X + W]          handle on its right edge
//   - grid of the issue times t_X + kΔ     diamonds on the time axis, any of them sets Δ
//   - before/after windows of length L     handles on their outer edges, or ∞ bands when unlimited
//   - thresholds S_p (solid) and S_s (dashed) as horizontal lines
// Optional second row: the time range of t_X (dates), a bar with two handles.
//
// Data flow: Python passes the current values in `data`; a drag changes them locally and sends
// only the changed fields with setStateValue("value", {...fields, seq}). Python applies them and
// echoes the last applied `seq` in `data`: data with an older seq (a rerun started by an earlier
// message) is ignored, so the handles never jump back while or after the user drags.

const NS = "http://www.w3.org/2000/svg";
const M = { left: 64, right: 104, top: 30, bottom: 36 };   // plot margins (px); right: threshold labels
const H_MAIN = 290, H_RANGE = 80;
const CLASS_FLUX = [["A", 1e-8], ["B", 1e-7], ["C", 1e-6], ["M", 1e-5], ["X", 1e-4]];
const Y_MIN = 4e-8, Y_MAX = 2e-3;                         // flux range of the y axis (A4 … X20)
// snap units of the time handles (hours): the smallest one that is at least ~4 px wide is used
const X_UNITS = [0.5, 1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720];
const DAY = 86400000;
const THROTTLE_MS = 250;

// ---------------------------------------------------------------------------------------- helpers
function el(tag, attrs = {}, parent = null, text = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// "36 h", "3 d 12 h", "2 w"
function fmtHours(h) {
  if (h < 72) return `${+h.toFixed(2)} h`;
  const d = Math.floor(h / 24), r = +(h - 24 * d).toFixed(2);
  return r ? `${d} d ${r} h` : `${d} d`;
}

// GOES class with an integer multiplier (1e-5 → "M", 5e-5 → "M5"), as cs_gui.labeling_form.flux_class
function fluxClass(f) {
  let cls = CLASS_FLUX[0];
  for (const c of CLASS_FLUX) if (f >= c[1] * (1 - 1e-9)) cls = c;
  const m = Math.max(1, Math.round(f / cls[1]));
  return m === 1 ? cls[0] : cls[0] + m;
}

// snap a flux to an integer multiple of its class (log scale: the multipliers 1…9 of each decade)
function snapFlux(f) {
  let cls = CLASS_FLUX[0];
  for (const c of CLASS_FLUX) if (f >= c[1] * 0.95) cls = c;
  return clamp(Math.max(1, Math.round(f / cls[1])) * cls[1], Y_MIN, Y_MAX);
}

const isoDay = (ms) => new Date(ms).toISOString().slice(0, 10);
const dayMs = (iso) => Date.parse(iso + "T00:00:00Z");

// ------------------------------------------------------------------------------------- component
function create(root) {
  const S = {
    v: null, lim: null, dirty: {}, lastSeq: null, drag: null, lastEmit: 0, timer: null,
    xd: null,               // x domain [min, max] hours, recomputed when no drag is in progress
    setStateValue: null,
  };
  const uid = Math.random().toString(36).slice(2, 8);   // ids of gradients/clip paths, unique per instance
  const wrap = document.createElement("div");
  wrap.className = "tl";
  root.appendChild(wrap);
  const svg = el("svg", { class: "tl-main", role: "group", "aria-label": "Labeling windows timeline" });
  const svgR = el("svg", { class: "tl-range", role: "group", "aria-label": "Time range of t_X" });
  wrap.appendChild(svg);
  wrap.appendChild(svgR);
  const help = document.createElement("div");
  help.className = "tl-help";
  help.textContent = "Drag the handles (or focus one and use the arrow keys). Exact values: fields below.";
  wrap.appendChild(help);

  // ---- send the changed fields (throttled while dragging, at once at the end of a drag)
  function emit(now = false) {
    const fire = () => {
      S.timer = null;
      S.lastEmit = Date.now();
      S.lastSeq = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      S.setStateValue("value", { ...S.dirty, seq: S.lastSeq });
    };
    if (now) { clearTimeout(S.timer); fire(); return; }
    if (S.timer) return;
    S.timer = setTimeout(fire, Math.max(0, THROTTLE_MS - (Date.now() - S.lastEmit)));
  }

  function set(field, value) {
    S.v[field] = value;
    // what Python receives: hours as numbers, thresholds as GOES classes, dates as ISO days
    S.dirty[field] = field === "thr" || field === "sthr" ? fluxClass(value)
      : field === "start" || field === "end" ? isoDay(value) : value;
  }

  // ---- x domain (hours around t_X) that shows every window with some margin
  function xDomain() {
    const v = S.v, Lv = v.L_on ? v.L : Math.max(v.W, v.step) * 0.9;
    return [-Math.max(Lv, 1.2 * v.step, 0.35 * v.W) * 1.12, Math.max(v.W + Lv, 2.3 * v.step, 1.4 * v.W) * 1.08];
  }

  // ---- geometry of the main panel: pixel ↔ value
  function geom() {
    const width = Math.max(320, wrap.clientWidth || 600);
    const pw = width - M.left - M.right, ph = H_MAIN - M.top - M.bottom;
    const [x0, x1] = S.xd;
    const ly0 = Math.log10(Y_MIN), ly1 = Math.log10(Y_MAX);
    return {
      width, pw, ph,
      X: (h) => M.left + (h - x0) / (x1 - x0) * pw,
      Xinv: (px) => x0 + (px - M.left) / pw * (x1 - x0),
      // log scale: y pixel is linear in log10(flux)
      Y: (f) => M.top + (1 - (Math.log10(f) - ly0) / (ly1 - ly0)) * ph,
      Yinv: (py) => Math.pow(10, ly0 + (1 - (py - M.top) / ph) * (ly1 - ly0)),
      hpp: (x1 - x0) / pw,   // hours per pixel
    };
  }
  const xUnit = (g) => X_UNITS.find((u) => u >= 4 * g.hpp) || X_UNITS[X_UNITS.length - 1];
  const snapH = (h, g) => Math.round(h / xUnit(g)) * xUnit(g);

  // ---- geometry of the range row: pixel ↔ day
  function geomR() {
    const width = Math.max(320, wrap.clientWidth || 600);
    const pw = width - M.left - M.right;
    const a = S.lim.range_min, b = S.lim.range_max;
    return {
      X: (ms) => M.left + (ms - a) / (b - a) * pw,
      Xinv: (px) => a + (px - M.left) / pw * (b - a),
    };
  }

  // ------------------------------------------------------------------------------------ drawing
  function handle(parent, kind, cx, cy, shape, label, cls) {
    const g = el("g", { class: `tl-handle ${cls}`, tabindex: 0, role: "slider", "aria-label": label,
                        "data-kind": kind }, parent);
    el("circle", { cx, cy, r: 14, class: "tl-hit" }, g);        // larger target for the pointer
    if (shape === "diamond") el("path", { d: `M${cx},${cy - 8} L${cx + 8},${cy} L${cx},${cy + 8} L${cx - 8},${cy} Z` }, g);
    else el("circle", { cx, cy, r: 7 }, g);
    el("title", {}, g, label);
    g.addEventListener("pointerdown", (e) => startDrag(e, kind));
    g.addEventListener("keydown", (e) => keyMove(e, kind));
    return g;
  }

  function bubble(parent, x, y, text) {
    const g = el("g", { class: "tl-bubble" }, parent);
    const t = el("text", { x, y: y - 14, "text-anchor": "middle" }, g, text);
    const w = Math.max(40, text.length * 8 + 16);
    g.insertBefore(el("rect", { x: x - w / 2, y: y - 33, width: w, height: 25, rx: 5 }), t);
  }

  function drawMain() {
    const v = S.v, g = geom();
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${g.width} ${H_MAIN}`);
    svg.setAttribute("width", g.width);
    svg.setAttribute("height", H_MAIN);
    const yTop = M.top, yBot = M.top + g.ph, xL = M.left, xR = M.left + g.pw;
    const defs = el("defs", {}, svg);
    for (const [id, x1, x2] of [[`fadeL-${uid}`, "1", "0"], [`fadeR-${uid}`, "0", "1"]]) {
      const lg = el("linearGradient", { id, x1, x2, y1: "0", y2: "0" }, defs);
      el("stop", { offset: "0", class: "tl-fade-on" }, lg);
      el("stop", { offset: "1", class: "tl-fade-off" }, lg);
    }
    el("clipPath", { id: `plot-${uid}` }, defs).appendChild(el("rect", { x: xL, y: yTop, width: g.pw, height: g.ph }));

    // y axis: decades of GOES classes, gridlines
    for (const [c, f] of CLASS_FLUX) {
      if (f < Y_MIN || f > Y_MAX) continue;
      const y = g.Y(f);
      el("line", { x1: xL, x2: xR, y1: y, y2: y, class: "tl-grid" }, svg);
      el("text", { x: xL - 8, y: y + 4, "text-anchor": "end", class: "tl-axis" }, svg, c);
    }
    el("text", { x: 12, y: (yTop + yBot) / 2, class: "tl-axis tl-ylabel",
                 transform: `rotate(-90 12 ${(yTop + yBot) / 2})`, "text-anchor": "middle" }, svg, "GOES flux (log)");

    // windows
    const P = el("g", { "clip-path": `url(#plot-${uid})` }, svg);
    const x0 = g.X(0), xW = g.X(v.W);
    if (v.L_on) {
      el("rect", { x: g.X(-v.L), y: yTop, width: x0 - g.X(-v.L), height: g.ph, class: "tl-strat" }, P);
      el("rect", { x: xW, y: yTop, width: g.X(v.W + v.L) - xW, height: g.ph, class: "tl-strat" }, P);
    } else {   // unlimited: bands fading towards the borders, with ∞
      el("rect", { x: xL, y: yTop, width: x0 - xL, height: g.ph, fill: `url(#fadeL-${uid})` }, P);
      el("rect", { x: xW, y: yTop, width: xR - xW, height: g.ph, fill: `url(#fadeR-${uid})` }, P);
      el("text", { x: xL + 16, y: yTop + 22, class: "tl-inf" }, svg, "∞");
      el("text", { x: xR - 16, y: yTop + 22, class: "tl-inf", "text-anchor": "end" }, svg, "∞");
    }
    el("rect", { x: x0, y: yTop, width: xW - x0, height: g.ph, class: "tl-flare" }, P);

    // band captions (only where there is room)
    // caption of a band: the first text that fits its width
    const cap = (x, w, ...texts) => {
      const text = texts.find((t) => w > t.length * 7.5);
      if (text) el("text", { x: x + w / 2, y: yBot - 8, "text-anchor": "middle", class: "tl-cap" }, svg, text);
    };
    const bL = v.L_on ? x0 - g.X(-v.L) : x0 - xL, aL = v.L_on ? g.X(v.W + v.L) - xW : xR - xW;
    cap(x0 - bL, bL, v.L_on ? `before · L = ${fmtHours(v.L)}` : "before (whole past)", "before");
    cap(xW, aL, v.L_on ? `after · L = ${fmtHours(v.L)}` : "after (whole future)", "after");
    cap(x0, xW - x0, `flare window W = ${fmtHours(v.W)}`, `W = ${fmtHours(v.W)}`, "W");

    // thresholds
    // S_p: handle and label right of the plot; S_s: handle inside the plot, label above the line,
    // so that both can be grabbed when the two thresholds coincide
    const thrLine = (kind, f, cls, name, hx) => {
      const y = g.Y(clamp(f, Y_MIN, Y_MAX));
      el("line", { x1: xL, x2: xR, y1: y, y2: y, class: `tl-thr ${cls}` }, svg);
      const hit = el("line", { x1: xL, x2: xR, y1: y, y2: y, class: "tl-thr-hit" }, svg);
      hit.addEventListener("pointerdown", (e) => startDrag(e, kind));
      if (hx > xR) el("text", { x: hx + 12, y: y + 5, class: `tl-thr-label ${cls}` }, svg, `${name} ${fluxClass(f)}`);
      else el("text", { x: hx, y: y - 12, "text-anchor": "middle", class: `tl-thr-label ${cls}` }, svg, `${name} ${fluxClass(f)}`);
      return () => handle(svg, kind, hx, y, "circle", `${name} = ${fluxClass(f)} (${f.toExponential(0)} W/m²)`, cls);
    };
    const hp = thrLine("thr", v.thr, "tl-p", "S_p", xR + 10);
    const hs = v.sthr_on ? thrLine("sthr", v.sthr, "tl-s", "S_s", xR - 34) : null;
    hp();
    if (hs) hs();

    // time axis with the grid of t_X: t_X + kΔ
    el("line", { x1: xL, x2: xR, y1: yBot, y2: yBot, class: "tl-axisline" }, svg);
    const [d0, d1] = S.xd;
    const kMin = Math.ceil(d0 / v.step), kMax = Math.floor(d1 / v.step);
    const every = Math.max(1, Math.ceil((kMax - kMin + 1) / 40));     // at most ~40 diamonds
    for (let k = kMin; k <= kMax; k++) {
      if (k === 0) continue;
      const x = g.X(k * v.step);
      el("line", { x1: x, x2: x, y1: yTop, y2: yBot, class: "tl-tick" }, svg);
      if (k % every === 0) handle(svg, `step:${k}`, x, yBot, "diamond",
        `Step Δ = ${fmtHours(v.step)} (issue time t_X ${k > 0 ? "+" : "−"} ${Math.abs(k)}Δ)`, "tl-h-step");
    }
    // a few time labels: t_X, W, L edges
    el("text", { x: x0, y: yBot + 22, "text-anchor": "middle", class: "tl-axis tl-strong" }, svg, "t_X");
    const lab = (h) => { const x = g.X(h); if (x > xL + 20 && x < xR - 20) el("text", { x, y: yBot + 22, "text-anchor": "middle", class: "tl-axis" }, svg, `${h > 0 ? "+" : "−"}${fmtHours(Math.abs(h))}`); };
    if (Math.abs(xW - x0) > 50) lab(v.W);
    if (v.L_on) { if (g.X(v.W + v.L) - xW > 50) lab(v.W + v.L); if (x0 - g.X(-v.L) > 50) lab(-v.L); }
    if (g.X(v.step) - x0 > 50 && Math.abs(g.X(v.step) - xW) > 50) lab(v.step);

    // window handles on the top edge
    el("line", { x1: x0, x2: x0, y1: yTop, y2: yBot, class: "tl-tx" }, svg);
    handle(svg, "W", xW, yTop, "circle", `Flare window W = ${fmtHours(v.W)}`, "tl-h-w");
    if (v.L_on) {
      handle(svg, "L+", g.X(v.W + v.L), yTop, "circle", `Stratification window L = ${fmtHours(v.L)}`, "tl-h-l");
      handle(svg, "L-", g.X(-v.L), yTop, "circle", `Stratification window L = ${fmtHours(v.L)}`, "tl-h-l");
    }

    // value bubble while dragging
    if (S.drag && S.drag.area === "main") {
      const k = S.drag.kind;
      if (k === "W") bubble(svg, xW, yTop, `W = ${fmtHours(v.W)}`);
      else if (k.startsWith("L")) bubble(svg, k === "L+" ? g.X(v.W + v.L) : g.X(-v.L), yTop, `L = ${fmtHours(v.L)}`);
      else if (k.startsWith("step")) bubble(svg, g.X(+k.split(":")[1] * v.step), yBot - 10, `Δ = ${fmtHours(v.step)}`);
      else { const f = v[k]; bubble(svg, xR - 60, g.Y(f), `${k === "thr" ? "S_p" : "S_s"} = ${fluxClass(f)} · ${f.toExponential(0)} W/m²`); }
    }
  }

  function drawRange() {
    const v = S.v;
    svgR.style.display = v.rng_on ? "" : "none";
    if (!v.rng_on) return;
    const width = Math.max(320, wrap.clientWidth || 600), r = geomR();
    svgR.replaceChildren();
    svgR.setAttribute("viewBox", `0 0 ${width} ${H_RANGE}`);
    svgR.setAttribute("width", width);
    svgR.setAttribute("height", H_RANGE);
    const y = 30, xL = M.left, xR = width - M.right;
    el("text", { x: xL - 8, y: y + 4, "text-anchor": "end", class: "tl-axis" }, svgR, "t_X");
    el("line", { x1: xL, x2: xR, y1: y, y2: y, class: "tl-axisline" }, svgR);
    const y0 = new Date(S.lim.range_min).getUTCFullYear(), y1 = new Date(S.lim.range_max).getUTCFullYear();
    const every = Math.max(1, Math.ceil((y1 - y0 + 1) / ((xR - xL) / 46)));
    for (let yr = y0 + 1; yr <= y1; yr++) {
      const x = r.X(Date.UTC(yr, 0, 1));
      el("line", { x1: x, x2: x, y1: y - 4, y2: y + 4, class: "tl-tick" }, svgR);
      if (yr % every === 0) el("text", { x, y: y + 22, "text-anchor": "middle", class: "tl-axis" }, svgR, String(yr));
    }
    const a = r.X(v.start), b = r.X(v.end);
    el("rect", { x: a, y: y - 6, width: Math.max(1, b - a), height: 12, rx: 3, class: "tl-flare" }, svgR);
    handle(svgR, "start", a, y, "circle", `From ${isoDay(v.start)}`, "tl-h-w");
    handle(svgR, "end", b, y, "circle", `To ${isoDay(v.end)}`, "tl-h-w");
    if (S.drag && S.drag.area === "range") {
      const k = S.drag.kind;
      bubble(svgR, k === "start" ? a : b, y, `${k === "start" ? "From" : "To"} ${isoDay(v[k])}`);
    }
  }

  function render() {
    if (!S.v) return;
    if (!S.drag) S.xd = xDomain();
    // the drawing is rebuilt: keep the keyboard focus on the same handle
    const focused = root.activeElement && root.activeElement.getAttribute && root.activeElement.getAttribute("data-kind");
    drawMain();
    drawRange();
    if (focused) {
      const sel = focused.startsWith("step") ? '[data-kind^="step"]' : `[data-kind="${focused}"]`;
      const h = wrap.querySelector(sel);
      if (h) h.focus();
    }
  }

  // ------------------------------------------------------------------------------- interaction
  // value of `kind` for a pointer at (px, py) of its svg, snapped and within the limits
  function apply(kind, px, py) {
    const v = S.v, lim = S.lim;
    if (kind === "start" || kind === "end") {
      const r = geomR();
      let ms = Math.round(r.Xinv(px) / DAY) * DAY;                 // snap to whole days
      ms = clamp(ms, lim.range_min, lim.range_max);
      if (kind === "start") ms = Math.min(ms, v.end - DAY); else ms = Math.max(ms, v.start + DAY);
      set(kind, ms);
      return;
    }
    const g = geom();
    // while dragging a time handle towards the right border, widen the domain so it can go on
    if (!["thr", "sthr"].includes(kind) && px > M.left + g.pw * 0.97) S.xd = [S.xd[0], S.xd[1] * 1.04];
    if (kind.startsWith("L") && px < M.left + g.pw * 0.03) S.xd = [S.xd[0] * 1.04, S.xd[1]];
    const h = g.Xinv(px);
    if (kind === "W") set("W", clamp(snapH(h, g), lim.W[0], lim.W[1]));
    else if (kind === "L+") set("L", clamp(snapH(h - v.W, g), lim.L[0], lim.L[1]));
    else if (kind === "L-") set("L", clamp(snapH(-h, g), lim.L[0], lim.L[1]));
    else if (kind.startsWith("step:")) {
      // the k-th issue time after t_X is at kΔ: Δ = h / k
      const k = +kind.split(":")[1];
      set("step", clamp(Math.round(h / k / xUnit(g)) * xUnit(g) || lim.step[0], lim.step[0], lim.step[1]));
    } else set(kind, snapFlux(g.Yinv(py)));
  }

  function local(e, target) {
    const b = target.getBoundingClientRect(), vb = target.viewBox.baseVal;
    return [(e.clientX - b.left) * vb.width / b.width, (e.clientY - b.top) * vb.height / b.height];
  }

  function startDrag(e, kind) {
    e.preventDefault();
    const area = kind === "start" || kind === "end" ? "range" : "main";
    const target = area === "range" ? svgR : svg;
    S.drag = { kind, area };
    target.setPointerCapture(e.pointerId);
    const move = (ev) => { const [x, y] = local(ev, target); apply(kind, x, y); render(); emit(); };
    const up = () => {
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
      target.removeEventListener("pointercancel", up);
      S.drag = null;
      render();
      emit(true);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
    target.addEventListener("pointercancel", up);
    render();
  }

  // arrow keys: one snap unit (times), one multiplier (thresholds), one day (dates)
  function keyMove(e, kind) {
    const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (!dir) return;
    e.preventDefault();
    const v = S.v, lim = S.lim, g = geom(), u = xUnit(g);
    if (kind === "W") set("W", clamp(v.W + dir * u, lim.W[0], lim.W[1]));
    else if (kind.startsWith("L")) set("L", clamp(v.L + (kind === "L-" ? -dir : dir) * u, lim.L[0], lim.L[1]));
    else if (kind.startsWith("step")) set("step", clamp(v.step + dir * u, lim.step[0], lim.step[1]));
    else if (kind === "start") set("start", clamp(v.start + dir * DAY, lim.range_min, v.end - DAY));
    else if (kind === "end") set("end", clamp(v.end + dir * DAY, v.start + DAY, lim.range_max));
    else {   // next/previous integer multiple of the class
      const f = v[kind], cls = fluxClass(f), base = CLASS_FLUX.find((c) => c[0] === cls[0])[1];
      const m = Math.round(f / base) + dir;
      set(kind, snapFlux(m < 1 ? base * 0.9 : m > 9 && cls[0] !== "X" ? base * 10 : m * base));
    }
    render();
    emit(true);
  }

  // ------------------------------------------------------------------------------ data from Python
  S.receive = (data) => {
    // ignore the data of reruns started before our last message (Python echoes the applied seq)
    if (S.drag || (S.lastSeq && data.seq !== S.lastSeq)) return;
    S.dirty = {};
    S.lim = { W: data.limits.W, step: data.limits.step, L: data.limits.L,
              range_min: dayMs(data.range_min), range_max: dayMs(data.range_max) };
    S.v = { W: data.W, step: data.step, L_on: data.L_on, L: data.L, thr: data.thr, sthr_on: data.sthr_on,
            sthr: data.sthr, rng_on: data.rng_on, start: dayMs(data.start), end: dayMs(data.end) };
    render();
  };

  new ResizeObserver(() => render()).observe(wrap);
  return S;
}

export default function (component) {
  const { data, setStateValue, parentElement } = component;
  // the function runs at every rerun: the DOM and the state are created once and kept
  let S = parentElement.__labelTimeline;
  if (!S) {
    S = create(parentElement);
    parentElement.__labelTimeline = S;
  }
  S.setStateValue = setStateValue;
  if (data) S.receive(data);
}
