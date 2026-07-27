// Run inverse_lab.html's own JavaScript under a stub DOM and dump values for comparison.
// Usage: node harness.js <inverse_lab.html> <out.json>
const fs = require("fs");
const zlib = require("zlib");

const htmlPath = process.argv[2], outPath = process.argv[3];
const html = fs.readFileSync(htmlPath, "utf8");

// ---- minimal PNG decoder (grayscale, 8-bit, non-interlaced -- what PIL writes) --------------
function decodePNG(buf) {
  let p = 8, w = 0, h = 0, bitDepth = 0, colorType = 0;
  const idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p), type = buf.toString("ascii", p + 4, p + 8);
    const data = buf.slice(p + 8, p + 8 + len);
    if (type === "IHDR") {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      bitDepth = data[8]; colorType = data[9];
      if (bitDepth !== 8 || colorType !== 0) throw new Error(`unsupported PNG ${bitDepth}/${colorType}`);
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    p += 12 + len;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const out = Buffer.alloc(w * h);
  let prev = Buffer.alloc(w);
  for (let y = 0; y < h; y++) {
    const ft = raw[y * (w + 1)], line = raw.slice(y * (w + 1) + 1, y * (w + 1) + 1 + w);
    const cur = Buffer.alloc(w);
    for (let x = 0; x < w; x++) {
      const a = x > 0 ? cur[x - 1] : 0, b = prev[x], c = x > 0 ? prev[x - 1] : 0;
      let v = line[x];
      if (ft === 1) v += a;
      else if (ft === 2) v += b;
      else if (ft === 3) v += (a + b) >> 1;
      else if (ft === 4) {
        const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      cur[x] = v & 255;
    }
    cur.copy(out, y * w);
    prev = cur;
  }
  return { width: w, height: h, gray: out };
}

// ---- stub DOM -------------------------------------------------------------------------------
const noop = () => {};
function makeCtx() {
  const ctx = new Proxy({}, {
    get(t, k) {
      if (k === "canvas") return t.__cv;
      if (k in t) return t[k];
      if (k === "measureText") return () => ({ width: 10 });
      if (k === "getImageData") return (x, y, w, h) => {
        const img = t.__img;
        const data = new Uint8ClampedArray(w * h * 4);
        if (img) for (let yy = 0; yy < Math.min(h, img.height); yy++)
          for (let xx = 0; xx < Math.min(w, img.width); xx++) {
            const g = img.gray[yy * img.width + xx], o = (yy * w + xx) * 4;
            data[o] = data[o + 1] = data[o + 2] = g; data[o + 3] = 255;
          }
        return { data, width: w, height: h };
      };
      if (k === "createImageData") return (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h });
      if (k === "drawImage") return (img) => { if (img && img.gray) t.__img = img; };
      return noop;                                       // every other 2-D call is a no-op
    },
    set(t, k, v) { t[k] = v; return true; },
  });
  return ctx;
}
function makeEl(id) {
  const el = {
    id, textContent: "", innerHTML: "", value: "0", max: "1000", min: "0", checked: false,
    hidden: false, options: { length: 0 }, dataset: {}, children: [],
    style: { setProperty: noop, cursor: "" },
    classList: { toggle: noop, add: noop, remove: noop, contains: () => false },
    addEventListener(ev, fn) { (this.__h = this.__h || {})[ev] = fn; },
    removeEventListener: noop, appendChild(c) { this.children.push(c); },
    setAttribute: noop, getAttribute: () => null, closest: () => null,
    setPointerCapture: noop, focus: noop,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 400, height: 200 }),
    querySelectorAll: () => [],
  };
  el.getContext = () => { const c = makeCtx(); c.__cv = el; el.__ctx = c; return c; };
  return el;
}
const els = {};
const document = {
  documentElement: { dataset: {} },
  getElementById(id) { return els[id] || (els[id] = makeEl(id)); },
  querySelectorAll: () => [], querySelector: () => makeEl("q"),
  createElement(tag) { return makeEl(tag); },
  addEventListener: noop, activeElement: null,
};
els["d1-data"] = makeEl("d1-data");
const i0 = html.indexOf("/*__D1_DATA_BEGIN__*/"), i1 = html.lastIndexOf("/*__D1_DATA_END__*/");
els["d1-data"].textContent = html.slice(i0, i1 + "/*__D1_DATA_END__*/".length);

class Image {
  set src(v) {
    const b64 = v.slice(v.indexOf(",") + 1);
    const d = decodePNG(Buffer.from(b64, "base64"));
    this.width = d.width; this.height = d.height; this.gray = d.gray;
    if (this.onload) this.onload();
  }
}
const localStorage = { getItem: () => null, setItem: noop, removeItem: noop };
const location = { hash: "" };
const history = { replaceState: noop };
const window = {
  matchMedia: () => ({ matches: false, addEventListener: noop }),
  devicePixelRatio: 1, addEventListener: noop, localStorage, location, history,
  requestAnimationFrame: noop, Image, document,
};
window.window = window;
const URL = { createObjectURL: () => "blob:x", revokeObjectURL: noop };
class Blob { constructor(p) { this.parts = p; } }
const atob = (s) => Buffer.from(s, "base64").toString("binary");
const setTimeout_ = setTimeout;

// ---- run the page script --------------------------------------------------------------------
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const pageJS = scripts[scripts.length - 1];
const run = new Function("window", "document", "localStorage", "location", "history", "URL",
  "Blob", "atob", "Image", "setTimeout", pageJS);
run(window, document, localStorage, location, history, URL, Blob, atob, Image, setTimeout_);

const api = window.__d1;
if (!api || !api.ready) { console.error("page did not boot:", api && api.error); process.exit(1); }

// ---- collect what we want to verify ---------------------------------------------------------
const NAMES = ["V", "so2_ppt", "oh", "h2so4", "part_s", "total_n", "sa", "reff_um", "so2_t", "sulf_t"];
const out = { cases: api.listCases(), gpd: api.GPD, probe: {} };

for (const key of ["30N_20km|sabr220|D2med", "60N_15km|aergeo|D5vhigh", "30N_20km|sabr220|D1low"]) {
  api.setCase(key);
  api.setNoise(false);
  const nt = Math.round(api.daysTotal * api.GPD);
  const ks = [0, 1, 12, 48, 100, Math.floor(nt / 2), nt - 1].filter(k => k >= 0 && k <= nt);
  const rec = { daysTotal: api.daysTotal, nt, ks, clean: {}, snapshot: {}, dist: {} };
  for (const n of NAMES) rec.clean[n] = ks.map(k => api.clean(n, k));
  for (const k of ks) {
    const s = api.snapshot(k);
    rec.snapshot[k] = { V: s.V, tracer: s.tracer, tracer_corr: s.tracer_corr,
      so2_ppt: s.so2_ppt, so2_cm3: s.so2_cm3, so2_ex_t: s.so2_ex_t, so2_model_t: s.so2_model_t,
      oh: s.oh, h2so4: s.h2so4, h2so4_ppt: s.h2so4_ppt, part: s.part, part_ug: s.part_ug,
      n: s.n, reff: s.reff, sa: s.sa, side: s.side, day: s.day };
    rec.dist[k] = Array.from(api.distAt(k));
  }
  // tracer identity across every grid point
  let worstId = 0;
  for (let k = 0; k <= nt; k++) {
    const s = api.snapshot(k);
    worstId = Math.max(worstId, Math.abs(s.tracer_corr - 1));
  }
  rec.tracer_identity_worst = worstId;

  // CSV: shape + determinism
  const csv = api.csvText();
  const lines = csv.trim().split("\n");
  rec.csv_rows = lines.filter(l => !l.startsWith("#")).length;      // header + data
  rec.csv_cols = lines.find(l => !l.startsWith("#")).split(",").length;
  rec.csv_header = lines.find(l => !l.startsWith("#"));
  rec.csv_same_twice = api.csvText() === csv;

  // noise: reproducible per seed, independent of visit order, and actually applied
  api.setNoise(true, 0.1, "abc");
  const a1 = [0, 5, 50].map(k => api.snapshot(Math.min(k, nt)).so2_ppt);
  const csvA = api.csvText();
  api.setCase(key === "30N_20km|sabr220|D2med" ? "60N_15km|sabr330|D2med" : "30N_20km|sabr220|D2med");
  api.setCase(key);
  api.setNoise(true, 0.1, "abc");
  const a2 = [0, 5, 50].map(k => api.snapshot(Math.min(k, nt)).so2_ppt);
  api.setNoise(true, 0.1, "zzz");
  const a3 = [0, 5, 50].map(k => api.snapshot(Math.min(k, nt)).so2_ppt);
  api.setNoise(false);
  const a0 = [0, 5, 50].map(k => api.snapshot(Math.min(k, nt)).so2_ppt);
  rec.noise = { same_seed_stable: a1.every((v, i) => v === a2[i]),
    diff_seed_differs: a1.every((v, i) => v !== a3[i]),
    csv_stable: api.csvText() !== csvA,          // noise off now, so must differ from noisy csv
    ratios: a1.map((v, i) => v / a0[i]) };
  out.probe[key] = rec;
}
fs.writeFileSync(outPath, JSON.stringify(out));
console.log("harness ok:", out.cases.length, "cases");
