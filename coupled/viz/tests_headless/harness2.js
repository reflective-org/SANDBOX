// Instructor mode, hash handling, controls and the dist CSV, under the same stub DOM.
// Usage: node harness2.js <inverse_lab.html>
const fs = require("fs");
const path = require("path");
const src = fs.readFileSync(path.join(__dirname, "harness.js"), "utf8");
// reuse harness.js's DOM+PNG stubs by re-running its prelude with a chosen hash
const prelude = src.slice(0, src.indexOf("// ---- run the page script"));
const HASH = "#case=60N_15km|sabr330|D2med&t=2.5&noise=1&sig=0.2&seed=hw1&key=1";

const bootstrap = prelude
  .replace('const htmlPath = process.argv[2], outPath = process.argv[3];',
           'const htmlPath = process.argv[2];')
  .replace('const location = { hash: "" };', `const location = { hash: ${JSON.stringify(HASH)} };`)
  + `
const scripts = [...html.matchAll(/<script>([\\s\\S]*?)<\\/script>/g)].map(m => m[1]);
const pageJS = scripts[scripts.length - 1];
new Function("window","document","localStorage","location","history","URL","Blob","atob","Image","setTimeout", pageJS)
  (window, document, localStorage, location, history, URL, Blob, atob, Image, setTimeout);
module.exports = { api: window.__d1, els, document };
`;
const tmp = path.join(__dirname, "_boot2.js");
fs.writeFileSync(tmp, bootstrap);
const { api, els } = require(tmp);

const fail = [];
const ok = (cond, msg) => { console.log(`${cond ? "  ok  " : "  FAIL"} ${msg}`); if (!cond) fail.push(msg); };

console.log("hash:", HASH);
ok(api.ready, "page booted");
ok(api.caseKey === "60N_15km|sabr330|D2med", `case from hash (${api.caseKey})`);
ok(api.k === Math.round(2.5 * api.GPD), `start time from hash: k=${api.k} (expect ${Math.round(2.5 * api.GPD)}) = day ${(api.k / api.GPD).toFixed(3)}`);

// instructor mode: clean columns present and equal to the noise-off values
const csv = api.csvText();
const lines = csv.split("\n").filter(l => l.length && !l.startsWith("#"));
const head = lines[0].split(",");
const cleanCols = head.filter(h => h.endsWith("_clean"));
const BASE = 28;   // 22 plume/time columns + 6 static entrained-background columns
ok(head.length === BASE + cleanCols.length,
   `columns: ${head.length} = ${BASE} + ${cleanCols.length} clean`);
ok(cleanCols.length >= 10, `clean columns: ${cleanCols.join(" ")}`);
for (const c of ["bg_so2_ppt","bg_n_percm3","bg_sa_um2_percm3","bg_reff_um",
                 "bg_partso4_percm3","bg_partso4_ug_m3"])
  ok(head.includes(c), `main CSV carries ${c}`);
ok(!cleanCols.some(c => c.startsWith("bg_")),
   "the static background is never noised, so it has no _clean twin");
ok(csv.includes("instructor mode"), "header announces instructor mode");
ok(csv.includes("sigma = 0.2") && csv.includes('seed = "hw1"'), "header records sigma and seed");

const row = lines[1 + 40].split(",").map(Number);            // some row well into the run
const ix = n => head.indexOf(n);
const noisy = row[ix("so2_obs_ppt")], cleanv = row[ix("so2_ppt_clean")];
ok(noisy !== cleanv, `noise applied: obs ${noisy.toExponential(4)} vs clean ${cleanv.toExponential(4)} (ratio ${(noisy / cleanv).toFixed(4)})`);
api.setNoise(false);
const k40 = Math.round(Number(lines[1 + 40].split(",")[0]) * api.GPD);
ok(Math.abs(api.clean("so2_ppt", k40) / cleanv - 1) < 1e-6,
   "the *_clean column equals the noise-off value");
ok(Math.abs(row[ix("tracer_corr")] / (row[ix("tracer_obs")] * row[ix("V_over_V0")]) - 1) < 1e-5,
   "tracer_corr = tracer_obs x V within the CSV row");
api.setNoise(true, 0.2, "hw1");

// dilution stays exact even with noise on
const V_hash = api.clean("V", api.k);
api.setNoise(true, 0.2, "other-seed");
ok(api.clean("V", api.k) === V_hash, "dilution is never noised (same under a different seed)");
api.setNoise(true, 0.2, "hw1");

// size-distribution CSV
const dcsv = api.csvText("dist");
const dlines = dcsv.split("\n").filter(l => l.length && !l.startsWith("#"));
ok(dlines.length === 81, `dist CSV: ${dlines.length - 1} bin rows + header`);
for (const c of ["dp_um","dlog10Dp","dNdlogDp_percm3","N_bin_percm3","dMdlogDp_ug_m3",
                 "M_bin_ug_m3","bg_dNdlogDp_percm3","bg_dMdlogDp_ug_m3","dNdlogDp_clean"])
  ok(dlines[0].split(",").includes(c), `dist CSV carries ${c}`);
ok(/total number = [0-9.e+-]+ \/cm3/.test(dcsv), "dist header carries the exact aggregates");
ok(/rho = 1\.83 g\/cm3/.test(dcsv) && /DRY midpoint diameters/.test(dcsv),
   "dist header states the mass assumption (pure H2SO4, dry diameters)");
ok(/AMBIENT air/.test(dcsv), "dist header states concentrations are ambient, not STP");

// the standalone background export
const bcsv = api.csvText("bg");
const blines = bcsv.split("\n").filter(l => l.length && !l.startsWith("#"));
ok(blines.length === 1 + 6 * 80, `background CSV: ${blines.length - 1} rows = 6 site x bg x 80 bins`);
ok(/STATIC for/.test(bcsv) && /x2\.1818/.test(bcsv),
   "background header explains it is static and density-scaled between altitudes");
ok(/AMBIENT air, NOT STP/.test(bcsv), "background header states ambient, not STP");
ok(bcsv.includes("SABRE-220") && bcsv.includes("SABRE-330") && !/\bsabr-/i.test(bcsv),
   "background labels name the SABRE campaign properly");
ok(!bcsv.includes("AER-2D"), "the AER-2D name stays out of user-facing text");

// controls
const k0 = api.k;
els["stepf"].__h.click();
ok(api.k === k0 + 1, `forward button: ${k0} -> ${api.k} (+30 min)`);
els["stepb"].__h.click(); els["stepb"].__h.click();
ok(api.k === k0 - 1, `back button: ${api.k}`);
els["scrub"].value = "5"; els["scrub"].__h.input();
ok(api.k === 5, `scrubber: k=${api.k}`);
els["in-day"].value = "3"; els["in-time"].value = "18:00"; els["in-day"].__h.change();
const hh = (api.meta.start_utc_hour + (api.k / api.GPD) * 24) % 24;
ok(Math.abs(hh - 18) < 1e-6 && Math.floor(api.k / api.GPD) === 3,
   `typed "day 3, 18:00" -> k=${api.k} = day ${(api.k / api.GPD).toFixed(4)}, ${hh.toFixed(2)}:00 UTC`);
els["nz-on"].checked = false; els["nz-on"].__h.change();
ok(api.values.so2_ppt === api.clean("so2_ppt", api.k), "unchecking noise returns exact values");

// case switch keeps the sampling time when the new case is long enough
api.setCase("30N_20km|sabr220|D1low");
ok(api.k === Math.round(3.75 * api.GPD) || api.k > 0, `case switch keeps time: k=${api.k}`);
api.setCase("60N_15km|aergeo|D5vhigh");
ok(api.k <= Math.round(api.daysTotal * api.GPD),
   `switching to a shorter case (${api.daysTotal} d) clamps time: k=${api.k}`);

fs.unlinkSync(tmp);
console.log(fail.length ? `\n${fail.length} FAILURES` : "\nall interaction checks passed");
process.exit(fail.length ? 1 : 0);
