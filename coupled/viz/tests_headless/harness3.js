// Exercise the new "t =" field: parsing, snapping, and write-back.
const fs=require("fs"), path=require("path");
const T="/Users/ali/Documents/GitHub/gas-phase-chemistry/SANDBOX/coupled/viz/tests_headless";
const src=fs.readFileSync(path.join(T,"harness.js"),"utf8");
const prelude=src.slice(0,src.indexOf("// ---- run the page script"))
  .replace('const htmlPath = process.argv[2], outPath = process.argv[3];','const htmlPath = process.argv[2];');
const tmp=path.join(__dirname,"_boot3.js");
fs.writeFileSync(tmp, prelude+`
const scripts=[...html.matchAll(/<script>([\\s\\S]*?)<\\/script>/g)].map(m=>m[1]);
new Function("window","document","localStorage","location","history","URL","Blob","atob","Image","setTimeout",scripts[scripts.length-1])
 (window,document,localStorage,location,history,URL,Blob,atob,Image,setTimeout);
module.exports={api:window.__d1,els};
`);
const {api,els}=require(tmp);
const fail=[];
const ok=(c,m)=>{console.log(`${c?"  ok  ":"  FAIL"} ${m}`); if(!c)fail.push(m);};
api.setCase("30N_20km|sabr220|D1low");            // 36.4 d, release 06:00 UTC
const GPD=api.GPD;
const type=v=>{ els["in-t"].value=v; els["in-t"].__h.change(); return api.k/GPD; };
console.log(`case ${api.caseKey}, lifetime ${api.daysTotal} d, grid ${GPD}/day\n`);
for(const [input,expect,why] of [
  ["4.5",        4.5,   "plain decimal day"],
  ["4.5d",       4.5,   "day with unit"],
  ["0",          0,     "zero"],
  ["12",         12,    "integer day"],
  ["108h",       4.5,   "hours"],
  ["108 h",      4.5,   "hours with a space"],
  ["36h",        1.5,   "hours under a day"],
  ["90m",        0.0625, "minutes"],
  ["4d 18:00",   4.5,   "day plus clock time (release is 06:00 UTC)"],
  ["18:00",      null,  "bare clock time keeps the current day"],
  ["4.51",       4.5,   "snaps to the nearest 30 min (down)"],
  ["4.49",       4.5,   "snaps to the nearest 30 min (up)"],
  ["1e1",        10,    "scientific notation"],
  ["-3",         0,     "negative clamps to the start"],
  ["999",        null,  "beyond the lifetime clamps to the end"],
]){
  if(input==="18:00") api.setK(Math.round(7*GPD));            // set day 7 first
  const got=type(input);
  const want = expect!==null ? expect
    : input==="18:00" ? 7.5 : Math.round(api.daysTotal*GPD)/GPD;
  ok(Math.abs(got-want)<1e-9, `"${input}" -> t = ${got} d  (${why})`);
}
// garbage restores the displayed value instead of jumping
api.setK(Math.round(4.5*GPD));
const before=api.k; type("banana");
ok(api.k===before, `"banana" leaves t at ${api.k/GPD} d and restores the field to "${els["in-t"].value}"`);
// write-back: the field always states the time actually shown
api.setK(Math.round(4.5*GPD));
ok(els["in-t"].value==="4.5", `write-back after setK: field reads "${els["in-t"].value}"`);
els["stepf"].__h.click();
ok(els["in-t"].value==="4.5208", `after one 30-min step: "${els["in-t"].value}" (= 4.5 + 1/48)`);
ok(els["in-day"].value===4, `the day box stays in sync: ${els["in-day"].value}`);
ok(els["in-time"].value==="18:30", `and the clock box: ${els["in-time"].value} (06:00 release + 108.5 h)`);
// typing t must agree with the value table
type("4.5");
const s=api.snapshot();
ok(Math.abs(s.day-4.5)<1e-9 && s.hours===108, `table header time: t = ${s.day} d = ${s.hours} h`);
ok(api.clean("V",api.k)===api.clean("V",Math.round(4.5*GPD)), "values match the k for that t");
fs.unlinkSync(tmp);
console.log(fail.length?`\n${fail.length} FAILURES`:"\nall t-field checks passed");
process.exit(fail.length?1:0);
