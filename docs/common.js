// 全章で共通の描画関数
const POS = "#eb6834", NEG = "#2a78d6";

// 全体の地図。下から、土台の第1章、入力で選ぶ第2・3章、その上で行う計算の第4・5章と積み上げる。今いる章を濃く塗る
const MAP_ROWS = [
  [["ch4_optimization.html", "第4章", "構造最適化と振動", "opt freq"], ["ch5_orbital.html", "第5章", "光の吸収", "TD"]],
  [["ch2_basis.html", "第2章", "基底関数", "def2-SVP"], ["ch3_functional.html", "第3章", "汎関数", "B3LYP"]],
  [["ch1_intro.html", "第1章", "土台：方程式を近似して解く仕組み", ""]]];
function drawMap(id, here) {
  const block = ([href, no, name, code]) => `<a class="${href === here ? "on" : ""}" href="${href}">
    <span class="no">${no}</span><b>${name}</b>${code ? `<code>${code}</code>` : ""}</a>`;
  document.getElementById(id).innerHTML = MAP_ROWS.map(r => `<div class="map-row">${r.map(block).join("")}</div>`).join("");
}

// 格子データ → 3Dmol 用の cube 形式の文字列
function decode(b64) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
function cube(d, values) {
  const n = d.n, s = d.step, o = d.origin;
  const head = ["cube", "cube", `${d.atoms.length} ${o[0]} ${o[1]} ${o[2]}`,
    `${n} ${s} 0 0`, `${n} 0 ${s} 0`, `${n} 0 0 ${s}`,
    ...d.atoms.map(a => `${a[0]} 0.0 ${a[1]} ${a[2]} ${a[3]}`)];
  const body = [];
  for (let i = 0; i < values.length; i += 6) body.push(Array.from(values.subarray(i, i + 6), v => v.toExponential(4)).join(" "));
  return head.join("\n") + "\n" + body.join("\n");
}
function makeViewer(id) {
  const v = $3Dmol.createViewer(document.getElementById(id), { backgroundColor: "#f1efe9" });
  return v;
}
function drawOrbital(v, d, values, iso, first, opts = {}) {
  const text = cube(d, values);
  if (first || opts.moveAtoms) {
    v.removeAllModels();
    v.addModel(text, "cube");
    v.setStyle({}, { stick: { radius: 0.12, color: "#9a9994" }, sphere: { scale: 0.22 } });
  }
  v.removeAllShapes();
  const vol = new $3Dmol.VolumeData(text, "cube");
  const op = opts.opacity || 0.85;
  v.addIsosurface(vol, { isoval: iso, color: POS, opacity: op });
  v.addIsosurface(vol, { isoval: -iso, color: NEG, opacity: op });
  if (first) { v.zoomTo(); if (opts.rotate) v.rotate(...opts.rotate); if (opts.view) opts.view(v); v.zoom(opts.zoom || 1); }
  v.render();
}

// 水分子を、分子面を正面にして O を上に置く向き
const WATER_VIEW = v => { v.rotate(90, "y"); v.rotate(180, "z"); v.rotate(-90, "x"); };

// スライダーなどの入力を、描画1回分にまとめる
function onInput(el, update) {
  let pending = false;
  el.addEventListener("input", () => {
    if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; update(); }); }
  });
}
