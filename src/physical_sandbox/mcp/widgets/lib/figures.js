// The numbers of a comparison, as the table rows and the site line the screens show. Plain script text, inlined
// inside the page's module script; it holds no page state.

/** Metres as millimetres. */
function mm(m) { return `${(m * 1000).toFixed(1)} mm`; }
/** Radians as degrees. */
function deg(r) { return `${(r * 180 / Math.PI).toFixed(1)}°`; }
/** Seconds, two decimals. */
function sec(s) { return `${s.toFixed(2)} s`; }

/** The table's `<tr>` rows. `when(t)` makes the HTML for a time that jumps the slider there, `esc` escapes text.
 *  A row is written only when its figures are there. */
function figureRows(figures, when, esc) {
  const f = figures;
  const dash = (v, fmt) => (v === null || v === undefined ? '—' : fmt(v));
  // What the numbers cannot tell yet, in words a person reads (the server's names otherwise)
  const UNKNOWN_NAMES = { clearance: '周りとのすき間', reach: '届いたか', world: '世界の食い違い', grasps: 'つかむ瞬間' };
  const unknown = f.unknown ?? [];
  const rows = [];
  const row = (name, value) => rows.push(`<tr><th>${esc(name)}</th><td>${value}</td></tr>`);
  if (f.tool_deviation_m) {
    row('手先のずれ（同じ時刻）', `最大 ${mm(f.tool_deviation_m.max)}（${when(f.tool_deviation_m.at_s)}）、平均 ${mm(f.tool_deviation_m.mean)}`);
  }
  if (f.shape_deviation_m !== undefined) row('通り道の形のずれ', mm(f.shape_deviation_m));
  const d = f.duration_s;
  if (d) row('動いた時間', d.b === undefined ? sec(d.a) : `A ${sec(d.a)}、B ${sec(d.b)}（B − A = ${d.diff >= 0 ? '+' : ''}${d.diff.toFixed(2)} s）`);
  if (f.jerk) row('なめらかさ（ジャーク）', `A ${dash(f.jerk.a, (v) => `${v.toPrecision(3)} m/s³`)}、B ${dash(f.jerk.b, (v) => `${v.toPrecision(3)} m/s³`)}`);
  for (const [name, j] of Object.entries(f.joints ?? {})) {
    row(`関節 ${name} のずれ`, `最大 ${deg(j.max_rad)}（${when(j.at_s)}）、平均 ${deg(j.mean_rad)}`);
  }
  const n = f.grasp_counts;
  if (n && !unknown.includes('grasps')) {
    const at = f.grip_threshold === null || f.grip_threshold === undefined ? '' : `（しきい値 ${Number(f.grip_threshold).toPrecision(3)}）`;
    row('つかんだ回数', n.b === undefined ? `${n.a} 回${at}` : `A ${n.a} 回、B ${n.b} 回${at}`);
    (f.grasps ?? []).forEach((g, i) => row(`${i + 1} 回目につかんだとき`, `手先のずれ ${mm(g.tool_gap_m)}（A の ${when(g.a_s)}、B の ${sec(g.b_s)}）`));
  }
  for (const name of unknown) {
    row(`${UNKNOWN_NAMES[name] ?? name}（まだ分かりません）`, `<span class="why">${esc((f.unknown_reasons ?? {})[name] ?? '')}</span>`);
  }
  return rows.join('');
}

/** The line under the picture: the site's name, its fit, its sizes and whether a person confirmed it. */
function siteLine(site, stride, esc) {
  const px = Object.entries(site.residualPx).map(([cam, v]) => `${esc(cam)} ${v.toFixed(1)} px`).join('、');
  return `場所 ${esc(site.site)}　ずれ ${px || '—'}　寸法 ${site.dimensions === 'measured' ? '実測' : '推定'}　` +
    `${site.confirmedByPerson ? '人が確認済み' : '人の確認はまだ'}` +
    (site.dimensionsNote ? `　<span class="note">${esc(site.dimensionsNote)}</span>` : '') +
    (stride > 1 ? `　<span class="why">コマは ${stride} 個おきに送っています</span>` : '');
}
