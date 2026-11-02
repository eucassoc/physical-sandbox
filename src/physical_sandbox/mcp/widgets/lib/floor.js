// The floor's drawing, shared by the chat widget and the stage page. Plain script text: widget_html inlines it
// inside the page's module script, so these functions are in scope for the page code that follows. It holds no
// page state: everything comes in as parameters.

/** Metres to centimetres, one decimal. */
function cmOf(m) { return Math.round(m * 1000) / 10; }

/** A grid step as a person reads it: "1 m" or "50 cm". */
function stepLabel(m) { return m >= 1 ? `${m} m` : `${cmOf(m)} cm`; }

/** The faint floor and its grid: lines of photo pixels (a null breaks a line), k = canvas pixels per photo pixel. */
function drawGrid(ctx, k, lines) {
  if (!lines) return;
  ctx.strokeStyle = 'rgba(155, 231, 242, 0.75)';
  ctx.lineWidth = 1.5;
  for (const line of lines) {
    const pts = line.filter(Boolean);
    if (pts.length < 2) continue;
    ctx.beginPath();
    pts.forEach(([u, v], i) => (i ? ctx.lineTo(u * k, v * k) : ctx.moveTo(u * k, v * k)));
    ctx.stroke();
  }
}

/** The floor's four corners: the orange fill, then the numbered orange squares. corners are {u, v} photo pixels. */
function drawCorners(ctx, k, corners) {
  ctx.fillStyle = 'rgba(255, 180, 84, 0.18)';
  ctx.beginPath();
  corners.forEach((c, i) => (i ? ctx.lineTo(c.u * k, c.v * k) : ctx.moveTo(c.u * k, c.v * k)));
  ctx.closePath();
  ctx.fill();
  corners.forEach((c, i) => {
    const x = c.u * k, y = c.v * k, s = 13;
    ctx.fillStyle = 'rgba(214, 118, 20, 0.85)';
    ctx.strokeStyle = '#ffb454';
    ctx.lineWidth = 2;
    ctx.fillRect(x - s, y - s, 2 * s, 2 * s);
    ctx.strokeRect(x - s, y - s, 2 * s, 2 * s);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 15px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), x, y + 1);
  });
}

/** The index of the first corner within `reach` photo pixels of p ({u, v}), or -1. */
function hitCorner(corners, p, reach) {
  return corners.findIndex((c) => Math.hypot(c.u - p.u, c.v - p.v) < reach);
}
