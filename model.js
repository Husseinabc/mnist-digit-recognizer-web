/* The whole model in plain JavaScript: preprocessing + Flatten -> Dense(64, ReLU) -> Dense(10, Softmax). */
const MODEL = (function () {

  function load() {
    return fetch('weights.json').then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  }

  // gray: array of w*h values 0..255 (bright = ink).
  // Returns Float32Array(784) in 0..255 (white digit on black), or null if nothing is drawn.
  // If `debug` is an object it is filled with the intermediate stages.
  function preprocess(gray, w, h, debug) {
    const n = w * h;
    const sorted = Float32Array.from(gray).sort();
    const med = (sorted[(n >> 1) - 1] + sorted[n >> 1]) / 2;
    let max = 0;
    const ink = new Float32Array(n);
    for (let i = 0; i < n; i++) { const v = Math.max(gray[i] - med, 0); ink[i] = v; if (v > max) max = v; }
    if (max < 30) return null;
    for (let i = 0; i < n; i++) ink[i] = ink[i] / max * 255;

    let x0 = w, x1 = -1, y0 = h, y1 = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (ink[y * w + x] > 25.5) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
    if (x1 < 0) return null;
    const cw = x1 - x0 + 1, ch = y1 - y0 + 1;

    const scale = 20 / Math.max(cw, ch);
    const nw = Math.max(1, Math.round(cw * scale)), nh = Math.max(1, Math.round(ch * scale));

    const small = new Float32Array(nw * nh);
    const sx = cw / nw, sy = ch / nh;
    for (let j = 0; j < nh; j++) for (let i = 0; i < nw; i++) {
      const xa = i * sx, xb = (i + 1) * sx, ya = j * sy, yb = (j + 1) * sy;
      let sum = 0, wsum = 0;
      for (let yy = Math.floor(ya); yy < Math.ceil(yb); yy++) {
        const wy = Math.min(yy + 1, yb) - Math.max(yy, ya);
        for (let xx = Math.floor(xa); xx < Math.ceil(xb); xx++) {
          const wx = Math.min(xx + 1, xb) - Math.max(xx, xa);
          sum += ink[(y0 + Math.min(yy, ch - 1)) * w + x0 + Math.min(xx, cw - 1)] * wx * wy;
          wsum += wx * wy;
        }
      }
      small[j * nw + i] = sum / wsum;
    }

    if (debug) {
      const crop = new Float32Array(cw * ch);
      for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = ink[(y0 + y) * w + x0 + x];
      debug.box = { x0: x0, y0: y0, x1: x1, y1: y1 };
      debug.crop = { data: crop, w: cw, h: ch };
      debug.small = { data: small, w: nw, h: nh };
    }

    let img = new Float32Array(784);
    const top = (28 - nh) >> 1, left = (28 - nw) >> 1;
    for (let j = 0; j < nh; j++) for (let i = 0; i < nw; i++) img[(top + j) * 28 + left + i] = small[j * nw + i];

    let total = 0, cy = 0, cx = 0;
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) { const v = img[y * 28 + x]; total += v; cy += v * y; cx += v * x; }
    if (total > 0) {
      cy /= total; cx /= total;
      const dy = Math.round(14 - cy), dx = Math.round(14 - cx);
      const out = new Float32Array(784);
      for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) out[ny * 28 + nx] = img[y * 28 + x];
      }
      img = out;
    }
    for (let i = 0; i < 784; i++) img[i] = Math.min(255, Math.max(0, Math.round(img[i])));
    return img;
  }

  // x: 784 values 0..255. W: {W1,b1,W2,b2} flat row-major. Returns {p: probabilities, hidden: ReLU activations}.
  function forward(x, W) {
    const H = W.b1.length, K = W.b2.length;
    const h = new Float32Array(H);
    for (let j = 0; j < H; j++) h[j] = W.b1[j];
    for (let i = 0; i < 784; i++) {
      const v = x[i] / 255;
      if (v === 0) continue;
      const row = i * H;
      for (let j = 0; j < H; j++) h[j] += v * W.W1[row + j];
    }
    for (let j = 0; j < H; j++) if (h[j] < 0) h[j] = 0;
    const z = new Float32Array(K);
    for (let k = 0; k < K; k++) z[k] = W.b2[k];
    for (let j = 0; j < H; j++) for (let k = 0; k < K; k++) z[k] += h[j] * W.W2[j * K + k];
    let m = -Infinity; for (let k = 0; k < K; k++) if (z[k] > m) m = z[k];
    let s = 0; const p = new Float32Array(K);
    for (let k = 0; k < K; k++) { p[k] = Math.exp(z[k] - m); s += p[k]; }
    for (let k = 0; k < K; k++) p[k] /= s;
    return { p: p, hidden: h };
  }

  function predict(x, W) { return forward(x, W).p; }

  return { load: load, preprocess: preprocess, forward: forward, predict: predict };
})();

if (typeof module !== 'undefined') module.exports = MODEL;
