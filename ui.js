/* Small shared UI helpers used by every page. */
const UI = (function () {

  const TABS = [
    ['index.html', 'ارسم', 'Draw'],
    ['upload.html', 'ارفع صورة', 'Upload'],
    ['inspect.html', 'كيف يرى النموذج', 'Inspect'],
    ['challenge.html', 'تحدي', 'Challenge']
  ];

  function nav(active) {
    const el = document.getElementById('nav');
    el.className = 'top';
    el.innerHTML = '<div class="top-in"><div class="logo">✎ Digits</div><nav class="tabs">' +
      TABS.map(function (t) {
        return '<a class="tab' + (t[0] === active ? ' on' : '') + '" href="' + t[0] + '">' + t[1] + '<small>' + t[2] + '</small></a>';
      }).join('') + '</nav></div>';
  }

  // Drawing canvas (white strokes on black). onChange is optional.
  function createPad(canvas, onChange) {
    const ctx = canvas.getContext('2d');
    let drawing = false, last = null;
    function reset() { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height); }
    function style() { ctx.lineWidth = 20; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; }
    function pos(e) {
      const r = canvas.getBoundingClientRect();
      return { x: (e.clientX - r.left) * canvas.width / r.width, y: (e.clientY - r.top) * canvas.height / r.height };
    }
    function changed() { if (onChange) onChange(); }
    canvas.addEventListener('pointerdown', function (e) {
      drawing = true; canvas.setPointerCapture(e.pointerId); last = pos(e); style();
      ctx.beginPath(); ctx.arc(last.x, last.y, 10, 0, Math.PI * 2); ctx.fill(); changed(); e.preventDefault();
    });
    canvas.addEventListener('pointermove', function (e) {
      if (!drawing) return;
      const p = pos(e); style(); ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke();
      last = p; changed(); e.preventDefault();
    });
    const stop = function () { drawing = false; };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    reset();
    return {
      clear: function () { reset(); changed(); },
      gray: function () {
        const w = canvas.width, h = canvas.height, d = ctx.getImageData(0, 0, w, h).data;
        const g = new Float32Array(w * h);
        for (let i = 0; i < g.length; i++) g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
        return { gray: g, w: w, h: h };
      }
    };
  }

  // 10 probability bars inside `el`.
  function buildBars(el) {
    el.innerHTML = '';
    const rows = [], fills = [], pcts = [];
    for (let d = 0; d < 10; d++) {
      const r = document.createElement('div'); r.className = 'bar';
      r.innerHTML = '<b>' + d + '</b><div class="track"><div class="fill"></div></div><span class="pct">0.0%</span>';
      el.appendChild(r); rows.push(r); fills.push(r.querySelector('.fill')); pcts.push(r.querySelector('.pct'));
    }
    return {
      reset: function () {
        for (let d = 0; d < 10; d++) { fills[d].style.width = '0%'; pcts[d].textContent = '0.0%'; rows[d].classList.remove('top'); }
      },
      set: function (p) {
        let best = 0; for (let k = 1; k < 10; k++) if (p[k] > p[best]) best = k;
        for (let k = 0; k < 10; k++) {
          fills[k].style.width = (p[k] * 100).toFixed(1) + '%';
          pcts[k].textContent = (p[k] * 100).toFixed(1) + '%';
          rows[k].classList.toggle('top', k === best);
        }
        return best;
      }
    };
  }

  // Big predicted digit + confidence + bars.
  function result(digitEl, confEl, bars) {
    return {
      empty: function () {
        digitEl.textContent = '?'; digitEl.classList.add('none');
        if (confEl) confEl.textContent = '—';
        bars.reset();
      },
      show: function (p) {
        const best = bars.set(p);
        digitEl.textContent = best; digitEl.classList.remove('none');
        if (confEl) confEl.textContent = 'الثقة · Confidence ' + (p[best] * 100).toFixed(1) + '%';
        return best;
      }
    };
  }

  // Draw grayscale values (0..255) on a canvas of w x h; maxPx = size of the longest side on screen.
  function drawGray(canvas, data, w, h, maxPx) {
    canvas.width = w; canvas.height = h;
    const c = canvas.getContext('2d'), id = c.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const v = data[i];
      id.data[i * 4] = v; id.data[i * 4 + 1] = v; id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255;
    }
    c.putImageData(id, 0, 0);
    if (maxPx) {
      const s = maxPx / Math.max(w, h);
      canvas.style.width = Math.round(w * s) + 'px';
      canvas.style.height = Math.round(h * s) + 'px';
    }
  }

  function loadModel(statusEl) {
    return MODEL.load().catch(function (e) {
      statusEl.textContent = 'تعذر تحميل weights.json — شغّل الموقع عبر سيرفر (http) وليس بفتح الملف مباشرة · Serve the site over HTTP.';
      statusEl.style.display = 'block';
      throw e;
    });
  }

  return { nav: nav, createPad: createPad, buildBars: buildBars, result: result, drawGray: drawGray, loadModel: loadModel };
})();
