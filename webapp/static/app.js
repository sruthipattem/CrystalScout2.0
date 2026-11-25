/* CrystalScout UI behaviour: reveal-on-scroll, count-up numbers, menus,
   tabs, drag-and-drop uploads, compare tray, loading overlay. No libraries. */
(function () {
  document.documentElement.classList.add('js');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // reveal on scroll
  const io = 'IntersectionObserver' in window ? new IntersectionObserver((es) => {
    es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
  }, { threshold: 0.08 }) : null;
  document.querySelectorAll('.reveal').forEach((el) => io ? io.observe(el) : el.classList.add('in'));

  // count-up numbers: <b data-count="505" data-dec="0">
  function countUp(el) {
    const end = parseFloat(el.dataset.count); const dec = parseInt(el.dataset.dec || '0', 10);
    const fmt = (v) => v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec });
    if (reduce || !isFinite(end)) { el.textContent = fmt(end); return; }
    const t0 = performance.now(); const dur = 1200;
    (function step(t) {
      const k = Math.min(1, (t - t0) / dur); const e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(end * e); if (k < 1) requestAnimationFrame(step);
    })(t0);
  }
  const cio = 'IntersectionObserver' in window ? new IntersectionObserver((es) => {
    es.forEach((e) => { if (e.isIntersecting) { countUp(e.target); cio.unobserve(e.target); } });
  }) : null;
  document.querySelectorAll('[data-count]').forEach((el) => cio ? cio.observe(el) : countUp(el));

  // user menu
  const ub = document.getElementById('userBtn'); const um = document.getElementById('userMenu');
  if (ub && um) {
    ub.addEventListener('click', (e) => { e.stopPropagation(); um.classList.toggle('open'); ub.setAttribute('aria-expanded', um.classList.contains('open')); });
    document.addEventListener('click', (e) => { if (!um.contains(e.target)) { um.classList.remove('open'); ub.setAttribute('aria-expanded', 'false'); } });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') um.classList.remove('open'); });
  }
  // mobile side nav
  const mt = document.getElementById('menuToggle'); const side = document.querySelector('.side');
  if (mt && side) {
    mt.addEventListener('click', (e) => { e.stopPropagation(); side.classList.toggle('open'); });
    document.addEventListener('click', (e) => { if (side.classList.contains('open') && !side.contains(e.target)) side.classList.remove('open'); });
  }
  // "/" focuses the quick estimate box
  const qi = document.getElementById('quickInput');
  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && qi && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { e.preventDefault(); qi.focus(); }
  });

  // tabs: <div class="tabs" data-tabs><button data-tab="a">..</button></div> + <div class="tabpanel" id="tab-a">
  document.querySelectorAll('[data-tabs]').forEach((bar) => {
    const btns = bar.querySelectorAll('button[data-tab]');
    function show(name) {
      btns.forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === name ? 'true' : 'false'));
      btns.forEach((b) => { const p = document.getElementById('tab-' + b.dataset.tab); if (p) p.hidden = b.dataset.tab !== name; });
      try { history.replaceState(null, '', '#' + name); } catch (e) { /* ignore */ }
    }
    btns.forEach((b) => b.addEventListener('click', () => show(b.dataset.tab)));
    const want = location.hash.slice(1); const init = bar.dataset.initial;
    show([...btns].some((b) => b.dataset.tab === want) ? want : (init || btns[0].dataset.tab));
  });

  // drop zones show the chosen file names
  document.querySelectorAll('.drop').forEach((z) => {
    const inp = z.querySelector('input[type=file]'); const out = z.querySelector('.files');
    if (!inp) return;
    const upd = () => { if (out) out.textContent = [...inp.files].map((f) => f.name).join(', '); };
    inp.addEventListener('change', upd);
    ['dragenter', 'dragover'].forEach((ev) => z.addEventListener(ev, (e) => { e.preventDefault(); z.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => z.addEventListener(ev, () => z.classList.remove('over')));
    z.addEventListener('drop', (e) => { e.preventDefault(); if (e.dataTransfer.files.length) { inp.files = e.dataTransfer.files; upd(); } });
  });

  // loading overlay for slow forms: <form data-loading="Scoring...">
  const ov = document.getElementById('loading');
  let stepTimer = null;
  const DEFAULT_STEPS = 'Reading your input|Checking Materials Project and OBELiX|Building the crystal structure|Computing structure descriptors|Running 500 decision trees|Preparing your results';
  function showLoading(title, steps) {
    if (!ov) return;
    document.getElementById('loadTitle').textContent = title || 'Working…';
    const list = (steps || DEFAULT_STEPS).split('|');
    const ol = document.getElementById('loadSteps');
    ol.innerHTML = list.map((t) => `<li>${t}</li>`).join('');
    const items = [...ol.children]; let k = 0;
    const tick = () => {
      items.forEach((li, i) => { li.className = i < k ? 'done' : (i === k ? 'now' : ''); });
      if (k < items.length - 1) k += 1;
    };
    tick(); clearInterval(stepTimer); stepTimer = setInterval(tick, 650);
    ov.classList.add('on');
  }
  window.CSLoading = showLoading;
  document.querySelectorAll('form[data-loading]').forEach((f) => f.addEventListener('submit', () => {
    if (f.checkValidity && !f.checkValidity()) return;
    showLoading(f.dataset.loading, f.dataset.steps);
  }));
  document.querySelectorAll('a[data-loading]').forEach((a) => a.addEventListener('click', (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    showLoading(a.dataset.loading, a.dataset.steps);
  }));
  window.addEventListener('pageshow', () => { if (ov) ov.classList.remove('on'); clearInterval(stepTimer); });

  // compare tray: checkboxes .cmp[value=mp-id]
  const tray = document.getElementById('tray');
  if (tray) {
    const boxes = document.querySelectorAll('input.cmp'); const cnt = tray.querySelector('b'); const go = tray.querySelector('a');
    const sync = () => {
      const ids = [...boxes].filter((b) => b.checked).map((b) => b.value);
      boxes.forEach((b) => { b.disabled = !b.checked && ids.length >= 4; });
      cnt.textContent = ids.length; tray.classList.toggle('on', ids.length > 0);
      go.href = go.dataset.base + '?ids=' + encodeURIComponent(ids.join(',')) + (go.dataset.run ? '&run=' + go.dataset.run : '');
      go.classList.toggle('disabled', ids.length < 2);
    };
    boxes.forEach((b) => b.addEventListener('change', sync));
    const clr = tray.querySelector('.clear'); if (clr) clr.addEventListener('click', () => { boxes.forEach((b) => { b.checked = false; }); sync(); });
    sync();
  }

  // auto-dismiss flash messages
  document.querySelectorAll('.flash').forEach((f) => setTimeout(() => {
    f.style.transition = 'opacity .5s, transform .5s'; f.style.opacity = '0'; f.style.transform = 'translateY(-6px)';
    setTimeout(() => f.remove(), 520);
  }, 6000));
})();
