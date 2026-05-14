// ═══════ ISL Translator — Starry Night Frontend ═══════

const API = 'http://localhost:5000';
let cameraActive = false;
let pollTimer    = null;

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initSmoothScroll();
  initDots();
  document.getElementById('videoStream').src = `${API}/video_feed`;
});

// ── NAVBAR ──
function initNavbar() {
  const h = document.getElementById('hamburger');
  const m = document.getElementById('mobileNav');
  h.addEventListener('click', () => m.classList.toggle('open'));
  m.querySelectorAll('a').forEach(a => a.addEventListener('click', () => m.classList.remove('open')));
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', function(e) {
      e.preventDefault();
      const t = document.querySelector(this.getAttribute('href'));
      if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

// ── TRICOLOR DOTS ANIMATION ──
function initDots() {
  const canvas = document.getElementById('dotsCanvas');
  const ctx    = canvas.getContext('2d');

  // Indian flag tricolour + navy (Ashoka Chakra)
  const COLORS = [
    { r: 255, g: 153, b:  51, a: 0.55 },   // Saffron  #FF9933
    { r: 255, g: 255, b: 255, a: 0.35 },   // White    #FFFFFF
    { r:  19, g: 136, b:   8, a: 0.55 },   // Green    #138808
    { r:   0, g:   0, b: 128, a: 0.40 },   // Navy     #000080 (chakra)
  ];

  const NUM_DOTS    = 110;
  const CONNECT_DIST = 140;   // px — max distance to draw a connecting line
  const SPEED       = 0.35;   // base speed

  let W, H, dots = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeDot() {
    const c = COLORS[Math.floor(Math.random() * COLORS.length)];
    const angle = Math.random() * Math.PI * 2;
    const speed = SPEED * (0.4 + Math.random() * 0.6);
    return {
      x:  Math.random() * W,
      y:  Math.random() * H,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      r:  1.2 + Math.random() * 2,       // dot radius
      c,
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 0.012 + Math.random() * 0.016,
    };
  }

  function init() {
    resize();
    dots = Array.from({ length: NUM_DOTS }, makeDot);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Update positions
    dots.forEach(d => {
      d.x  += d.vx;
      d.y  += d.vy;
      d.pulse += d.pulseSpeed;

      // Wrap edges
      if (d.x < -10) d.x = W + 10;
      if (d.x > W + 10) d.x = -10;
      if (d.y < -10) d.y = H + 10;
      if (d.y > H + 10) d.y = -10;
    });

    // Draw connecting lines between nearby dots
    for (let i = 0; i < dots.length; i++) {
      for (let j = i + 1; j < dots.length; j++) {
        const a = dots[i], b = dots[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECT_DIST) {
          const fade = 1 - dist / CONNECT_DIST;
          // Blend the two dot colours for the line
          const cr = Math.round((a.c.r + b.c.r) / 2);
          const cg = Math.round((a.c.g + b.c.g) / 2);
          const cb = Math.round((a.c.b + b.c.b) / 2);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(${cr},${cg},${cb},${fade * 0.18})`;
          ctx.lineWidth   = 0.75;
          ctx.stroke();
        }
      }
    }

    // Draw dots
    dots.forEach(d => {
      const pulse  = Math.sin(d.pulse);
      const radius = d.r * (1 + pulse * 0.25);
      const alpha  = d.c.a * (0.75 + pulse * 0.25);
      const { r, g, b } = d.c;

      // Soft glow
      const grd = ctx.createRadialGradient(d.x, d.y, 0, d.x, d.y, radius * 4);
      grd.addColorStop(0,   `rgba(${r},${g},${b},${alpha * 0.5})`);
      grd.addColorStop(0.4, `rgba(${r},${g},${b},${alpha * 0.15})`);
      grd.addColorStop(1,   `rgba(${r},${g},${b},0)`);
      ctx.beginPath();
      ctx.arc(d.x, d.y, radius * 4, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();

      // Core dot
      ctx.beginPath();
      ctx.arc(d.x, d.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  init();
  draw();
  window.addEventListener('resize', () => { resize(); });
}

// ── CAMERA TOGGLE ──
async function toggleCamera() {
  if (cameraActive) {
    await sendControl('stop_camera');
    stopUI();
  } else {
    await sendControl('start_camera');
    startUI();
  }
}

function startUI() {
  cameraActive = true;
  const btn      = document.getElementById('btnCam');
  const camLabel = document.getElementById('camLabel');
  const camIcon  = document.getElementById('camIcon');
  const overlay  = document.getElementById('feedOverlayTop');

  btn.classList.add('active');
  camLabel.textContent = 'Stop Camera';
  camIcon.innerHTML    = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
  overlay.style.display = 'flex';

  pollTimer = setInterval(pollStatus, 300);
}

function stopUI() {
  cameraActive = false;
  clearInterval(pollTimer); pollTimer = null;

  const btn      = document.getElementById('btnCam');
  const camLabel = document.getElementById('camLabel');
  const camIcon  = document.getElementById('camIcon');
  const overlay  = document.getElementById('feedOverlayTop');

  btn.classList.remove('active');
  camLabel.textContent = 'Start Camera';
  camIcon.innerHTML    = '<path d="M8 5v14l11-7z"/>';
  overlay.style.display = 'none';

  setDetected('—', null);
  setBuildingText('—');
  showFinal('');
  updateBars(0, 0);
}

// ── POLL STATUS ──
async function pollStatus() {
  try {
    const res = await fetch(`${API}/status`);
    if (!res.ok) return;
    updateUI(await res.json());
  } catch (_) {}
}

// ── UPDATE UI ──
function updateUI(d) {
  if (!d.hands) {
    setDetected('—', null);
    setMeta('');
  } else if (d.current_word) {
    setDetected(d.current_word, d.current_conf);
    if (d.top_vote_word && d.top_vote_count > 0) {
      setMeta(`Vote: ${d.top_vote_word} (${d.top_vote_count} / 6)`);
    } else {
      setMeta('');
    }
  } else {
    setDetected('...', null);
    setMeta('');
  }

  updateBars(d.buf_pct, d.vote_pct);

  setBuildingText(
    d.sentence_words && d.sentence_words.length
      ? d.sentence_words.join(' ')
      : '—'
  );

  showFinal(d.completed_sentence || '');

  const sel = document.getElementById('langSelect');
  if (sel && d.lang_code) sel.value = d.lang_code;
}

function setDetected(word, conf) {
  const el = document.getElementById('detectedWord');
  if (el) el.textContent = word;
  const badge = document.getElementById('signMeta');
  if (badge) badge.textContent = conf !== null && conf !== undefined ? `${conf}% confidence` : '';
}
function setMeta(text) {
  const el = document.getElementById('signMeta');
  if (el) el.textContent = text;
}
function setBuildingText(text) {
  const el = document.getElementById('buildingText');
  if (el) el.textContent = text;
}
function showFinal(text) {
  const card = document.getElementById('finalCard');
  const el   = document.getElementById('finalText');
  if (!card || !el) return;
  if (text) {
    el.textContent      = text;
    card.style.display  = 'block';
  } else {
    card.style.display  = 'none';
    el.textContent      = '';
  }
}
function updateBars(bufPct, votePct) {
  const b = document.getElementById('bufferBar');
  const v = document.getElementById('voteBar');
  if (b) b.style.width = `${Math.min(bufPct  * 100, 100)}%`;
  if (v) v.style.width = `${Math.min(votePct * 100, 100)}%`;
}

// ── CONTROLS ──
async function sendControl(action, extra = {}) {
  try {
    const res = await fetch(`${API}/control`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ action, ...extra }),
    });
    if (!res.ok) return;
    const d = await res.json();
    if (d.sentence_words !== undefined)
      setBuildingText(d.sentence_words.length ? d.sentence_words.join(' ') : '—');
    if (d.completed_sentence !== undefined)
      showFinal(d.completed_sentence);
    const sel = document.getElementById('langSelect');
    if (sel && d.lang_code) sel.value = d.lang_code;
  } catch (e) { console.error(e); }
}

function finalizeSentence() { sendControl('finalize'); }
function clearSentence()    { sendControl('clear'); setBuildingText('—'); showFinal(''); }
function undoWord()         { sendControl('undo'); }
function speakOutput()      { sendControl('speak'); }
function changeLanguage() {
  const sel = document.getElementById('langSelect');
  if (sel) sendControl('language', { lang: sel.value });
}
