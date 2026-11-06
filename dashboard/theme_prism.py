"""
dashboard/theme_prism.py
========================
NeuroTwin "Prism" visual layer: bright English colours + rich motion.

Adds (on top of the Aurora theme, never replacing its logic):
  * animated aurora blobs + dot-grid + scan-line background
  * neon rotating-border cards, shimmering headlines, staggered entrance
  * glowing buttons, sliding tabs, glass sidebar, themed header
  * a live "Hologram Twin" (canvas): rotating particle body, scanning ring,
    pulsing heart, orbiting Empatica E4 sensor chips, scrolling ECG
  * scrolling facts ticker + animated Measured -> Estimated -> Reported flow
  * cursor glow, 3D card tilt and a floating neural-particle layer (JS)

Scientific-honesty rules are preserved: the hologram is ILLUSTRATIVE and its
colours never encode a stress level; Calm Mode switches all motion off.
Everything is local (no CDN, no external fonts).
"""
from __future__ import annotations

import json
from string import Template

import streamlit as st
import streamlit.components.v1 as components

from dashboard.theme_aurora import compact_html

# Bright English colour palette
CYAN = "#00E5FF"
PINK = "#FF2D95"
LIME = "#B8FF2C"
SUN = "#FFE600"
ORANGE = "#FF8A00"
VIOLET = "#8B5CF6"
BLUE = "#2F6BFF"

_CSS = Template(r"""
@property --nt-a { syntax: '<angle>'; inherits: false; initial-value: 0deg; }
:root {
  --p-cyan:#00E5FF; --p-pink:#FF2D95; --p-lime:#B8FF2C; --p-sun:#FFE600;
  --p-orange:#FF8A00; --p-violet:#8B5CF6; --p-blue:#2F6BFF;
  --p-text:$text; --p-muted:$muted;
  --p-card:rgba(12,18,46,.82);
  --p-rainbow:linear-gradient(90deg,#00E5FF,#2F6BFF,#8B5CF6,#FF2D95,#FF8A00,#FFE600,#B8FF2C,#00E5FF);
}

/* ---------- page + animated background ---------- */
.stApp { background:$app_bg !important; color:var(--p-text) !important;
  font-family:"Avenir Next","Segoe UI",system-ui,-apple-system,Roboto,sans-serif !important; }
.stApp::before { content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  opacity:$blob_opacity !important;
  background:
    radial-gradient(circle at 20% 30%, rgba(0,229,255,.55) 0, transparent 30%),
    radial-gradient(circle at 80% 25%, rgba(255,45,149,.50) 0, transparent 30%),
    radial-gradient(circle at 60% 80%, rgba(139,92,246,.55) 0, transparent 32%),
    radial-gradient(circle at 15% 85%, rgba(255,230,0,.35) 0, transparent 26%);
  background-size:120% 120% !important; animation:pBlobs 26s ease-in-out infinite alternate !important; }
@keyframes pBlobs { 0%{transform:translate3d(-4%,-3%,0) rotate(0deg) scale(1)}
  50%{transform:translate3d(5%,4%,0) rotate(8deg) scale(1.12)} 100%{transform:translate3d(-2%,6%,0) rotate(-6deg) scale(1.05)} }
.stApp::after { content:""; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:$grid_opacity;
  background-image:radial-gradient(circle,currentColor 1px,transparent 1.4px); background-size:34px 34px; color:$dot;
  -webkit-mask-image:linear-gradient(180deg,transparent,#000 30%,#000 70%,transparent);
          mask-image:linear-gradient(180deg,transparent,#000 30%,#000 70%,transparent);
  animation:pGrid 40s linear infinite; }
@keyframes pGrid { to { background-position:340px 340px; } }
[data-testid="stAppViewContainer"], [data-testid="stMain"], .block-container { position:relative; z-index:1; }
.block-container { padding-top:3.2rem !important; max-width:1280px; }

/* ---------- header, rainbow progress line, hide auto page nav ---------- */
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stHeader"] * { color:var(--p-text) !important; }
[data-testid="stHeader"]::after { content:""; position:absolute; left:0; right:0; bottom:0; height:3px;
  background:var(--p-rainbow); background-size:200% 100%; animation:pSlide 6s linear infinite;
  box-shadow:0 0 14px rgba(0,229,255,.8); }
@keyframes pSlide { to { background-position:200% 0; } }
[data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"] { display:none !important; }

/* ---------- headings ---------- */
h1, .prism-title { font-weight:900 !important; letter-spacing:-.02em; background:var(--p-rainbow);
  background-size:250% 100%; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; animation:pSlide 7s linear infinite; }
h2, h3 { color:var(--p-text) !important; position:relative; }
h2::after { content:""; display:block; width:64px; height:4px; margin-top:.35rem; border-radius:4px;
  background:var(--p-rainbow); background-size:200% 100%; animation:pSlide 5s linear infinite; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:var(--p-text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:var(--p-muted) !important; }
[data-testid="stWidgetLabel"] *, .stRadio label, .stCheckbox label { color:var(--p-text); }

/* ---------- entrance animation (staggered) ---------- */
[data-testid="stMain"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
[data-testid="stMain"] [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
  animation:pRise .7s cubic-bezier(.2,.8,.2,1) both; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] > :nth-child(2){animation-delay:.05s}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > :nth-child(3){animation-delay:.1s}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > :nth-child(4){animation-delay:.15s}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > :nth-child(5){animation-delay:.2s}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > :nth-child(6){animation-delay:.25s}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > :nth-child(7){animation-delay:.3s}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > :nth-child(n+8){animation-delay:.35s}
@keyframes pRise { from{opacity:0; transform:translateY(22px) scale(.985); filter:blur(4px)} to{opacity:1; transform:none; filter:none} }

/* ---------- neon rotating-border cards (always dark glass so text stays readable) ---------- */
.nt-card, .metric-card, .glass, .diag-box, .prism-flow-node, [data-testid="stMetric"] {
  position:relative; color:#EAF1FF !important; border:1.5px solid transparent !important; border-radius:18px !important;
  background:linear-gradient(var(--p-card),var(--p-card)) padding-box,
    conic-gradient(from var(--nt-a),#00E5FF,#8B5CF6,#FF2D95,#FFE600,#B8FF2C,#00E5FF) border-box !important;
  animation:pSpin 9s linear infinite; transition:transform .25s ease, box-shadow .25s ease;
  box-shadow:0 10px 34px rgba(0,0,0,.25); transform-style:preserve-3d; will-change:transform; }
@keyframes pSpin { to { --nt-a:360deg; } }
.nt-card:hover, .metric-card:hover, [data-testid="stMetric"]:hover { box-shadow:0 14px 44px rgba(0,229,255,.35), 0 0 0 1px rgba(255,45,149,.35); }
[data-testid="stMetric"] { padding:.9rem 1rem; }
[data-testid="stMetric"] * { color:#EAF1FF !important; }
[data-testid="stMetricValue"] { font-weight:800 !important; text-shadow:0 0 16px rgba(0,229,255,.7); }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
  position:relative; overflow:hidden; border-radius:14px !important; font-weight:800 !important; letter-spacing:.02em;
  border:1.5px solid rgba(0,229,255,.7) !important; color:#fff !important;
  background:linear-gradient(135deg,rgba(47,107,255,.85),rgba(139,92,246,.85)) !important;
  transition:transform .2s ease, box-shadow .2s ease; box-shadow:0 6px 22px rgba(47,107,255,.45); }
[data-testid="stBaseButton-primary"] { background:linear-gradient(135deg,#FF2D95,#FF8A00) !important; border-color:#FFE600 !important;
  box-shadow:0 6px 24px rgba(255,45,149,.55); animation:pGlow 2.4s ease-in-out infinite; }
@keyframes pGlow { 50% { box-shadow:0 8px 34px rgba(255,138,0,.85), 0 0 0 5px rgba(255,230,0,.18); } }
.stButton > button *, .stDownloadButton > button * { color:#fff !important; }
.stButton > button::after, .stDownloadButton > button::after { content:""; position:absolute; top:0; left:-120%; width:60%; height:100%;
  background:linear-gradient(100deg,transparent,rgba(255,255,255,.55),transparent); transform:skewX(-20deg); transition:left .6s ease; }
.stButton > button:hover::after, .stDownloadButton > button:hover::after { left:140%; }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-3px) scale(1.03); box-shadow:0 12px 30px rgba(0,229,255,.55); }
.stButton > button:active { transform:scale(.96); }

/* ---------- sidebar (dark glass in both themes) ---------- */
[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(7,10,32,.96),rgba(18,10,48,.96)) !important;
  border-right:1px solid rgba(0,229,255,.35); box-shadow:6px 0 30px rgba(0,229,255,.12); }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] label { color:#EAF1FF; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.28rem .6rem; border-radius:12px; transition:all .25s ease; border:1px solid transparent; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(0,229,255,.14); transform:translateX(6px); border-color:rgba(0,229,255,.4); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:linear-gradient(90deg,rgba(255,45,149,.35),rgba(0,229,255,.2));
  border-color:rgba(255,230,0,.7); box-shadow:0 0 18px rgba(255,45,149,.45); font-weight:800; }

/* ---------- tabs, expanders, alerts, uploader, charts ---------- */
[data-baseweb="tab-list"] { gap:.4rem; }
[data-baseweb="tab"] { border-radius:12px 12px 0 0; transition:all .25s ease; font-weight:700; }
[data-baseweb="tab"]:hover { background:rgba(0,229,255,.15); transform:translateY(-2px); }
[data-baseweb="tab-highlight"] { background:var(--p-rainbow) !important; background-size:200% 100% !important; height:4px !important; animation:pSlide 4s linear infinite; }
[data-testid="stExpander"] details { border-radius:16px !important; border:1px solid rgba(0,229,255,.4) !important; background:$glass; }
[data-testid="stExpander"] summary:hover { color:var(--p-pink) !important; }
[data-testid="stAlert"] { border-radius:16px; background:$glass !important; border:1px solid rgba(255,230,0,.55); }
[data-testid="stAlert"] * { color:var(--p-text) !important; }
[data-testid="stFileUploaderDropzone"] { border:2px dashed #00E5FF !important; border-radius:18px !important; animation:pDash 3s ease-in-out infinite; }
@keyframes pDash { 50% { border-color:#FF2D95; box-shadow:0 0 26px rgba(255,45,149,.45); } }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background:var(--p-card); border-radius:18px; padding:8px;
  border:1px solid rgba(0,229,255,.35); box-shadow:0 8px 28px rgba(0,0,0,.25); }
[data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg,#00E5FF,#FF2D95,#FFE600,#B8FF2C) !important;
  background-size:200% 100% !important; animation:pSlide 2s linear infinite; }
hr { border:none !important; height:2px !important; background:var(--p-rainbow) !important; background-size:200% 100%; animation:pSlide 8s linear infinite; opacity:.7; }
::-webkit-scrollbar { width:10px; } ::-webkit-scrollbar-thumb { background:linear-gradient(#00E5FF,#FF2D95); border-radius:8px; }

/* ---------- journey ribbon + badges ---------- */
.journey-node.active { background:linear-gradient(90deg,rgba(255,45,149,.35),rgba(0,229,255,.3)) !important; border-radius:12px; box-shadow:0 0 18px rgba(0,229,255,.6); animation:pGlow 2.4s ease-in-out infinite; }
.journey-node.active .journey-dot { background:#FFE600 !important; box-shadow:0 0 12px #FFE600; }
.journey-arrow { animation:pNudge 1.4s ease-in-out infinite; color:#00E5FF !important; }
@keyframes pNudge { 50% { transform:translateX(5px); } }
.aurora-badge { animation:pGlow 3s ease-in-out infinite; }

/* ---------- hero extras ---------- */
.prism-title { font-size:clamp(2.4rem,5vw,4.2rem); line-height:1.02; margin:.2rem 0 .4rem; display:block; }
.prism-sub { font-size:1.25rem; font-weight:700; color:var(--p-text); margin-bottom:.8rem; }
.prism-sub b { color:#FF2D95; }
.prism-ticker { overflow:hidden; border-radius:14px; border:1.5px solid rgba(0,229,255,.55); background:var(--p-card); margin:1rem 0;
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent); mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent); }
.prism-ticker-track { display:inline-flex; gap:2.2rem; white-space:nowrap; padding:.7rem 0; animation:pMarquee 38s linear infinite; }
.prism-ticker:hover .prism-ticker-track { animation-play-state:paused; }
.prism-ticker-track span { color:#EAF1FF; font-weight:800; font-size:.82rem; letter-spacing:.12em; text-transform:uppercase; }
.prism-ticker-track i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.6rem; box-shadow:0 0 10px currentColor; background:currentColor; animation:pBlink 1.6s ease-in-out infinite; }
@keyframes pMarquee { to { transform:translateX(-50%); } }
@keyframes pBlink { 50% { transform:scale(.55); opacity:.5; } }
.prism-flow { display:flex; align-items:stretch; gap:0; margin:1rem 0 1.4rem; flex-wrap:wrap; }
.prism-flow-node { flex:1 1 220px; padding:1.1rem 1.2rem; text-align:left; }
.prism-flow-node b { display:block; font-size:1.05rem; letter-spacing:.1em; margin-bottom:.35rem; }
.prism-flow-node small { color:#C7D3F7; font-size:.8rem; line-height:1.45; display:block; }
.prism-flow-link { flex:0 0 70px; position:relative; align-self:center; height:4px; border-radius:4px; background:rgba(255,255,255,.18); overflow:visible; }
.prism-flow-link::after { content:""; position:absolute; top:-5px; left:0; width:14px; height:14px; border-radius:50%;
  background:#FFE600; box-shadow:0 0 16px #FFE600,0 0 30px #FF2D95; animation:pPacket 1.8s ease-in-out infinite; }
@keyframes pPacket { 0%{left:0; opacity:0} 15%{opacity:1} 85%{opacity:1} 100%{left:calc(100% - 14px); opacity:0} }
.prism-note { font-size:.72rem; color:var(--p-muted); text-align:center; margin-top:.3rem; letter-spacing:.04em; }

/* ---------- contrast guards (sidebar + ribbon stay dark glass in BOTH themes) ---------- */
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color:#EAF1FF !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { background:rgba(12,18,46,.85) !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] { background:rgba(12,18,46,.85) !important; }
.journey-ribbon { background:rgba(12,18,46,.88) !important; border:1.5px solid rgba(0,229,255,.55) !important; border-radius:14px; }
.journey-node, .journey-node * { color:#EAF1FF !important; }
.journey-ribbon + *, .journey-ribbon { box-shadow:0 6px 24px rgba(0,229,255,.25); }

/* ---------- JS-driven extras ---------- */
#nt-cursor-glow { position:fixed; width:460px; height:460px; border-radius:50%; pointer-events:none; z-index:2; left:-999px; top:-999px;
  transform:translate(-50%,-50%); background:radial-gradient(circle,rgba(0,229,255,.20),rgba(255,45,149,.10) 40%,transparent 70%); mix-blend-mode:screen; }
#nt-particles { position:fixed; inset:0; width:100vw; height:100vh; pointer-events:none; z-index:0; opacity:$particle_opacity; }
iframe[title="streamlit_component"], iframe[title="st.iframe"] { color-scheme:normal; }

@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
$calm
""")

_CALM = """
*, *::before, *::after { animation:none !important; transition:none !important; }
#nt-cursor-glow, #nt-particles { display:none !important; }
"""


def generate_prism_css(theme: str = "dark", calm_mode: bool = False) -> str:
    dark = theme == "dark"
    if dark:
        v = dict(
            text="#F2F6FF", muted="#B9C5EE",
            app_bg=("radial-gradient(1200px 700px at 12% -10%,rgba(47,107,255,.30),transparent 60%),"
                    "radial-gradient(900px 600px at 100% 10%,rgba(255,45,149,.20),transparent 60%),"
                    "linear-gradient(180deg,#050716,#0A0F2E 60%,#050716)"),
            blob_opacity="0.30", grid_opacity="0.30", dot="#7FA8FF", glass="rgba(12,18,46,.78)", particle_opacity="0.9",
        )
    else:
        v = dict(
            text="#0E1738", muted="#3A4A7E",
            app_bg=("radial-gradient(1200px 700px at 10% -10%,rgba(0,229,255,.38),transparent 60%),"
                    "radial-gradient(900px 600px at 100% 0%,rgba(255,45,149,.22),transparent 60%),"
                    "radial-gradient(900px 700px at 50% 110%,rgba(255,230,0,.30),transparent 60%),"
                    "linear-gradient(180deg,#F7FAFF,#E8F1FF)"),
            blob_opacity="0.40", grid_opacity="0.35", dot="#2F6BFF", glass="rgba(255,255,255,.80)", particle_opacity="0.65",
        )
    return _CSS.safe_substitute(calm=_CALM if calm_mode else "", **v)


def _embed(html: str, height: int) -> None:
    """Prefer the new st.iframe; fall back to components.html on older Streamlit."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # pragma: no cover
        components.html(html, height=height)


def inject_prism_theme(theme: str = "dark", calm_mode: bool = False) -> None:
    """Inject the Prism stylesheet. Call AFTER the Aurora/legacy themes so it wins."""
    st.markdown(f"<style>{generate_prism_css(theme, calm_mode)}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Parent-page effects: cursor glow, 3D tilt, neural particles
# ---------------------------------------------------------------------------
_FX_JS = r"""
<script>
(function(){
 var CALM = __CALM__;
 try {
  var P = window.parent, D = P.document;
  if (P.__ntPrismCleanup) { P.__ntPrismCleanup(); P.__ntPrismCleanup = null; }
  if (CALM) return;
  var glow = D.createElement('div'); glow.id = 'nt-cursor-glow'; D.body.appendChild(glow);
  var cv = D.createElement('canvas'); cv.id = 'nt-particles'; D.body.appendChild(cv);
  var cx = cv.getContext('2d'), W = 0, H = 0, raf = 0, mx = -999, my = -999;
  var cols = ['#00E5FF','#FF2D95','#FFE600','#B8FF2C','#8B5CF6','#FF8A00'];
  function size(){ W = cv.width = P.innerWidth; H = cv.height = P.innerHeight; }
  size();
  var pts = []; for (var i = 0; i < 70; i++) pts.push({x:Math.random()*W, y:Math.random()*H, vx:(Math.random()-.5)*.35, vy:(Math.random()-.5)*.35, r:1.2+Math.random()*2, c:cols[i%cols.length]});
  function draw(){
    cx.clearRect(0,0,W,H);
    for (var i=0;i<pts.length;i++){ var a=pts[i]; a.x+=a.vx; a.y+=a.vy;
      if(a.x<0||a.x>W)a.vx*=-1; if(a.y<0||a.y>H)a.vy*=-1;
      cx.beginPath(); cx.arc(a.x,a.y,a.r,0,6.283); cx.fillStyle=a.c; cx.shadowColor=a.c; cx.shadowBlur=10; cx.fill(); cx.shadowBlur=0;
      for (var j=i+1;j<pts.length;j++){ var b=pts[j], dx=a.x-b.x, dy=a.y-b.y, d=dx*dx+dy*dy;
        if(d<16900){ cx.strokeStyle=a.c; cx.globalAlpha=(1-d/16900)*.35; cx.beginPath(); cx.moveTo(a.x,a.y); cx.lineTo(b.x,b.y); cx.stroke(); cx.globalAlpha=1; } }
      var mdx=a.x-mx, mdy=a.y-my, md=mdx*mdx+mdy*mdy;
      if(md<22500){ cx.strokeStyle='#fff'; cx.globalAlpha=(1-md/22500)*.6; cx.beginPath(); cx.moveTo(a.x,a.y); cx.lineTo(mx,my); cx.stroke(); cx.globalAlpha=1; a.x+=mdx*.004; a.y+=mdy*.004; }
    }
    raf = P.requestAnimationFrame(draw);
  }
  draw();
  function onMove(e){
    mx = e.clientX; my = e.clientY; glow.style.left = mx+'px'; glow.style.top = my+'px';
    var t = e.target && e.target.closest ? e.target.closest('.nt-card,.metric-card,[data-testid="stMetric"]') : null;
    if (t){ var r = t.getBoundingClientRect(), px=(e.clientX-r.left)/r.width-.5, py=(e.clientY-r.top)/r.height-.5;
      t.style.transform='perspective(800px) rotateX('+(-py*9)+'deg) rotateY('+(px*11)+'deg) translateY(-4px)'; }
  }
  function onOut(e){ var t = e.target && e.target.closest ? e.target.closest('.nt-card,.metric-card,[data-testid="stMetric"]') : null; if (t) t.style.transform=''; }
  D.addEventListener('mousemove', onMove); D.addEventListener('mouseout', onOut); P.addEventListener('resize', size);
  P.__ntPrismCleanup = function(){
    P.cancelAnimationFrame(raf); D.removeEventListener('mousemove', onMove); D.removeEventListener('mouseout', onOut);
    P.removeEventListener('resize', size); glow.remove(); cv.remove();
  };
 } catch (err) { /* sandboxed: CSS-only mode */ }
})();
</script>
"""


def inject_prism_fx(calm_mode: bool = False) -> None:
    """Install (or, in Calm Mode, remove) the cursor-glow / tilt / particle effects."""
    _embed(_FX_JS.replace("__CALM__", "true" if calm_mode else "false"), height=1)


# ---------------------------------------------------------------------------
# Hologram Twin (canvas)
# ---------------------------------------------------------------------------
_HOLO_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;font-family:"Avenir Next","Segoe UI",system-ui,sans-serif}
canvas{width:100%;height:100%;display:block}
</style></head><body><canvas id="c"></canvas><script>
var CFG = __CFG__;
var cv = document.getElementById('c'), ctx = cv.getContext('2d'), W, H, DPR;
function resize(){ DPR=Math.min(2,window.devicePixelRatio||1); W=cv.clientWidth; H=cv.clientHeight; cv.width=W*DPR; cv.height=H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); }
resize(); window.addEventListener('resize', resize);
var STOPS=[[0,[0,229,255]],[.22,[47,107,255]],[.45,[139,92,246]],[.65,[255,45,149]],[.82,[255,138,0]],[1,[255,230,0]]];
function grad(t){ for(var i=1;i<STOPS.length;i++){ if(t<=STOPS[i][0]){ var a=STOPS[i-1],b=STOPS[i],k=(t-a[0])/(b[0]-a[0]); return [a[1][0]+(b[1][0]-a[1][0])*k,a[1][1]+(b[1][1]-a[1][1])*k,a[1][2]+(b[1][2]-a[1][2])*k]; } } return STOPS[STOPS.length-1][1]; }
var P=[];
function rnd(a,b){return a+Math.random()*(b-a);}
function cyl(x0,y0,z0,x1,y1,z1,r,n){ for(var i=0;i<n;i++){ var t=Math.random(),a=Math.random()*6.283; P.push([x0+(x1-x0)*t+Math.cos(a)*r, y0+(y1-y0)*t, z0+(z1-z0)*t+Math.sin(a)*r]); } }
function sph(cx,cy,cz,r,n){ for(var i=0;i<n;i++){ var u=Math.random()*2-1,a=Math.random()*6.283,s=Math.sqrt(1-u*u); P.push([cx+r*s*Math.cos(a),cy+r*u,cz+r*s*Math.sin(a)]); } }
sph(0,.82,0,.13,300); cyl(0,.70,0,0,.63,0,.05,40);
for(var i=0;i<760;i++){ var t=Math.random(),y=.58-.53*t,rx=.22-.06*t,a=Math.random()*6.283; P.push([Math.cos(a)*rx,y,Math.sin(a)*.11]); }
[-1,1].forEach(function(s){ sph(s*.245,.56,0,.07,70); cyl(s*.26,.56,0,s*.34,.28,0,.045,140); cyl(s*.34,.28,0,s*.38,.02,0,.04,120); sph(s*.385,-.02,0,.045,40);
  cyl(s*.09,.05,0,s*.11,-.38,0,.075,220); cyl(s*.11,-.38,0,s*.10,-.85,0,.055,170); sph(s*.10,-.9,.05,.05,50); });
var mxT=0, myT=0, mx=0, my=0;
window.addEventListener('pointermove', function(e){ mxT=(e.clientX/W-.5)*2; myT=(e.clientY/H-.5)*2; });
var CH=[{n:'EDA',f:'4 Hz',c:'#B8FF2C',a:'w',y:.1},{n:'TEMP',f:'4 Hz',c:'#FF8A00',a:'w',y:.3},{n:'BVP',f:'64 Hz',c:'#FFE600',a:'w',y:-.1},{n:'ACC',f:'32 Hz',c:'#00E5FF',a:'w',y:.5},{n:'HR (derived)',f:'1 Hz',c:'#FF2D95',a:'h',y:.4}];
var PERIOD=.87;
function ecg(p){ function g(c,w,h){return h*Math.exp(-Math.pow((p-c)/w,2));} return g(.16,.035,.12)-g(.29,.012,.14)+g(.31,.012,1)-g(.335,.014,.28)+g(.56,.06,.2); }
function rr(x,y,w,h,r){ ctx.beginPath(); ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath(); }
function frame(ms){
  var t = CFG.calm ? 1.6 : ms/1000;
  mx += (mxT-mx)*.05; my += (myT-my)*.05;
  ctx.clearRect(0,0,W,H);
  var ecgH=64, bodyH=H-ecgH-14, cx=W/2, cy=bodyH/2+8, sc=bodyH*.46;
  var th = t*.5 + mx*.9, c=Math.cos(th), s=Math.sin(th), pitch=my*.12;
  function proj(p){ var x=p[0]*c+p[2]*s, z=-p[0]*s+p[2]*c, y=p[1]; var y2=y*Math.cos(pitch)-z*Math.sin(pitch); var z2=y*Math.sin(pitch)+z*Math.cos(pitch); var f=3.4/(3.4-z2); return [cx+x*sc*f, cy-y2*sc*f, z2, f]; }
  var g=ctx.createRadialGradient(cx,cy,10,cx,cy,bodyH*.75); g.addColorStop(0,CFG.light?'rgba(47,107,255,.18)':'rgba(47,107,255,.28)'); g.addColorStop(.6,'rgba(139,92,246,.10)'); g.addColorStop(1,'rgba(0,0,0,0)'); ctx.fillStyle=g; ctx.fillRect(0,0,W,bodyH+10);
  // floor rings
  for(var k=0;k<3;k++){ var rad=(.45+k*.2)*sc; ctx.save(); ctx.translate(cx,cy+sc*.97); ctx.scale(1,.28); ctx.beginPath(); ctx.arc(0,0,rad,0,6.283); ctx.setLineDash([6,8]); ctx.lineDashOffset=-t*30*(k%2?-1:1); ctx.strokeStyle=['#00E5FF','#FF2D95','#FFE600'][k]; ctx.globalAlpha=.55-k*.12; ctx.lineWidth=1.6; ctx.stroke(); ctx.restore(); }
  ctx.globalAlpha=1; ctx.setLineDash([]);
  var scan=Math.sin(t*.9)*.95;
  for(var i=0;i<P.length;i++){ var q=proj(P[i]), yy=P[i][1], near=Math.abs(yy-scan); var col=grad((yy+.92)/1.84), hot=near<.045;
    var al=Math.max(.25,Math.min(1,.55+q[2]*.9)); if(hot) al=1;
    ctx.fillStyle=hot?'rgba(255,255,255,.95)':'rgba('+(col[0]|0)+','+(col[1]|0)+','+(col[2]|0)+','+al.toFixed(2)+')';
    var sz=(hot?2.6:1.7)*q[3]; ctx.fillRect(q[0],q[1],sz,sz); }
  var sr=proj([0,scan,0]); ctx.save(); ctx.translate(sr[0],sr[1]); ctx.scale(1,.25); ctx.beginPath(); ctx.arc(0,0,.5*sc*sr[3],0,6.283); ctx.strokeStyle='#00E5FF'; ctx.lineWidth=2.5; ctx.shadowColor='#00E5FF'; ctx.shadowBlur=18; ctx.stroke(); ctx.restore();
  // heart
  var ph=(t%PERIOD)/PERIOD, beat=Math.exp(-ph*7)+.6*Math.exp(-Math.pow((ph-.18)*9,2)), hp=proj([.07,.36,.1]);
  var hr=(10+beat*22)*hp[3]; var hg=ctx.createRadialGradient(hp[0],hp[1],0,hp[0],hp[1],hr*2.2); hg.addColorStop(0,'rgba(255,45,149,.95)'); hg.addColorStop(1,'rgba(255,45,149,0)'); ctx.fillStyle=hg; ctx.beginPath(); ctx.arc(hp[0],hp[1],hr*2.2,0,6.283); ctx.fill();
  ctx.strokeStyle='#FF2D95'; ctx.lineWidth=2; ctx.globalAlpha=Math.max(0,1-ph*1.3); ctx.beginPath(); ctx.arc(hp[0],hp[1],10+ph*70,0,6.283); ctx.stroke(); ctx.globalAlpha=1;
  // wrist band (E4)
  var wp=proj([-.38,.04,0]); ctx.shadowColor='#FFE600'; ctx.shadowBlur=16; ctx.fillStyle='#FFE600'; ctx.beginPath(); ctx.arc(wp[0],wp[1],5*wp[3],0,6.283); ctx.fill(); ctx.shadowBlur=0;
  // sensor chips
  ctx.font='700 11px "Avenir Next","Segoe UI",system-ui,sans-serif'; ctx.textBaseline='middle';
  var chips=[]; CH.forEach(function(h,i){ var ang=t*.32+i*6.283/CH.length; var pos=[Math.cos(ang)*.95, h.y+Math.sin(t*.7+i)*.05, Math.sin(ang)*.6]; var q=proj(pos); chips.push({h:h,q:q,i:i}); });
  chips.sort(function(a,b){return a.q[2]-b.q[2];});
  chips.forEach(function(o){ var h=o.h,q=o.q, anc=h.a==='h'?hp:wp, a=Math.max(.35,Math.min(1,.65+q[2]*.9));
    ctx.globalAlpha=a*.7; ctx.strokeStyle=h.c; ctx.lineWidth=1.3; ctx.setLineDash([4,5]); ctx.lineDashOffset=-t*40; ctx.beginPath(); ctx.moveTo(anc[0],anc[1]); ctx.lineTo(q[0],q[1]); ctx.stroke(); ctx.setLineDash([]);
    var pk=(t*.8+o.i*.2)%1; ctx.fillStyle=h.c; ctx.beginPath(); ctx.arc(anc[0]+(q[0]-anc[0])*pk, anc[1]+(q[1]-anc[1])*pk, 3, 0, 6.283); ctx.fill();
    ctx.globalAlpha=a; var label=h.n+' · '+h.f, tw=ctx.measureText(label).width+26, bx=q[0]-tw/2, by=q[1]-12;
    ctx.fillStyle='rgba(8,12,36,.88)'; rr(bx,by,tw,24,12); ctx.fill(); ctx.strokeStyle=h.c; ctx.lineWidth=1.6; ctx.shadowColor=h.c; ctx.shadowBlur=12; ctx.stroke(); ctx.shadowBlur=0;
    ctx.fillStyle=h.c; ctx.beginPath(); ctx.arc(bx+11,q[1],4,0,6.283); ctx.fill(); ctx.fillStyle='#F2F6FF'; ctx.textAlign='left'; ctx.fillText(label,bx+20,q[1]+.5); ctx.globalAlpha=1; });
  // ECG strip
  var ey=H-ecgH/2-4; ctx.fillStyle='rgba(8,12,36,.88)'; rr(6,H-ecgH-6,W-12,ecgH,12); ctx.fill(); ctx.strokeStyle='rgba(0,229,255,.45)'; ctx.lineWidth=1; ctx.stroke();
  var lg=ctx.createLinearGradient(0,0,W,0); lg.addColorStop(0,'#00E5FF'); lg.addColorStop(.6,'#8B5CF6'); lg.addColorStop(1,'#FF2D95');
  ctx.strokeStyle=lg; ctx.lineWidth=2.2; ctx.shadowColor='#00E5FF'; ctx.shadowBlur=10; ctx.beginPath();
  for(var x=14;x<W-14;x+=2){ var tt=t-(W-14-x)/130; var pp=((tt%PERIOD)+PERIOD)%PERIOD/PERIOD; var yv=ey-ecg(pp)*(ecgH*.36); if(x===14)ctx.moveTo(x,yv); else ctx.lineTo(x,yv); }
  ctx.stroke(); ctx.shadowBlur=0;
  // labels
  ctx.textAlign='left'; ctx.fillStyle=CFG.light?'#0E1738':'#B9C5EE'; ctx.font='800 11px "Avenir Next","Segoe UI",sans-serif';
  var on=(Math.floor(t*2)%2===0)||CFG.calm; ctx.fillStyle=on?'#B8FF2C':'rgba(184,255,44,.25)'; ctx.beginPath(); ctx.arc(16,18,5,0,6.283); ctx.fill();
  ctx.fillStyle=CFG.light?'#0E1738':'#EAF1FF'; ctx.fillText('TWIN · LIVE DEMO',28,19);
  ctx.textAlign='left'; ctx.fillStyle=CFG.light?'#3A4A7E':'#B9C5EE'; ctx.font='600 10px "Avenir Next","Segoe UI",sans-serif'; ctx.fillText('ILLUSTRATIVE · NOT YOUR DATA · COLOUR ≠ STRESS',16,36);
  if(!CFG.calm) requestAnimationFrame(frame);
}
if(CFG.calm){ frame(0); window.addEventListener('resize', function(){frame(0);}); } else requestAnimationFrame(frame);
</script></body></html>
"""


def render_hologram_twin(calm_mode: bool = False, theme: str = "dark", height: int = 470) -> None:
    """Animated, illustrative hologram of the digital twin with orbiting E4 sensor chips."""
    cfg = json.dumps({"calm": bool(calm_mode), "light": theme != "dark"})
    _embed(_HOLO_HTML.replace("__CFG__", cfg), height=height)


# ---------------------------------------------------------------------------
# Ticker + flow strip (pure HTML/CSS)
# ---------------------------------------------------------------------------
_TICKER = [
    ("#00E5FF", "35 participants"), ("#FF2D95", "Leave-one-subject-out validated"),
    ("#FFE600", "23 physiological features"), ("#B8FF2C", "Empatica E4 · EDA · HR · BVP · TEMP · ACC"),
    ("#FF8A00", "Abstains instead of faking confidence"), ("#8B5CF6", "Measured · Estimated · Reported"),
    ("#00E5FF", "Research prototype · not a medical device"),
]


def render_ticker() -> None:
    items = "".join(f"<span><i style='color:{c}'></i>{t}</span>" for c, t in _TICKER)
    st.markdown(compact_html(f"<div class='prism-ticker'><div class='prism-ticker-track'>{items}{items}</div></div>"),
                unsafe_allow_html=True)


def render_flow_strip() -> None:
    def node(color: str, icon: str, title: str, body: str) -> str:
        return (f"<div class='prism-flow-node' style='box-shadow:0 0 28px {color}55'>"
                f"<b style='color:{color}'>{icon} {title}</b><small>{body}</small></div>")
    html = ("<div class='prism-flow'>"
            + node("#00E5FF", "📡", "MEASURED", "Raw wristband signals: EDA, HR, BVP, skin temperature, motion. Quality-vetoed.")
            + "<div class='prism-flow-link'></div>"
            + node("#FF2D95", "🧠", "ESTIMATED", "23 features → subject-independent models → calibrated evidence, or an honest abstention.")
            + "<div class='prism-flow-link'></div>"
            + node("#FFE600", "💬", "REPORTED", "Your own words: PSS-10, WHO-5 and check-ins, stored locally with consent.")
            + "</div>")
    st.markdown(compact_html(html), unsafe_allow_html=True)
