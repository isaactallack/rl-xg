// 3D replay of one shot: the ball and every car, a few seconds either side of the touch.
//
// Rocket League is left-handed (x, y, z-up) and three.js is right-handed y-up, so world points
// map as (x, y, z) -> (-x, z, -y) / 100. That keeps the game's own view: looking from the blue goal
// towards orange, +x is on the left. Car rotations map the same way: quaternion (x, y, z, w) -> (x, -z, y, w).
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const U = 1 / 100;
const BALL_R = 92.75;
const GOAL_HALF_W = 892.755, GOAL_H = 642.775, GOAL_DEPTH = 880;
const ARENA_X = 4096, ARENA_Y = 5120, CEILING = 2044;
const FIELD = [[4096, 3968], [2944, 5120], [-2944, 5120], [-4096, 3968], [-4096, -3968], [-2944, -5120], [2944, -5120], [4096, -3968]];
const BIG_PADS = [[3072, 4096], [-3072, 4096], [3072, -4096], [-3072, -4096], [3584, 0], [-3584, 0]];
const KIND = {
  goal: ["Goal", "#2fd27a"], saved: ["Saved", "#f0b429"], blocked: ["Blocked", "#f0b429"],
  post: ["Hit the post", "#f5c84c"], bar: ["Hit the bar", "#f5c84c"], wide: ["Wide", "#a9b3c4"],
  over: ["Over the bar", "#a9b3c4"], wall: ["Off the wall", "#a9b3c4"], short: ["Fell short", "#a9b3c4"],
};
const CAMS = [["follow", "Behind ball"], ["broadcast", "Broadcast"], ["goal", "Goal cam"], ["top", "Top down"], ["free", "Free"]];
const SPEEDS = [0.25, 0.5, 1];

const V = (p) => new THREE.Vector3(-p[0] * U, p[2] * U, -p[1] * U);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const PLAY = `<svg width="16" height="16" viewBox="0 0 12 12"><path d="M2.5 1.2v9.6L10.5 6z"/></svg>`;
const PAUSE = `<svg width="16" height="16" viewBox="0 0 12 12"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>`;
const REPLAY = `<svg width="16" height="16" viewBox="0 0 12 12"><path d="M6 1.5a4.5 4.5 0 1 1-4.3 5.8h1.6A3 3 0 1 0 6 3v1.8L2.8 2.3 6 0z"/></svg>`;

let active = null;   // only one viewer at a time

export async function openReplay({ host, matchId, idx, shot, colors, label, clock, onClose }) {
  if (active) active.close();
  const res = await fetch(`/api/matches/${encodeURIComponent(matchId)}/clips/${idx}`);
  if (!res.ok) { const e = new Error(`clip ${res.status}`); e.status = res.status; throw e; }
  const clip = await res.json();
  active = new Viewer(host, clip, shot, colors, label, clock, onClose);
}

class Viewer {
  constructor(host, clip, shot, colors, label, clock, onClose) {
    this.host = host; this.clip = clip; this.shot = shot; this.onClose = onClose;
    this.colors = { blue: new THREE.Color(colors.blue || "#3b8cff"), orange: new THREE.Color(colors.orange || "#ff7a2f") };
    this.teamHex = { blue: colors.blue || "#3b8cff", orange: colors.orange || "#ff7a2f" };
    this.t = clip.t; this.t0 = clip.t[0]; this.t1 = clip.t[clip.t.length - 1];
    this.cur = this.t0; this.playing = false; this.speed = 1; this.mode = "follow"; this.showPred = true;
    // which goal is being attacked, in three.js space (orange goal sits at -z)
    this.atkSign = clip.team === "blue" ? -1 : 1;
    this.goalCentre = V([0, (clip.team === "blue" ? 1 : -1) * ARENA_Y, 320]);
    this.buildDom(label, clock);
    this.buildScene();
    this.onKey = e => this.key(e);
    addEventListener("keydown", this.onKey, true);
    this.last = performance.now();
    this.raf = requestAnimationFrame(t => this.frame(t));
    setTimeout(() => { if (this.cur === this.t0) this.play(true); }, 350);
  }

  // ---------- DOM ----------
  buildDom(label, clock) {
    const s = this.shot, kind = this.clip.end_kind || (s.goal ? "goal" : "wide");
    const [kText, kCol] = KIND[kind] || [label, "#a9b3c4"];
    const col = this.teamHex[s.team] || this.teamHex[this.clip.team];
    const span = this.t1 - this.t0;
    const pct = t => `${100 * (t - this.t0) / span}%`;
    const tEnd = Math.min(Math.max(this.clip.t_end ?? 0, 0), this.t1);
    this.host.innerHTML = `
      <div class="r3d-box">
        <div class="r3d-stage">
          <div class="r3d-hud">
            <div class="r3d-info">
              <div class="who"><span style="width:26px;height:26px;border-radius:50%;display:grid;place-items:center;background:${col};font-size:12px;font-weight:800;color:#fff">${esc((s.player || "?")[0].toUpperCase())}</span>${esc(s.player)}</div>
              <div class="meta">${esc(clock)} · <span style="color:${kCol};font-weight:700">${esc(kText)}</span>${s.end_by ? ` by ${esc(s.end_by)}` : ""}</div>
              <div class="nums">
                <div><b>${(s.xg ?? 0).toFixed(2)}</b><span>xG</span></div>
                <div><b>${s.xgot == null ? "–" : s.xgot.toFixed(2)}</b><span>xGOT</span></div>
              </div>
            </div>
            <button class="r3d-close" aria-label="Close 3D replay" title="Close (Esc)">×</button>
          </div>
          <div class="r3d-clock"><span class="tt">0.00 s</span><span class="ph"></span></div>
          <div class="r3d-legend"><span><i style="border-color:${col}"></i>Ball path</span><span><i class="dash" style="border-color:#dfe6ef"></i>Where the touch was sending it</span></div>
        </div>
        <div class="r3d-bar">
          <button class="r3d-play" aria-label="Play or pause (space)" title="Play / pause (space)">${PLAY}</button>
          <div class="grp hide-sm" aria-label="Playback speed">${SPEEDS.map(v => `<button data-speed="${v}" class="${v === 1 ? "on" : ""}">${v}×</button>`).join("")}</div>
          <div class="r3d-scrub">
            <span class="mk" style="left:${pct(0)};color:#eef2f8">Shot</span>
            ${kind !== "goal" || tEnd > 0.05 ? `<span class="mk" style="left:${pct(tEnd)};color:${kCol}">${esc(kText)}</span>` : ""}
            <input type="range" min="0" max="1000" value="0" aria-label="Time">
          </div>
          <div class="r3d-time">0.00 s</div>
          <div class="grp" aria-label="Camera">${CAMS.map(([k, l], i) => `<button data-cam="${k}" class="${k === this.mode ? "on" : ""}" title="${l} (${i + 1})">${l}</button>`).join("")}
            <button data-pred class="on hide-sm" title="Show where the touch was sending the ball">Heading</button></div>
        </div>
      </div>`;
    this.host.classList.add("open");
    const q = sel => this.host.querySelector(sel);
    this.stage = q(".r3d-stage"); this.scrub = q(".r3d-scrub input"); this.playBtn = q(".r3d-play");
    this.timeEl = q(".r3d-time"); this.clockEl = q(".r3d-clock .tt"); this.phaseEl = q(".r3d-clock .ph");
    q(".r3d-close").onclick = () => this.close();
    this.host.onclick = e => { if (e.target === this.host) this.close(); };
    this.playBtn.onclick = () => this.playing ? this.pause() : this.play();
    this.scrub.oninput = () => { this.pause(); this.cur = this.t0 + (+this.scrub.value / 1000) * span; };
    this.host.querySelectorAll("[data-speed]").forEach(b => b.onclick = () => {
      this.speed = +b.dataset.speed; this.host.querySelectorAll("[data-speed]").forEach(x => x.classList.toggle("on", x === b));
    });
    this.host.querySelectorAll("[data-cam]").forEach(b => b.onclick = () => this.setMode(b.dataset.cam));
    const pb = q("[data-pred]");
    pb.onclick = () => { this.showPred = !this.showPred; pb.classList.toggle("on", this.showPred); if (this.pred) this.pred.visible = this.showPred; };
  }

  setMode(m) {
    this.mode = m;
    this.host.querySelectorAll("[data-cam]").forEach(x => x.classList.toggle("on", x.dataset.cam === m));
  }
  play(fromStart = false) {
    if (fromStart || this.cur >= this.t1 - 1e-3) this.cur = this.t0;
    this.playing = true; this.playBtn.innerHTML = PAUSE;
  }
  pause() { this.playing = false; this.playBtn.innerHTML = this.cur >= this.t1 - 1e-3 ? REPLAY : PLAY; }

  key(e) {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); this.close(); return; }
    if (e.target.closest && e.target.closest("input[type=text], select, textarea")) return;
    if (e.key === " ") { e.preventDefault(); this.playing ? this.pause() : this.play(); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault(); e.stopPropagation(); this.pause();
      this.cur = Math.max(this.t0, Math.min(this.t1, this.cur + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 0.5 : 1 / 30)));
    } else if (/^[1-5]$/.test(e.key)) this.setMode(CAMS[+e.key - 1][0]);
  }

  // ---------- scene ----------
  buildScene() {
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    this.stage.prepend(r.domElement);
    const scene = this.scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0d13);
    scene.fog = new THREE.Fog(0x0a0d13, 140, 300);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.addEventListener("start", () => this.setMode("free"));

    scene.add(new THREE.HemisphereLight(0xa8c4ff, 0x0b0f16, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(25, 60, 15); scene.add(sun);
    scene.add(new THREE.AmbientLight(0xffffff, 0.25));

    this.buildArena();
    this.buildActors();
    this.buildPaths();

    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(this.stage);
    this.resize();
    // start the camera somewhere sensible, then let the preset ease in
    const b = this.ballAt(this.t0);
    this.camera.position.copy(b.clone().add(new THREE.Vector3(30, 25, -this.atkSign * 30)));
    this.controls.target.copy(b);
  }

  floorTexture() {
    const CW = 1024, CH = 1280, cv = document.createElement("canvas");
    cv.width = CW; cv.height = CH;
    const g = cv.getContext("2d");
    const P = (x, y) => [(ARENA_X - x) / (2 * ARENA_X) * CW, (ARENA_Y - y) / (2 * ARENA_Y) * CH];  // top = orange end
    const poly = () => { g.beginPath(); FIELD.forEach(([x, y], i) => { const [a, b] = P(x, y); i ? g.lineTo(a, b) : g.moveTo(a, b); }); g.closePath(); };
    poly(); g.fillStyle = "#141d29"; g.fill();
    g.save(); poly(); g.clip();
    for (let k = 0; k < 16; k += 2) { g.fillStyle = "#17212e"; g.fillRect(0, k * CH / 16, CW, CH / 16); }
    g.globalAlpha = 0.09; g.fillStyle = this.teamHex.orange; g.fillRect(0, 0, CW, CH / 2);
    g.fillStyle = this.teamHex.blue; g.fillRect(0, CH / 2, CW, CH / 2); g.globalAlpha = 1;
    g.restore();
    g.strokeStyle = "rgba(190,210,240,.35)"; g.lineWidth = 3;
    poly(); g.stroke();
    g.beginPath(); g.moveTo(0, CH / 2); g.lineTo(CW, CH / 2); g.stroke();
    g.beginPath(); g.arc(CW / 2, CH / 2, 1000 / (2 * ARENA_X) * CW, 0, Math.PI * 2); g.stroke();
    for (const s of [1, -1]) {
      const [x0, y0] = P(1800, s * ARENA_Y), [x1, y1] = P(-1800, s * (ARENA_Y - 1300));
      g.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    }
    for (const [x, y] of BIG_PADS) {
      const [a, b] = P(x, y);
      g.beginPath(); g.arc(a, b, 16, 0, Math.PI * 2); g.fillStyle = "rgba(255,190,90,.55)"; g.fill();
    }
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
    return tex;
  }

  buildArena() {
    const scene = this.scene;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(2 * ARENA_X * U, 2 * ARENA_Y * U),
      new THREE.MeshStandardMaterial({ map: this.floorTexture(), transparent: true, alphaTest: 0.5, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    // see-through walls with bright edges
    const wallMat = new THREE.MeshBasicMaterial({ color: 0x8fb3ff, transparent: true, opacity: 0.035, side: THREE.DoubleSide, depthWrite: false });
    const edge = new THREE.LineBasicMaterial({ color: 0x3a4b63 });
    const wall = (a, b, z0, z1) => {
      const pts = [V([a[0], a[1], z0]), V([b[0], b[1], z0]), V([b[0], b[1], z1]), V([a[0], a[1], z1])];
      const geo = new THREE.BufferGeometry().setFromPoints([pts[0], pts[1], pts[2], pts[0], pts[2], pts[3]]);
      scene.add(new THREE.Mesh(geo, wallMat));
    };
    for (let i = 0; i < FIELD.length; i++) {
      const a = FIELD[i], b = FIELD[(i + 1) % FIELD.length];
      if (Math.abs(a[1]) === ARENA_Y && Math.abs(b[1]) === ARENA_Y) {      // back wall: leave the goal mouth open
        const y = a[1], sx = Math.sign(a[0]);
        wall([a[0], y], [sx * GOAL_HALF_W, y], 0, CEILING);
        wall([-sx * GOAL_HALF_W, y], [b[0], y], 0, CEILING);
        wall([sx * GOAL_HALF_W, y], [-sx * GOAL_HALF_W, y], GOAL_H, CEILING);
      } else wall(a, b, 0, CEILING);
    }
    for (const z of [0, CEILING]) {
      const g = new THREE.BufferGeometry().setFromPoints(FIELD.map(([x, y]) => V([x, y, z])));
      scene.add(new THREE.LineLoop(g, z ? new THREE.LineBasicMaterial({ color: 0x26324a }) : edge));
    }
    for (const [x, y] of FIELD) scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([V([x, y, 0]), V([x, y, CEILING])]), new THREE.LineBasicMaterial({ color: 0x1d2738 })));

    for (const [team, sy] of [["orange", 1], ["blue", -1]]) this.buildGoal(team, sy);
  }

  buildGoal(team, sy) {
    const col = this.colors[team], y0 = sy * ARENA_Y, y1 = sy * (ARENA_Y + GOAL_DEPTH);
    const g = new THREE.Group();
    const frame = new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.6, roughness: 0.4 });
    const bar = (a, b, r = 14) => {
      const A = V(a), B = V(b), len = A.distanceTo(B);
      const m = new THREE.Mesh(new THREE.CylinderGeometry(r * U, r * U, len, 10), frame);
      m.position.copy(A).add(B).multiplyScalar(0.5);
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize());
      g.add(m);
    };
    bar([GOAL_HALF_W, y0, 0], [GOAL_HALF_W, y0, GOAL_H]);
    bar([-GOAL_HALF_W, y0, 0], [-GOAL_HALF_W, y0, GOAL_H]);
    bar([GOAL_HALF_W, y0, GOAL_H], [-GOAL_HALF_W, y0, GOAL_H]);
    for (const x of [GOAL_HALF_W, -GOAL_HALF_W]) { bar([x, y1, 0], [x, y1, GOAL_H], 7); bar([x, y0, GOAL_H], [x, y1, GOAL_H], 7); }
    bar([GOAL_HALF_W, y1, GOAL_H], [-GOAL_HALF_W, y1, GOAL_H], 7);
    // nets
    const net = new THREE.MeshBasicMaterial({ color: col, wireframe: true, transparent: true, opacity: 0.22 });
    const fill = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false });
    const quad = (pts, sw, sh) => {
      const [a, b, c, d] = pts.map(V);
      const geo = new THREE.PlaneGeometry(1, 1, sw, sh), pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++) {   // bilinear map of the unit plane onto the quad
        const u = pos.getX(i) + 0.5, v = pos.getY(i) + 0.5;
        const p = a.clone().lerp(b, u).lerp(d.clone().lerp(c, u), v);
        pos.setXYZ(i, p.x, p.y, p.z);
      }
      g.add(new THREE.Mesh(geo, net)); g.add(new THREE.Mesh(geo, fill));
    };
    const W = GOAL_HALF_W, H = GOAL_H;
    quad([[W, y1, 0], [-W, y1, 0], [-W, y1, H], [W, y1, H]], 18, 7);   // back
    quad([[W, y0, H], [-W, y0, H], [-W, y1, H], [W, y1, H]], 18, 9);   // roof
    for (const x of [W, -W]) quad([[x, y0, 0], [x, y1, 0], [x, y1, H], [x, y0, H]], 9, 7);
    // goal floor glow
    const glow = new THREE.Mesh(new THREE.PlaneGeometry(2 * W * U, GOAL_DEPTH * U), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.18 }));
    glow.rotation.x = -Math.PI / 2; glow.position.copy(V([0, (y0 + y1) / 2, 1]));
    g.add(glow);
    this.scene.add(g);
  }

  label(text, color, bold) {
    const cv = document.createElement("canvas"), g = cv.getContext("2d");
    const font = `${bold ? 800 : 600} 44px Inter, system-ui, sans-serif`;
    g.font = font;
    const w = Math.ceil(g.measureText(text).width) + 48, h = 72;
    cv.width = w; cv.height = h; g.font = font;
    g.fillStyle = "rgba(10,13,19,.78)"; g.beginPath(); g.roundRect(2, 2, w - 4, h - 4, 18); g.fill();
    g.lineWidth = 4; g.strokeStyle = color; g.stroke();
    g.fillStyle = "#eef2f8"; g.textBaseline = "middle"; g.textAlign = "center"; g.fillText(text, w / 2, h / 2 + 2);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    const sh = 1.5; sp.scale.set(sh * w / h, sh, 1); sp.renderOrder = 10;
    return sp;
  }

  blob(radius, opacity) {
    const cv = document.createElement("canvas"); cv.width = cv.height = 64;
    const g = cv.getContext("2d"), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, "rgba(0,0,0,1)"); grd.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(radius * 2, radius * 2),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, opacity, depthWrite: false }));
    m.rotation.x = -Math.PI / 2;
    return m;
  }

  buildActors() {
    const scene = this.scene;
    // ball
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(BALL_R * U, 40, 24),
      new THREE.MeshStandardMaterial({ color: 0xe8edf5, roughness: 0.35, metalness: 0.15, emissive: 0x3a4250, emissiveIntensity: 0.6 }));
    scene.add(this.ball);
    this.ballShadow = this.blob(BALL_R * U * 1.3, 0.55); scene.add(this.ballShadow);
    // cars: an Octane-sized hitbox with a cabin and wheels, in the team colour
    const wheelGeo = new THREE.CylinderGeometry(15 * U, 15 * U, 10 * U, 16); wheelGeo.rotateX(Math.PI / 2);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111418, roughness: 0.8 });
    const glass = new THREE.MeshStandardMaterial({ color: 0x0d1117, roughness: 0.2, metalness: 0.6 });
    const L = (x, y, z) => new THREE.Vector3(-x * U, z * U, -y * U);   // car-local RL axes -> three axes
    this.cars = this.clip.cars.map(c => {
      const col = this.colors[c.team], isShooter = c.pid === this.clip.shooter;
      const grp = new THREE.Group();
      const paint = new THREE.MeshStandardMaterial({ color: col, roughness: 0.45, metalness: 0.25, emissive: col, emissiveIntensity: isShooter ? 0.45 : 0.2 });
      const body = new THREE.Mesh(new THREE.BoxGeometry(118 * U, 36.2 * U, 84.2 * U), paint);
      body.position.copy(L(13.88, 0, 20.75)); grp.add(body);
      const cabin = new THREE.Mesh(new THREE.BoxGeometry(56 * U, 22 * U, 70 * U), glass);
      cabin.position.copy(L(-8, 0, 49)); grp.add(cabin);
      const spoiler = new THREE.Mesh(new THREE.BoxGeometry(10 * U, 6 * U, 80 * U), paint);
      spoiler.position.copy(L(-42, 0, 42)); grp.add(spoiler);
      for (const [x, y] of [[48, 38], [48, -38], [-30, 38], [-30, -38]]) {
        const w = new THREE.Mesh(wheelGeo, wheelMat); w.position.copy(L(x, y, 8)); grp.add(w);
      }
      scene.add(grp);
      const tag = this.label(c.name, this.teamHex[c.team], isShooter); scene.add(tag);
      const shadow = this.blob(0.9, 0.5); scene.add(shadow);
      let ring = null;
      if (isShooter) {
        ring = new THREE.Mesh(new THREE.RingGeometry(1.1, 1.35, 40), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false }));
        ring.rotation.x = -Math.PI / 2; scene.add(ring);
      }
      return { c, grp, tag, shadow, ring };
    });
  }

  buildPaths() {
    const clip = this.clip, col = this.colors[clip.team];
    // the real ball path from the touch to where the shot ended
    const i0 = clip.t.findIndex(t => t >= 0);
    const tEnd = Math.max(clip.t_end ?? 0, 0.05);
    let i1 = clip.t.findIndex(t => t > tEnd); if (i1 < 0) i1 = clip.t.length - 1;
    const pts = [];
    for (let i = Math.max(i0, 0); i <= i1; i++) if (clip.ball[i] && clip.ball[i][0] != null) pts.push(V(clip.ball[i]));
    if (pts.length >= 2) {
      const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
      const segs = Math.max(8, pts.length * 3);
      const geo = new THREE.TubeGeometry(curve, segs, 0.16, 8, false);
      this.pathMesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.95 }));
      this.pathIdx = geo.index.count; this.pathT = [clip.t[Math.max(i0, 0)], clip.t[i1]];
      this.scene.add(this.pathMesh);
    }
    // where the touch was sending it
    if (clip.pred && clip.pred.length >= 2) {
      const geo = new THREE.BufferGeometry().setFromPoints(clip.pred.map(V));
      this.pred = new THREE.Line(geo, new THREE.LineDashedMaterial({ color: 0xdfe6ef, dashSize: 0.9, gapSize: 0.6, transparent: true, opacity: 0.6 }));
      this.pred.computeLineDistances();
      this.scene.add(this.pred);
    }
    // touch point and end point
    if (i0 >= 0 && clip.ball[i0]?.[0] != null) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.28, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      m.position.copy(V(clip.ball[i0])); this.scene.add(m);
    }
    if (clip.end && clip.end_kind && clip.end_kind !== "goal") {
      const [txt, hex] = KIND[clip.end_kind] || ["", "#a9b3c4"];
      const p = V(clip.end);
      const mk = new THREE.Mesh(new THREE.SphereGeometry(0.45, 20, 14), new THREE.MeshBasicMaterial({ color: new THREE.Color(hex) }));
      mk.position.copy(p); this.scene.add(mk);
      this.endTag = this.label(txt, hex, true); this.endTag.position.copy(p).add(new THREE.Vector3(0, 2.2, 0)); this.scene.add(this.endTag);
      this.endMk = mk;
    }
  }

  resize() {
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  // ---------- playback ----------
  locate(t) {
    const T = this.t;
    let lo = 0, hi = T.length - 1;
    if (t <= T[0]) return [0, 0, 0];
    if (t >= T[hi]) return [hi, hi, 0];
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
    return [lo, hi, (t - T[lo]) / Math.max(T[hi] - T[lo], 1e-6)];
  }
  lerpPos(arr, i, j, a) {
    const p = arr[i], q = arr[j];
    const ok = v => v && v[0] != null;
    if (!ok(p) && !ok(q)) return null;
    if (!ok(p)) return V(q);
    if (!ok(q)) return V(p);
    return V(p).lerp(V(q), a);
  }
  ballAt(t) {
    const [i, j, a] = this.locate(t);
    return this.lerpPos(this.clip.ball, i, j, a) || new THREE.Vector3();
  }
  carQuat(q) { return q && q[0] != null ? new THREE.Quaternion(q[0], -q[2], q[1], q[3]).normalize() : null; }

  update() {
    const [i, j, a] = this.locate(this.cur);
    const b = this.lerpPos(this.clip.ball, i, j, a);
    if (b) {
      this.ball.position.copy(b); this.ball.visible = true;
      this.ballShadow.position.set(b.x, 0.02, b.z);
      const hgt = Math.max(0, b.y - BALL_R * U);
      this.ballShadow.material.opacity = 0.55 * Math.max(0.15, 1 - hgt / 12);
      this.ballShadow.scale.setScalar(1 + hgt / 10);
      this.ball.rotation.x += 0.02 * (this.playing ? this.speed : 0);
    } else this.ball.visible = false;
    for (const car of this.cars) {
      const p = this.lerpPos(car.c.pos, i, j, a);
      const qa = this.carQuat(car.c.quat[i]), qb = this.carQuat(car.c.quat[j]);
      const vis = !!p;
      car.grp.visible = car.tag.visible = car.shadow.visible = vis;
      if (car.ring) car.ring.visible = vis;
      if (!vis) continue;
      car.grp.position.copy(p);
      if (qa && qb) car.grp.quaternion.slerpQuaternions(qa, qb, a); else if (qa || qb) car.grp.quaternion.copy(qa || qb);
      car.tag.position.copy(p).add(new THREE.Vector3(0, 2.4, 0));
      car.shadow.position.set(p.x, 0.015, p.z);
      car.shadow.material.opacity = 0.5 * Math.max(0, 1 - p.y / 10);
      if (car.ring) { car.ring.position.set(p.x, 0.03, p.z); car.ring.material.opacity = 0.85 * Math.max(0.2, 1 - p.y / 15); }
    }
    // reveal the ball path as the ball travels it
    if (this.pathMesh) {
      const [ta, tb] = this.pathT;
      const f = Math.max(0, Math.min(1, (this.cur - ta) / Math.max(tb - ta, 1e-6)));
      const n = Math.floor(this.pathIdx * f / 6) * 6;
      this.pathMesh.geometry.setDrawRange(0, n);
      this.pathMesh.visible = n > 0;
    }
    const ended = this.cur >= (this.clip.t_end ?? 0) - 1e-3;
    if (this.endTag) { this.endTag.visible = ended; this.endMk.visible = ended; }
    if (this.pred) this.pred.visible = this.showPred && this.cur >= -0.05;
    // HUD
    const rel = this.cur;
    const txt = `${rel < 0 ? "−" : "+"}${Math.abs(rel).toFixed(2)} s`;
    this.clockEl.textContent = txt; this.timeEl.textContent = txt;
    this.phaseEl.textContent = rel < -0.02 ? "build-up" : !ended ? "shot" : "after";
    if (document.activeElement !== this.scrub) this.scrub.value = String(Math.round(1000 * (this.cur - this.t0) / (this.t1 - this.t0)));
  }

  cameraGoal() {
    const b = this.ball.position, G = this.goalCentre, s = this.atkSign;
    switch (this.mode) {
      case "follow": {
        const dir = new THREE.Vector3(G.x - b.x, 0, G.z - b.z);
        if (dir.lengthSq() < 1e-6) dir.set(0, 0, s); dir.normalize();
        return [b.clone().addScaledVector(dir, -16).add(new THREE.Vector3(0, 5.5 + b.y * 0.3, 0)), b.clone().lerp(G, 0.25)];
      }
      case "broadcast":
        return [new THREE.Vector3(62, 34, b.z * 0.85), new THREE.Vector3(b.x * 0.5, 2, b.z)];
      case "goal":
        return [new THREE.Vector3(0, 8, G.z + s * 14), b.clone()];
      case "top":
        return [new THREE.Vector3(0, 105, G.z * 0.45 - s * 12), new THREE.Vector3(0, 0, G.z * 0.45)];
      default: return null;
    }
  }

  frame(now) {
    if (!this.renderer) return;
    const dt = Math.min(0.25, (now - this.last) / 1000); this.last = now;
    if (this.playing) {
      this.cur += dt * this.speed;
      if (this.cur >= this.t1) { this.cur = this.t1; this.pause(); }
    }
    this.update();
    const goal = this.cameraGoal();
    if (goal) {
      const k = 1 - Math.exp(-dt * 5);
      this.camera.position.lerp(goal[0], k);
      this.controls.target.lerp(goal[1], k);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.raf = requestAnimationFrame(t => this.frame(t));
  }

  close() {
    cancelAnimationFrame(this.raf);
    removeEventListener("keydown", this.onKey, true);
    this.ro?.disconnect();
    this.controls?.dispose();
    this.scene?.traverse(o => {
      o.geometry?.dispose();
      const ms = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      ms.forEach(m => { m.map?.dispose(); m.dispose(); });
    });
    this.renderer?.dispose(); this.renderer?.forceContextLoss?.();
    this.renderer = null;
    this.host.classList.remove("open"); this.host.innerHTML = ""; this.host.onclick = null;
    if (active === this) active = null;
    this.onClose?.();
  }
}
