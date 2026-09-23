/* Procedural textures, drawn on a 2D canvas at load. No downloaded assets: the only photographic
   texture in the scene is the real fundus image, which should be the one thing that looks real. */
import * as THREE from "three";

function canvas(size: number) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  return { c, g: c.getContext("2d")! };
}

/** Sclera: warm off-white with faint vessels creeping in from the edges (never pure white). */
export function scleraTexture(size = 1024) {
  const { c, g } = canvas(size);
  g.fillStyle = "#f7f1e6";
  g.fillRect(0, 0, size, size);

  // subtle mottling so the white is not flat
  for (let i = 0; i < 2600; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 6 + Math.random() * 26;
    g.fillStyle = `rgba(${226 + Math.random() * 22}, ${210 + Math.random() * 20}, ${196 + Math.random() * 20}, 0.05)`;
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.fill();
  }

  // vessels: branching strokes, denser away from the centre of the visible face (u ~ 0.5)
  const vessel = (x: number, y: number, ang: number, len: number, w: number, depth: number) => {
    if (depth > 3 || len < 6) return;
    const steps = Math.ceil(len / 5);
    g.strokeStyle = `rgba(172, 74, 58, ${0.05 + 0.05 * (4 - depth)})`;
    g.lineWidth = w;
    g.lineCap = "round";
    g.beginPath();
    g.moveTo(x, y);
    let cx = x;
    let cy = y;
    let a = ang;
    for (let i = 0; i < steps; i++) {
      a += (Math.random() - 0.5) * 0.45;
      cx += Math.cos(a) * 5;
      cy += Math.sin(a) * 5;
      g.lineTo(cx, cy);
    }
    g.stroke();
    if (Math.random() < 0.75) vessel(cx, cy, a + 0.5, len * 0.6, w * 0.65, depth + 1);
    if (Math.random() < 0.75) vessel(cx, cy, a - 0.5, len * 0.6, w * 0.65, depth + 1);
  };
  for (let i = 0; i < 90; i++) {
    const edge = Math.random() < 0.5 ? 0.08 : 0.92;          // left/right of the equator strip
    vessel(size * edge, Math.random() * size, (Math.random() - 0.5) * 1.2 + (edge < 0.5 ? 0 : Math.PI),
           90 + Math.random() * 150, 2.6, 0);
  }

  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

/** Iris: radial fibres, crypts and a dark limbal ring. Warm brown — the common iris colour in the
    population this is built for, and it sits inside the page palette rather than fighting it. */
export function irisTexture(size = 1024) {
  const { c, g } = canvas(size);
  const mid = size / 2;
  const R = mid;

  const base = g.createRadialGradient(mid, mid, R * 0.16, mid, mid, R);
  base.addColorStop(0, "#7a4a1d");
  base.addColorStop(0.35, "#8d5a24");
  base.addColorStop(0.72, "#6b3d16");
  base.addColorStop(0.95, "#3c2110");
  base.addColorStop(1, "#241408");
  g.fillStyle = base;
  g.fillRect(0, 0, size, size);

  // radial fibres
  for (let i = 0; i < 2200; i++) {
    const a = Math.random() * Math.PI * 2;
    const r0 = R * (0.17 + Math.random() * 0.1);
    const r1 = R * (0.55 + Math.random() * 0.42);
    const warm = Math.random();
    g.strokeStyle = warm > 0.6
      ? `rgba(214, 158, 78, ${0.05 + Math.random() * 0.16})`
      : `rgba(58, 31, 12, ${0.05 + Math.random() * 0.2})`;
    g.lineWidth = 0.6 + Math.random() * 2.4;
    g.beginPath();
    let a2 = a;
    g.moveTo(mid + Math.cos(a) * r0, mid + Math.sin(a) * r0);
    const steps = 7;
    for (let s = 1; s <= steps; s++) {
      a2 += (Math.random() - 0.5) * 0.045;
      const r = r0 + ((r1 - r0) * s) / steps;
      g.lineTo(mid + Math.cos(a2) * r, mid + Math.sin(a2) * r);
    }
    g.stroke();
  }

  // crypts: darker lacunae around the collarette
  for (let i = 0; i < 34; i++) {
    const a = Math.random() * Math.PI * 2;
    const r = R * (0.3 + Math.random() * 0.16);
    g.fillStyle = `rgba(30, 16, 6, ${0.16 + Math.random() * 0.26})`;
    g.beginPath();
    g.ellipse(mid + Math.cos(a) * r, mid + Math.sin(a) * r,
              R * (0.02 + Math.random() * 0.055), R * (0.012 + Math.random() * 0.03), a, 0, Math.PI * 2);
    g.fill();
  }

  // collarette highlight + limbal ring
  g.strokeStyle = "rgba(226, 178, 104, 0.24)";
  g.lineWidth = R * 0.035;
  g.beginPath();
  g.arc(mid, mid, R * 0.33, 0, Math.PI * 2);
  g.stroke();

  const limbal = g.createRadialGradient(mid, mid, R * 0.86, mid, mid, R);
  limbal.addColorStop(0, "rgba(20, 10, 4, 0)");
  limbal.addColorStop(1, "rgba(16, 8, 3, 0.92)");
  g.fillStyle = limbal;
  g.fillRect(0, 0, size, size);

  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

/** Planar UVs on a sphere cap, so a circular photograph maps onto curved geometry undistorted. */
export function planarUVs(geo: THREE.BufferGeometry, radius: number) {
  const pos = geo.attributes.position;
  const uv = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    uv[i * 2] = pos.getX(i) / (2 * radius) + 0.5;
    uv[i * 2 + 1] = pos.getY(i) / (2 * radius) + 0.5;
  }
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  return geo;
}
