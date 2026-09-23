/* Camera director. Scroll progress picks a point on a keyframe path; the camera eases toward it
   every frame. Keyframes are deliberately sparse — the journey is one continuous move from arm's
   length, through the pupil, to the back of the eye, and out again. */
import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense } from "react";
import * as THREE from "three";

import { progress } from "../scroll";
import { Eye } from "./Eye";

type Key = { at: number; pos: [number, number, number]; look: [number, number, number]; fov: number };

/* The camera path. Outside the eye it circles and approaches; from 0.36 to 0.83 it tracks sideways
   across the retina, stopping over each of the five annotated lesions in Eye.tsx (the `at` values
   match). The look target is offset sideways from the camera rather than the camera being offset
   from the lesion: that keeps the stop centred in the open half of the screen, away from the card,
   without the camera ever wandering off the edge of the fundus. */
const PATH: Key[] = [
  { at: 0.0,   pos: [0.0, 0.05, 6.4],       look: [-0.75, 0, 0],         fov: 34 },  // arm's length
  { at: 0.067, pos: [0.6, 0.5, 5.2],        look: [-0.75, 0, 0],         fov: 34 },  // drift in
  { at: 0.149, pos: [1.3, 0.35, 3.9],       look: [0.75, 0.05, 0],       fov: 34 },  // swing round
  { at: 0.2,   pos: [0.5, 0.15, 2.2],       look: [0.3, 0, 0.2],         fov: 32 },  // approach
  { at: 0.238, pos: [0.1, 0.05, 1.25],      look: [0, 0, 0.4],           fov: 30 },  // at the cornea
  { at: 0.27,  pos: [0.0, 0.0, 0.5],        look: [0, 0, -0.6],          fov: 42 },  // in the pupil
  { at: 0.305, pos: [0.0, 0.0, -0.05],      look: [0, 0, -0.95],         fov: 58 },  // retina fills
  { at: 0.36,  pos: [-0.1, -0.02, -0.26],   look: [-0.1, -0.02, -0.95],  fov: 56 },  // whole fundus
  { at: 0.429, pos: [-0.684, 0.164, -0.4],  look: [-0.884, 0.164, -0.95], fov: 50 }, // haemorrhage
  { at: 0.531, pos: [-0.478, -0.083, -0.4], look: [-0.278, -0.083, -0.95], fov: 50 }, // optic disc
  { at: 0.62,  pos: [-0.219, -0.289, -0.52], look: [-0.419, -0.289, -0.95], fov: 40 }, // microaneurysm
  { at: 0.702, pos: [0.145, -0.127, -0.42], look: [0.345, -0.127, -0.95], fov: 48 },  // hard exudate
  { at: 0.787, pos: [0.523, -0.2, -0.4],    look: [0.323, -0.2, -0.95],  fov: 50 },  // haemorrhage
  { at: 0.86,  pos: [0.2, -0.05, -0.12],    look: [0, 0, -0.7],          fov: 54 },  // back to centre
  { at: 0.93,  pos: [0.0, 0.0, 1.4],        look: [0, 0, 0],             fov: 38 },  // out of the pupil
  { at: 1.0,   pos: [0.0, 0.05, 5.6],       look: [0, 0, 0],             fov: 34 },  // whole eye again
];

const vA = new THREE.Vector3();
const vB = new THREE.Vector3();
const lookA = new THREE.Vector3();
const lookB = new THREE.Vector3();
const target = new THREE.Vector3();
const lookTarget = new THREE.Vector3();

function ease(t: number) {
  return t * t * (3 - 2 * t);
}

function Director() {
  useFrame(({ camera }, delta) => {
    const p = progress.current;
    let i = 0;
    while (i < PATH.length - 2 && p > PATH[i + 1].at) i++;
    const a = PATH[i];
    const b = PATH[i + 1];
    const t = ease(Math.min(1, Math.max(0, (p - a.at) / (b.at - a.at))));

    target.lerpVectors(vA.fromArray(a.pos), vB.fromArray(b.pos), t);
    lookTarget.lerpVectors(lookA.fromArray(a.look), lookB.fromArray(b.look), t);

    // critically damped follow: the camera never snaps, even if the user flings the scrollbar
    const k = 1 - Math.pow(0.0015, delta);
    camera.position.lerp(target, k);
    camera.lookAt(lookTarget);

    const cam = camera as THREE.PerspectiveCamera;
    cam.fov += (a.fov + (b.fov - a.fov) * t - cam.fov) * k;
    cam.updateProjectionMatrix();
    // dev-only readout, for checking the path lands where the copy says it does
    if (import.meta.env.DEV) {
      (window as unknown as { __cam: unknown }).__cam =
        { p, i, pos: camera.position.toArray().map((n) => +n.toFixed(3)), fov: +cam.fov.toFixed(1) };
    }
  });
  return null;
}

export function Scene() {
  return (
    <div className="stage">
      <Canvas
        dpr={[1, 1.75]}
        gl={{ antialias: true, alpha: true }}
        camera={{ position: [0, 0.05, 6.4], fov: 34, near: 0.01, far: 40 }}
      >
        <ambientLight intensity={0.75} color="#fff4e2" />
        <directionalLight position={[2.4, 3.2, 4]} intensity={2.1} color="#fffaf0" />
        <directionalLight position={[-3, -1.5, 2]} intensity={0.5} color="#d9b98a" />
        <Suspense fallback={null}>
          <Eye />
        </Suspense>
        <Director />
      </Canvas>
    </div>
  );
}
