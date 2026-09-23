/* The eye. Built as a real aperture rather than a picture of one: the sclera sphere has a hole at
   the front, the iris is a ring around that hole, and the retina is a curved cap inside. The camera
   flies through the pupil, so "entering the eye" is geometry, not a cross-fade.

   Inside, the lesions are drawn the way an annotation tool draws them: a contour at full strength
   over a faint fill, so the photograph underneath stays readable. Five are called out by name as the
   camera tracks across the retina. Their positions are the centroids of the real IDRiD masks (see
   public/retina/stops.json), not wherever happened to look good. */
import { Html } from "@react-three/drei";
import { useFrame, useLoader } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { progress } from "../scroll";
import { irisTexture, planarUVs, scleraTexture } from "./textures";

const R = 1;                     // eyeball radius
const APERTURE = 0.46;           // angular radius of the front opening (radians)
const RETINA_R = 0.92;

/** Lesion layers, drawn in the order a clinician reads them. */
const LESIONS = [
  { file: "ma", at: 0.325, fill: 0.18 },
  { file: "he", at: 0.35, fill: 0.2 },
  { file: "ex", at: 0.375, fill: 0.15 },
] as const;

const COLOUR = {
  ma: "#c2643a",
  he: "#b04d2b",
  ex: "#d8b070",
  od: "#4f9aa8",
} as const;

/** The five annotated stops, left to right across the retina. `at` is where the camera arrives. */
const STOPS = [
  { kind: "he", pos: [-0.684, 0.1637, -0.613], r: 0.05, at: 0.429,
    name: "Haemorrhage", sub: "Superior nasal" },
  { kind: "od", pos: [-0.4779, -0.0833, -0.8017], r: 0.095, at: 0.531,
    name: "Optic disc", sub: "Every quadrant is measured from here" },
  { kind: "ma", pos: [-0.2192, -0.2894, -0.8653], r: 0.022, at: 0.62,
    name: "Microaneurysm", sub: "Inferior temporal · nine pixels wide" },
  { kind: "ex", pos: [0.1447, -0.1272, -0.9196], r: 0.055, at: 0.702,
    name: "Hard exudate", sub: "Inferior temporal" },
  { kind: "he", pos: [0.5233, -0.2002, -0.7497], r: 0.045, at: 0.787,
    name: "Haemorrhage", sub: "Inferior temporal" },
] as const;

function smooth(edge0: number, edge1: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export function Eye() {
  const sclera = useMemo(() => scleraTexture(), []);
  const iris = useMemo(() => irisTexture(), []);
  const fundus = useLoader(THREE.TextureLoader, "/retina/fundus.jpg");
  const fills = useLoader(THREE.TextureLoader, LESIONS.map((l) => `/retina/${l.file}.png`));
  const edges = useLoader(THREE.TextureLoader, [
    ...LESIONS.map((l) => `/retina/${l.file}_edge.png`), "/retina/od_edge.png",
  ]);

  // TextureLoader leaves colorSpace unset, which makes three treat these sRGB files as linear and
  // wash every colour out -- the annotation reds came through as pale grey until this was set.
  useMemo(() => {
    [fundus, ...fills, ...edges].forEach((t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      t.needsUpdate = true;
    });
  }, [fundus, fills, edges]);

  const group = useRef<THREE.Group>(null);
  const corneaMat = useRef<THREE.MeshPhysicalMaterial>(null);
  const irisMat = useRef<THREE.MeshStandardMaterial>(null);
  const scleraMat = useRef<THREE.MeshStandardMaterial>(null);
  const retinaMat = useRef<THREE.MeshStandardMaterial>(null);
  const flash = useRef<THREE.PointLight>(null);
  const fillMats = useRef<(THREE.MeshBasicMaterial | null)[]>([]);
  const edgeMats = useRef<(THREE.MeshBasicMaterial | null)[]>([]);
  const ringMats = useRef<(THREE.MeshBasicMaterial | null)[]>([]);
  const labels = useRef<(HTMLDivElement | null)[]>([]);

  // sphere with the front cap removed, rotated so the opening faces +z
  const scleraGeo = useMemo(() => {
    const g = new THREE.SphereGeometry(R, 96, 96, 0, Math.PI * 2, APERTURE, Math.PI - APERTURE);
    g.rotateX(Math.PI / 2);
    return g;
  }, []);

  const retinaGeo = useMemo(() => {
    const g = new THREE.SphereGeometry(RETINA_R, 128, 128, 0, Math.PI * 2, 0, 0.95);
    g.rotateX(-Math.PI / 2);                       // cap faces the camera, sitting at the back
    return planarUVs(g, RETINA_R * Math.sin(0.95));
  }, []);

  const interiorGeo = useMemo(() => new THREE.SphereGeometry(R * 0.985, 64, 64), []);

  const corneaGeo = useMemo(() => {
    const g = new THREE.SphereGeometry(R * 1.02, 64, 64, 0, Math.PI * 2, 0, APERTURE * 1.5);
    g.rotateX(Math.PI / 2);
    return g;
  }, []);

  useFrame((state) => {
    const p = progress.current;
    // every fade is "on the way in, off again on the way out": one factor rising as we enter the
    // pupil, a second falling as we leave, so nothing is stranded invisible at the end of the page
    const out = smooth(0.88, 0.96, p);
    const veil = (a: number, b: number) => smooth(a, b, p) * (1 - out);
    const inside = veil(0.235, 0.3);                  // 0 outside the eye, 1 within the vitreous

    if (group.current) {
      // a slow, living drift while we are outside; steady once inside
      const idle = (1 - inside) * 0.09;
      group.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.22) * idle;
      group.current.rotation.x = Math.cos(state.clock.elapsedTime * 0.17) * idle * 0.5;
    }
    if (corneaMat.current) corneaMat.current.opacity = 0.4 * (1 - veil(0.17, 0.24));
    if (irisMat.current) irisMat.current.opacity = 1 - veil(0.235, 0.285);
    if (scleraMat.current) scleraMat.current.opacity = 1 - veil(0.26, 0.33);

    // the fundus camera's own light. The retina is black without it, so this both lights the scene
    // and sets the photograph's own brightness; the slow breath keeps it from looking like a decal.
    const lit = veil(0.19, 0.3);
    if (flash.current) {
      flash.current.intensity = lit * 3.1 * (1 + Math.sin(state.clock.elapsedTime * 0.8) * 0.045);
    }
    if (retinaMat.current) retinaMat.current.color.setScalar(0.04 + 0.96 * lit);

    const gone = 1 - smooth(0.83, 0.9, p);
    LESIONS.forEach((l, i) => {
      const on = smooth(l.at, l.at + 0.03, p) * gone;
      const f = fillMats.current[i];
      const e = edgeMats.current[i];
      if (f) f.opacity = l.fill * on;
      if (e) e.opacity = 0.92 * on;
    });
    // the optic disc outline arrives with the stop that names it
    const od = edgeMats.current[LESIONS.length];
    if (od) od.opacity = 0.85 * smooth(0.5, 0.53, p) * gone;

    STOPS.forEach((s, i) => {
      const on = smooth(s.at - 0.04, s.at - 0.008, p) * (1 - smooth(s.at + 0.055, s.at + 0.095, p));
      const m = ringMats.current[i];
      if (m) m.opacity = on;
      const el = labels.current[i];
      if (el) {
        el.style.opacity = String(on);
        el.style.transform = `translate(${(1 - on) * -12}px, -50%)`;
      }
    });
  });

  return (
    <group ref={group}>
      <mesh geometry={scleraGeo}>
        <meshStandardMaterial
          ref={scleraMat}
          map={sclera}
          side={THREE.DoubleSide}
          roughness={0.42}
          metalness={0}
          transparent
        />
      </mesh>

      {/* iris ring: the hole in the middle is the pupil the camera passes through */}
      <mesh position={[0, 0, R * Math.cos(APERTURE) - 0.02]} rotation={[0, 0, 0]}>
        <ringGeometry args={[R * Math.sin(APERTURE) * 0.34, R * Math.sin(APERTURE) * 1.04, 128]} />
        <meshStandardMaterial
          ref={irisMat}
          map={iris}
          side={THREE.DoubleSide}
          roughness={0.55}
          transparent
        />
      </mesh>

      <mesh geometry={corneaGeo}>
        <meshPhysicalMaterial
          ref={corneaMat}
          transparent
          opacity={0.4}
          roughness={0.06}
          metalness={0}
          clearcoat={1}
          clearcoatRoughness={0.04}
          color="#eef6fb"
          side={THREE.FrontSide}
          depthWrite={false}
        />
      </mesh>

      {/* the inside of the eye is dark: without this the pupil shows the lit far wall of the sclera
          and reads as a white hole. Unlit on purpose, so the flash is the only light in here. */}
      <mesh geometry={interiorGeo}>
        <meshBasicMaterial color="#140a06" side={THREE.BackSide} />
      </mesh>

      {/* the real photograph: IDRiD, expert-annotated */}
      <mesh geometry={retinaGeo} position={[0, 0, -0.02]}>
        {/* BackSide: we are inside the eye looking at the back wall, so the faces turned towards
            the camera are the cap's inner ones. FrontSide culls exactly what we came here to see. */}
        <meshStandardMaterial
          ref={retinaMat}
          map={fundus}
          side={THREE.BackSide}
          roughness={0.85}
          toneMapped={false}
        />
      </mesh>

      {LESIONS.map((l, i) => (
        <mesh key={l.file} geometry={retinaGeo} position={[0, 0, -0.014 + 0.003 * i]}>
          <meshBasicMaterial
            ref={(m) => { fillMats.current[i] = m; }}
            map={fills[i]}
            side={THREE.BackSide}
            transparent
            opacity={0}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      ))}

      {[...LESIONS.map((l) => l.file), "od"].map((file, i) => (
        <mesh key={`${file}-edge`} geometry={retinaGeo} position={[0, 0, -0.004 + 0.003 * i]}>
          <meshBasicMaterial
            ref={(m) => { edgeMats.current[i] = m; }}
            map={edges[i]}
            side={THREE.BackSide}
            transparent
            opacity={0}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      ))}

      {STOPS.map((s, i) => (
        <group key={`${s.name}-${i}`} position={[s.pos[0], s.pos[1], s.pos[2]]}>
          <mesh renderOrder={3}>
            <ringGeometry args={[s.r, s.r * 1.13, 64]} />
            <meshBasicMaterial
              ref={(m) => { ringMats.current[i] = m; }}
              color={COLOUR[s.kind]}
              transparent
              opacity={0}
              depthTest={false}
              side={THREE.DoubleSide}
              toneMapped={false}
            />
          </mesh>
          <Html position={[s.r * 1.25, 0, 0]} style={{ pointerEvents: "none" }} zIndexRange={[3, 0]}>
            <div
              className="mk"
              ref={(el) => { labels.current[i] = el; }}
              style={{ color: COLOUR[s.kind], opacity: 0 }}
            >
              <i className="lead" />
              <span className="chip">
                <b>{s.name}</b>
                <small>{s.sub}</small>
              </span>
            </div>
          </Html>
        </group>
      ))}

      <pointLight ref={flash} position={[0, 0, 0.55]} intensity={0} distance={4} color="#fff3dd" />
    </group>
  );
}
