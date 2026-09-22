/* One source of truth for "where are we". Lenis smooths the wheel, writes a 0..1 number into a ref,
   and everything else — camera, lesion opacity, the depth readout — reads that same ref inside the
   render loop. Nothing animates off React state, so scrolling never triggers a re-render. */
import Lenis from "lenis";

export const progress = { current: 0 };

type Listener = (p: number) => void;
const listeners = new Set<Listener>();

/** For the few bits of DOM chrome that do need to update (progress bar, depth label). */
export function onProgress(fn: Listener) {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

export function startScroll() {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const lenis = new Lenis({ duration: reduced ? 0 : 1.15, smoothWheel: !reduced });

  // Lenis owns the scroll position: a plain window.scrollTo is overwritten on the next frame. Expose
  // it in dev so the page can be driven to an exact point while tuning the camera path.
  if (import.meta.env.DEV) (window as unknown as { __lenis: Lenis }).__lenis = lenis;

  let last = -1;
  const update = (p: number) => {
    const v = Math.min(1, Math.max(0, p));
    if (Math.abs(v - last) < 0.0004) return;
    last = v;
    progress.current = v;
    listeners.forEach((fn) => fn(v));
  };

  // Read the real scroll position every frame rather than trusting Lenis's own event. Anything that
  // moves the page without going through Lenis -- an anchor jump, find-in-page, the browser
  // restoring a position on reload -- still has to move window.scrollY, so this never desynchronises.
  let raf = 0;
  const loop = (t: number) => {
    lenis.raf(t);
    const limit = document.documentElement.scrollHeight - window.innerHeight;
    update(limit > 0 ? window.scrollY / limit : 0);
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);

  return () => {
    cancelAnimationFrame(raf);
    lenis.destroy();
  };
}

/** Where the camera is, in words. Shown in the corner readout. */
export function depthLabel(p: number) {
  if (p < 0.067) return ["the eye", "0 mm"];
  if (p < 0.17) return ["tear film", "≈ 0.1 mm"];
  if (p < 0.235) return ["cornea · iris", "≈ 0.5 mm"];
  if (p < 0.285) return ["pupil", "≈ 3 mm"];
  if (p < 0.33) return ["vitreous", "≈ 17 mm"];
  if (p < 0.62) return ["retina · lesions", "≈ 24 mm"];
  if (p < 0.88) return ["evidence", "—"];
  return ["decision", "—"];
}
