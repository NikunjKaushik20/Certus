/* Capture: the page someone stands at with a patient in the chair.

   It runs in the order the room runs in. Find the person, record consent, open the visit, photograph
   each eye, then grade. The quality gate is the reason the page exists: a retake is cheap while the
   patient is still sitting there and impossible once they have gone home. */
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { abstainText } from "../abstain";
import { ApiError, api, postForm } from "../api";
import { useAuthedImage } from "../useAuthedImage";

type Site = { id: string; name: string; district: string; state: string; kind: string };
type Device = { id: string; site_id: string; make: string; model: string; domain_key: string };
type Patient = { id: string; pseudo_id: string; sex: string; birth_year: number | null; created_at: string };
type Encounter = { id: string; patient_id: string; site_id: string; status: string };
type ImageRow = { id: string; eye: string; field: string; sha256: string; fov_ok: boolean; canvas: number };
type Job = { id: string; image_id: string; state: string; attempts: number; error: string | null;
             runtime_ms: number | null; executor: string | null };
type Report = {
  encounter: Encounter;
  images: ImageRow[];
  quality: Record<string, Quality | null>;
  predictions: Record<string, Prediction | null>;
  worst_grade: number | null;
  decision: string;
  referral: { id: string; priority: string; reason: string } | null;
};

const GRADE = ["No retinopathy", "Mild", "Moderate", "Severe", "Proliferative"];

/* Demo data. It fills the same fields anyone else would type and uploads through the same endpoint,
   so a judge can press the button, change the screening id, swap in their own photographs, and the
   page behaves identically. The two images are IDRiD_55 and IDRiD_56 at their original 4288x2848,
   from the segmentation test split, so they are photographs the model was never trained on. */
const DEMO_PHOTOS = { R: "/demo/right.jpg", L: "/demo/left.jpg" } as const;
const LESION = { MA: "Microaneurysms", HE: "Haemorrhages", EX: "Hard exudates", SE: "Soft exudates" } as const;
type Quality = { label: string; probs: Record<string, any>; gate_action: string };
type Finding = {
  id: string; lesion_type: string; threshold: number; lesion_count: number;
  pixel_area: number; area_frac: number; peak_prob: number; quadrant_counts: Record<string, number>;
};
type Prediction = {
  id: string; image_id: string; dr_grade: number; ordinal_probs: number[]; p_referable: number;
  p_referable_raw: number; decision: string; abstained: boolean; abstain_reason: string | null;
  threshold_used: number | null; runtime_ms: number | null; executor: string | null;
  evidence: number[]; attention: number[]; findings: Finding[];
};

function Eye({
  eye, image, quality, prediction, tick,
}: {
  eye: string; image: ImageRow | undefined; quality: Quality | null | undefined;
  prediction: Prediction | null | undefined; tick: number;
}) {
  const [mask, setMask] = useState<string | null>(null);
  const photo = useAuthedImage(image ? `/v1/images/${image.id}/file` : null, tick);
  const overlay = useAuthedImage(mask ? `/v1/findings/${mask}/mask` : null);

  const retake = quality?.gate_action === "recapture_requested";
  const adviceList: string[] = quality?.probs?.recapture_advice || [];

  return (
    <div className={`eye-panel ${retake ? "retake" : ""}`}>
      <header>
        <b>{eye === "L" ? "Left eye" : "Right eye"}</b>
        {quality && <span className={`tag q-${quality.label}`}>{quality.label}</span>}
      </header>

      <div className="shot">
        {photo
          ? <img src={photo} alt={`${eye} fundus`} />
          : <div className="empty">{image ? "waiting to be graded" : "no image yet"}</div>}
        {overlay && <img className="overlay" src={overlay} alt="lesion mask" />}
      </div>

      {retake && (
        <div className="alert">
          <b>Take this one again.</b> Image quality is inadequate for safe automated grading.
          {adviceList.length > 0 && (
            <div style={{ marginTop: "0.4rem", fontSize: "0.85rem", lineHeight: "1.3" }}>
              {adviceList.map((adv, i) => (
                <div key={i} style={{ marginTop: "0.2rem" }}>&#x2022; {adv}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {prediction && (
        <>
          <div className="readout">
            <div>
              <dt>Grade</dt>
              <dd>{prediction.dr_grade}<small>{GRADE[prediction.dr_grade]}</small></dd>
            </div>
            <div>
              <dt>Referable</dt>
              <dd>{(prediction.p_referable * 100).toFixed(1)}%<small>
                raw {(prediction.p_referable_raw * 100).toFixed(1)}%</small></dd>
            </div>
            <div>
              <dt>Decision</dt>
              <dd className={`d-${prediction.decision}`}>
                {prediction.decision}
                <small>{prediction.runtime_ms ? `${prediction.runtime_ms} ms · ${prediction.executor}` : ""}</small>
              </dd>
            </div>
          </div>

          {prediction.abstained && (
            <div className="alert soft">
              <b>Sent to a reviewer.</b> {abstainText(prediction.abstain_reason)}
            </div>
          )}

          <table className="data compact">
            <thead>
              <tr><th>Lesion</th><th className="n">Regions</th><th className="n">Peak</th><th className="n">Threshold</th><th /></tr>
            </thead>
            <tbody>
              {prediction.findings.map((f) => (
                <tr key={f.id}>
                  <td>{LESION[f.lesion_type as keyof typeof LESION] ?? f.lesion_type}</td>
                  <td className="n">{f.lesion_count}</td>
                  <td className="n">{f.peak_prob.toFixed(3)}</td>
                  <td className="n">{f.threshold.toFixed(3)}</td>
                  <td className="n">
                    <button
                      className="link"
                      onClick={() => setMask(mask === f.id ? null : f.id)}
                    >
                      {mask === f.id ? "hide" : "show"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small dim" style={{ marginBottom: 0 }}>
            Regions are separate blobs above the threshold for that lesion, which is close to a lesion
            count but not the same thing. Confluent haemorrhages split into several; dense exudates
            merge into one. Treat the number as a magnitude and the overlay as the evidence.
          </p>
        </>
      )}
    </div>
  );
}

export default function Technician() {
  const [sites, setSites] = useState<Site[]>([]);
  const [devices, setDevices] = useState<Device[]>([]);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [siteId, setSiteId] = useState("");
  const [deviceId, setDeviceId] = useState("");
  const [patient, setPatient] = useState<Patient | null>(null);
  const [encounter, setEncounter] = useState<Encounter | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [tick, setTick] = useState(0);   // bumped on every refresh so processed images are refetched
  const [jobs, setJobs] = useState<Job[]>([]);
  const lastJobSig = useRef("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [draft, setDraft] = useState({ pseudo_id: "", sex: "unknown", birth_year: "", diabetes_years: "" });

  useEffect(() => {
    api<Site[]>("/v1/sites").then((s) => { setSites(s); setSiteId((v) => v || s[0]?.id || ""); }).catch(() => {});
    api<Device[]>("/v1/devices").then(setDevices).catch(() => {});
    api<Patient[]>("/v1/patients").then(setPatients).catch(() => {});
  }, []);

  /* Uploading queues a job; the worker grades it on its own. Without this the page sat on an empty
     panel with no way to tell a slow grade from a stuck one, which is exactly what it looked like. */
  useEffect(() => {
    if (!encounter) { setJobs([]); return; }
    let alive = true;
    let timer = 0;
    const poll = async () => {
      try {
        const js = await api<Job[]>(`/v1/encounters/${encounter.id}/jobs`);
        if (!alive) return;
        setJobs(js);
        const pending = js.some((j) => j.state === "queued" || j.state === "running");
        // Refresh whenever the set of job states changes and nothing is left running. Comparing a
        // signature rather than a "was pending" flag also catches the case where the page opens
        // after the work already finished.
        const sig = js.map((j) => `${j.id}:${j.state}`).sort().join("|");
        if (!pending && sig !== lastJobSig.current && js.length) {
          lastJobSig.current = sig;
          refresh();
        } else if (!pending) {
          lastJobSig.current = sig;
        }
        timer = window.setTimeout(poll, pending ? 1500 : 5000);
      } catch {
        if (alive) timer = window.setTimeout(poll, 4000);
      }
    };
    poll();
    return () => { alive = false; window.clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [encounter]);

  async function run<T>(tag: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(tag);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      const err = e as ApiError;
      setError(err.unreachable ? err.message : `${err.status}: ${err.message}`);
      return undefined;
    } finally {
      setBusy(null);
    }
  }

  function fillDemoPatient() {
    const stamp = new Date().toISOString().slice(2, 10).replace(/-/g, "");
    setDraft({
      pseudo_id: `SIT-${stamp}-${String(Math.floor(Math.random() * 900) + 100)}`,
      sex: "female",
      birth_year: "1971",
      diabetes_years: "12",
    });
  }

  async function registerPatient() {
    const p = await run("patient", () => api<Patient>("/v1/patients", {
      method: "POST",
      body: {
        pseudo_id: draft.pseudo_id.trim(),
        sex: draft.sex,
        birth_year: draft.birth_year ? Number(draft.birth_year) : null,
        diabetes_years: draft.diabetes_years ? Number(draft.diabetes_years) : null,
      },
    }));
    if (!p) return;
    setPatient(p);
    setPatients((all) => (all.some((x) => x.id === p.id) ? all : [p, ...all]));
  }

  async function openEncounter() {
    if (!patient || !siteId) return;
    // Consent first. The server refuses to open a visit without one, which is the right answer; the
    // page just makes the order obvious instead of surfacing a 409.
    await run("consent", () => api("/v1/consents", {
      method: "POST", body: { patient_id: patient.id, scope: "screening", method: "abdm" },
    }));
    const enc = await run("encounter", () => api<Encounter>("/v1/encounters", {
      method: "POST", body: { patient_id: patient.id, site_id: siteId },
    }));
    if (enc) { setEncounter(enc); setReport(null); }
  }

  async function upload(eye: "L" | "R", file: File) {
    if (!encounter) return;
    const form = new FormData();
    form.append("eye", eye);
    form.append("field", "macula");
    if (deviceId) form.append("device_id", deviceId);
    form.append("file", file);
    await run(`upload-${eye}`, () => postForm(`/v1/encounters/${encounter.id}/images`, form));
    await refresh();
  }

  async function useDemoPhotos() {
    if (!encounter) return;
    for (const eye of ["R", "L"] as const) {
      const res = await fetch(DEMO_PHOTOS[eye]);
      const blob = await res.blob();
      await upload(eye, new File([blob], `${eye}-demo.jpg`, { type: "image/jpeg" }));
    }
  }

  async function refresh() {
    if (!encounter) return;
    const r = await run("report", () => api<Report>(`/v1/encounters/${encounter.id}/report`));
    if (r) { setReport(r); setTick((n) => n + 1); }
  }

  async function analyse() {
    if (!encounter) return;
    await run("analyse", () => api(`/v1/encounters/${encounter.id}/analyse`, { method: "POST" }));
    await refresh();
  }

  const shown = patients.filter((p) => p.pseudo_id.toLowerCase().includes(search.toLowerCase()));
  const byEye = (eye: string) => report?.images.find((i) => i.eye === eye);

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Capture</div>
          <h1>One visit at a time.</h1>
          <p className="dim">
            Work down the page. Nothing is graded until both eyes are photographed, and a photograph
            the model cannot read comes back as a retake rather than a grade.
          </p>
        </div>
        <div className="picker">
          <label>Site</label>
          <select value={siteId} onChange={(e) => setSiteId(e.target.value)}>
            {sites.map((s) => <option key={s.id} value={s.id}>{s.name} · {s.district}</option>)}
          </select>
          <label>Camera</label>
          <select value={deviceId} onChange={(e) => setDeviceId(e.target.value)}>
            <option value="">not recorded</option>
            {devices.filter((d) => !siteId || d.site_id === siteId).map((d) => (
              <option key={d.id} value={d.id}>{d.make} {d.model} · {d.domain_key}</option>
            ))}
          </select>
        </div>
      </header>

      {error && <div className="error" role="alert">{error}</div>}

      {/* ---------------------------------------------------------------- 1 */}
      <section className="step">
        <div className="step-n">1</div>
        <div className="step-body">
          <h2>Who is in the chair</h2>
          <div className="two">
            <div>
              <label>Find someone already registered</label>
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="screening id"
              />
              <ul className="people">
                {shown.slice(0, 6).map((p) => (
                  <li key={p.id}>
                    <button
                      className={patient?.id === p.id ? "on" : ""}
                      onClick={() => { setPatient(p); setEncounter(null); setReport(null); }}
                    >
                      <b>{p.pseudo_id}</b>
                      <span>{p.sex}{p.birth_year ? ` · born ${p.birth_year}` : ""}</span>
                    </button>
                  </li>
                ))}
                {shown.length === 0 && <li className="none">No match. Register them on the right.</li>}
              </ul>
            </div>
            <div>
              <div className="label-row">
                <label>Or register a new one</label>
                <button className="demo-fill" onClick={fillDemoPatient} type="button">Use demo data</button>
              </div>
              <input
                value={draft.pseudo_id}
                onChange={(e) => setDraft({ ...draft, pseudo_id: e.target.value })}
                placeholder="screening id, e.g. SIT-2026-0141"
              />
              <div className="row">
                <select value={draft.sex} onChange={(e) => setDraft({ ...draft, sex: e.target.value })}>
                  <option value="unknown">sex not recorded</option>
                  <option value="female">female</option>
                  <option value="male">male</option>
                  <option value="other">other</option>
                </select>
                <input
                  value={draft.birth_year}
                  onChange={(e) => setDraft({ ...draft, birth_year: e.target.value })}
                  placeholder="birth year"
                  inputMode="numeric"
                />
                <input
                  value={draft.diabetes_years}
                  onChange={(e) => setDraft({ ...draft, diabetes_years: e.target.value })}
                  placeholder="years with diabetes"
                  inputMode="decimal"
                />
              </div>
              <button onClick={registerPatient} disabled={!draft.pseudo_id.trim() || busy !== null}>
                {busy === "patient" ? "Saving…" : "Register"}
              </button>
              <p className="small dim">
                No name, no ABHA number in the clear. The screening id is what the camp already uses,
                and anything identifying is hashed before it is stored.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------- 2 */}
      <section className={`step ${patient ? "" : "off"}`}>
        <div className="step-n">2</div>
        <div className="step-body">
          <h2>Consent, then open the visit</h2>
          <p className="dim">
            {patient
              ? <>Screening <b>{patient.pseudo_id}</b> at {sites.find((s) => s.id === siteId)?.name ?? "this site"}.</>
              : "Pick someone first."}
          </p>
          <button onClick={openEncounter} disabled={!patient || !siteId || busy !== null}>
            {busy === "encounter" || busy === "consent" ? "Opening…" : "Record consent and open"}
          </button>
          {encounter && <code className="id">visit {encounter.id.slice(0, 8)}</code>}
        </div>
      </section>

      {/* ---------------------------------------------------------------- 3 */}
      <section className={`step ${encounter ? "" : "off"}`}>
        <div className="step-n">3</div>
        <div className="step-body">
          <h2>Photograph both eyes</h2>
          <div className="row uploads">
            {(["R", "L"] as const).map((eye) => (
              <label key={eye} className="drop">
                <b>{eye === "L" ? "Left eye" : "Right eye"}</b>
                <span>{byEye(eye) ? "uploaded" : "choose a file"}</span>
                <input
                  type="file"
                  accept="image/*"
                  disabled={!encounter || busy !== null}
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(eye, f); }}
                />
              </label>
            ))}
          </div>
          <div className="row">
            <button className="demo-fill" onClick={useDemoPhotos} disabled={!encounter || busy !== null}>
              Use demo photographs
            </button>
          </div>
          {report?.images.length ? (
            <p className="job-line">
              {jobs.length === 0
                ? `${report.images.length} photograph${report.images.length > 1 ? "s" : ""} ready · not graded yet`
                : `${jobs.filter((j) => j.state === "done").length} of ${jobs.length} graded`}
              {jobs.some((j) => j.state === "running") && " · one running now"}
              {jobs.some((j) => j.state === "queued") && ` · ${jobs.filter((j) => j.state === "queued").length} waiting`}
              {jobs.some((j) => j.error) && ` · ${jobs.filter((j) => j.error).length} failed`}
            </p>
          ) : null}
          <div className="row">
            <button onClick={analyse} disabled={!report?.images.length || busy !== null}>
              {busy === "analyse" ? "Grading…" : "Grade this visit now"}
            </button>
            <button className="ghost" onClick={refresh} disabled={!encounter || busy !== null}>
              Refresh
            </button>
          </div>
          <p className="small dim">
            Uploading stores the photograph. Nothing is graded until you press the button, so you can
            retake an eye before spending the compute. A retake replaces that eye rather than adding
            to the visit. Grading runs on whichever device the server was started with, and on one
            GPU that is the same card a training run would be using.
          </p>
        </div>
      </section>

      {/* ---------------------------------------------------------------- 4 */}
      {report && (
        <section className="step">
          <div className="step-n">4</div>
          <div className="step-body">
            <div className="panel-head">
              <h2>What the model found</h2>
              <Link className="sheet-link" to={`/app/result/${report.encounter.id}`}>Result sheet →</Link>
            </div>
            <div className="verdict">
              <div>
                <dt>Visit decision</dt>
                <dd className={`d-${report.decision}`}>{report.decision}</dd>
              </div>
              <div>
                <dt>Worst grade</dt>
                <dd>{report.worst_grade ?? "—"}<small>{report.worst_grade != null ? GRADE[report.worst_grade] : "not graded"}</small></dd>
              </div>
              <div>
                <dt>Referral</dt>
                <dd>{report.referral ? report.referral.priority : "none"}<small>{report.referral?.reason ?? ""}</small></dd>
              </div>
            </div>
            <div className="eyes">
              {(["R", "L"] as const).map((eye) => {
                const img = byEye(eye);
                return (
                  <Eye
                    key={eye}
                    eye={eye}
                    image={img}
                    quality={img ? report.quality[img.id] : null}
                    prediction={img ? report.predictions[img.id] : null}
                    tick={tick}
                  />
                );
              })}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
