/* Review queue: the page an ophthalmologist works through between clinics.

   The queue is on the left in the order the SLA demands, urgent first. Open a case and the model's
   grade arrives with the evidence it was built from, so the question is never "do you trust this",
   it is "do you agree with this". Keeping or changing the grade takes one click either way, and both
   are recorded. */
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { abstainText } from "../abstain";
import { ApiError, api, authedBlob } from "../api";
import { useAuthedImage } from "../useAuthedImage";

type Referral = {
  id: string; encounter_id: string; priority: string; state: string; reason: string;
  worst_grade: number | null; assigned_to: string | null; sla_due_at: string;
  created_at: string; closed_at: string | null;
};
type Stats = {
  by_state: { state: string; priority: string; n: number }[];
  open_past_sla: number;
  closed: number;
  median_turnaround_hours: number | null;
  within_48h_fraction: number | null;
};
type Finding = {
  id: string; type: string; count: number; threshold: number; peak_prob: number;
  quadrants: Record<string, number>; mask_url: string;
};
type StructureSummary = {
  disc_found?: boolean;
  disc_diameter_px?: number;
  fovea_found?: boolean;
  macula_radius_px?: number;
  optic_disc?: { found: boolean; diameter_px: number };
  fovea?: { found: boolean; macula_radius_px: number };
  vessels?: { density_pct: number; vessel_area_px: number };
  neovascularization?: {
    nvd_detected: boolean;
    nve_detected: boolean;
    nv_risk_score: number;
    criteria: string;
  };
  enhancement?: {
    applied: boolean;
    focus_score?: number;
    recapture_advice?: string;
  };
  gradcam_uri?: string;
  second_reader?: { logit: number; p_referable: number };
};
type EyeCtx = {
  image_id: string; eye: string; field: string;
  prediction: {
    id: string; dr_grade: number; decision: string; p_referable: number; abstained: boolean;
    abstain_reason: string | null; attention: number[]; evidence: number[]; findings: Finding[];
    structures?: StructureSummary | null;
  } | null;
};
type Context = {
  referral: {
    id: string; priority: string; reason: string; state: string; sla_due_at: string;
    worst_grade: number | null; hours_remaining: number;
  };
  encounter: { id: string; patient_id: string; site_id: string };
  eyes: EyeCtx[];
};
type User = { id: string; name: string; role: string; site_id: string | null };

const GRADE = ["No retinopathy", "Mild", "Moderate", "Severe", "Proliferative"];
const LESION = { MA: "Microaneurysms", HE: "Haemorrhages", EX: "Hard exudates", SE: "Soft exudates" } as const;
const QUADRANT = ["superior_temporal", "superior_nasal", "inferior_temporal", "inferior_nasal"];

function EyeCard({ eye }: { eye: EyeCtx }) {
  const [mask, setMask] = useState<string | null>(null);
  const [gcamOn, setGcamOn] = useState(false);
  const [gcamUrl, setGcamUrl] = useState<string | null>(null);
  // 'idle' | 'loading' | 'done' | 'error'
  const [gcamState, setGcamState] = useState<string>("idle");
  const [gcamError, setGcamError] = useState<string | null>(null);
  const photo   = useAuthedImage(`/v1/images/${eye.image_id}/file`);
  const overlay = useAuthedImage(mask ? `/v1/findings/${mask}/mask` : null);
  const p = eye.prediction;

  // The worker renders the referable Grad-CAM for every eye sent to a person
  // (structures.gradcam_uri), so fetch it with the case: the toggle is then instant
  // instead of a 15-30 s CPU wait inside a 30-second review.
  useEffect(() => {
    if (!p?.structures?.gradcam_uri) return;
    let url: string | null = null;
    let live = true;
    authedBlob(`/v1/predictions/${p.id}/gradcam`)
      .then((u) => { url = u; if (live) { setGcamUrl(u); setGcamState("done"); } else URL.revokeObjectURL(u); })
      .catch(() => { /* the toggle retries on demand and reports the error */ });
    return () => { live = false; if (url) URL.revokeObjectURL(url); };
  }, [p?.id, p?.structures?.gradcam_uri]);

  function toggleGcam() {
    if (gcamOn) {
      setGcamOn(false);
      return;
    }
    if (gcamUrl) {
      setGcamOn(true);
      return;
    }
    if (!p) return;
    setGcamOn(true);
    setGcamState("loading");
    setGcamError(null);
    // Use authedBlob directly so we get a real error if the API call fails.
    // useAuthedImage silently swallows errors (sets url=null), which left the
    // spinner stuck forever when the request returned a non-200.
    // Not precomputed (auto-cleared eye): 9 tile backward passes, ~3-5 s on GPU, ~15-30 s on CPU.
    authedBlob(`/v1/predictions/${p.id}/gradcam`)
      .then((url) => { setGcamUrl(url); setGcamState("done"); })
      .catch((e) => {
        setGcamOn(false);
        setGcamState("error");
        setGcamError(e?.message ?? "Grad-CAM failed");
      });
  }

  const displayImg = gcamOn && gcamUrl ? gcamUrl : photo;

  return (
    <div className="eye-panel">
      <header>
        <b>{eye.eye === "L" ? "Left eye" : "Right eye"}</b>
        <div style={{ display: "flex", gap: "0.35rem", alignItems: "center" }}>
          {p?.structures?.enhancement?.applied && (
            <span className="tag" style={{ background: "#e0f2fe", color: "#0369a1", fontSize: "0.75rem" }} title="Enhanced with Green-channel CLAHE & Bilateral Edge-Preserving Denoising">
              Enhanced
            </span>
          )}
          {p && <span className={`tag d-${p.decision}`}>{p.decision}</span>}
        </div>
      </header>

      <div className="shot">
        {displayImg
          ? <img src={displayImg} alt={`${eye.eye} fundus${gcamOn ? " Grad-CAM" : ""}`} />
          : <div className="empty">loading</div>}
        {!gcamOn && overlay && <img className="overlay" src={overlay} alt="lesion mask" />}
      </div>

      {p && (
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", margin: "0.35rem 0 0.1rem" }}>
          <button
            id={`gcam-btn-${eye.image_id}`}
            className={`link small${gcamOn ? " on" : ""}`}
            onClick={toggleGcam}
            disabled={gcamState === "loading"}
            title="Grad-CAM: highlights the pixels that drove the referable score (precomputed for eyes sent to review)"
          >
            {gcamState === "loading"
              ? "computing Grad-CAM\u2026"
              : gcamOn
              ? "hide Grad-CAM \u2715"
              : "show Grad-CAM \u25a6"}
          </button>
          {gcamState === "error" && (
            <span className="small" style={{ color: "var(--bad, #c00)" }} title={gcamError ?? ""}>
              Grad-CAM failed \u2014 check API logs
            </span>
          )}
        </div>
      )}

      {!p && <p className="dim small">No prediction on this image.</p>}

      {p && (
        <>
          <div className="readout">
            <div><dt>Grade</dt><dd>{p.dr_grade}<small>{GRADE[p.dr_grade]}</small></dd></div>
            <div><dt>Referable</dt><dd>{(p.p_referable * 100).toFixed(1)}%</dd></div>
          </div>

          {p.structures?.vessels && (
            <div className="readout" style={{ marginTop: "0.25rem", borderTop: "1px dashed var(--edge, #ddd)", paddingTop: "0.25rem" }}>
              <div><dt>Vessel Area</dt><dd>{p.structures.vessels.density_pct}%</dd></div>
              <div><dt>Optic Disc</dt><dd>{p.structures.optic_disc?.found || p.structures.disc_found ? `Found (${Math.round(p.structures.optic_disc?.diameter_px ?? p.structures.disc_diameter_px ?? 0)}px)` : "Not found"}</dd></div>
            </div>
          )}

          {p.structures?.neovascularization && (
            p.structures.neovascularization.nvd_detected || p.structures.neovascularization.nve_detected ? (
              <div className="alert warn" style={{ background: "#fff0f0", border: "1px solid #f5c2c2", padding: "0.4rem 0.6rem", borderRadius: "4px", margin: "0.4rem 0", fontSize: "0.82rem" }}>
                <b style={{ color: "#c00" }}>⚠️ Neovascularization (NV):</b> {p.structures.neovascularization.criteria} (Risk: {p.structures.neovascularization.nv_risk_score.toFixed(2)})
              </div>
            ) : (
              <div style={{ fontSize: "0.78rem", color: "#2e7d32", margin: "0.3rem 0" }}>
                ✓ No Neovascularization detected (NV Risk: {p.structures.neovascularization.nv_risk_score.toFixed(2)})
              </div>
            )
          )}

          {p.abstained && (
            <div className="alert soft"><b>The model would not call this one.</b> {abstainText(p.abstain_reason)}</div>
          )}
          {p.structures?.second_reader && (
            <div style={{ fontSize: "0.78rem", color: "var(--muted, #666)", margin: "0.3rem 0" }}>
              Second reader (whole-image grader, no explanation): p(referable){" "}
              {(p.structures.second_reader.p_referable * 100).toFixed(1)}%
            </div>
          )}

          <table className="data compact">
            <thead>
              <tr>
                <th>Lesion</th><th className="n">Regions</th>
                <th className="n">ST</th><th className="n">SN</th>
                <th className="n">IT</th><th className="n">IN</th><th />
              </tr>
            </thead>
            <tbody>
              {p.findings.map((f) => (
                <tr key={f.id}>
                  <td>{LESION[f.type as keyof typeof LESION] ?? f.type}</td>
                  <td className="n">{f.count}</td>
                  {QUADRANT.map((q) => <td key={q} className="n">{f.quadrants?.[q] ?? 0}</td>)}
                  <td className="n">
                    <button className="link" onClick={() => setMask(mask === f.id ? null : f.id)}>
                      {mask === f.id ? "hide" : "show"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small dim" style={{ marginBottom: 0 }}>
            A region is a separate blob above the calibrated threshold for that lesion, and the
            quadrant it falls in is worked out from the optic disc in this image. Confluent
            haemorrhages split into several regions and dense exudates merge into one, so read the
            number as a magnitude and the overlay as the evidence.
          </p>
        </>
      )}
    </div>
  );
}

export default function Review() {
  const [refs, setRefs] = useState<Referral[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [agree, setAgree] = useState<{ n: number; agreement_rate?: number; median_seconds_per_review?: number | null } | null>(null);
  const [reviewers, setReviewers] = useState<User[]>([]);
  const [reviewerId, setReviewerId] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);
  const [ctx, setCtx] = useState<Context | null>(null);
  const [filter, setFilter] = useState<string>("open");
  const [notes, setNotes] = useState("");
  const [finalGrade, setFinalGrade] = useState<number | null>(null);
  // Optional explanation ratings for the reader study (study/PROTOCOL.md). Blank = not rated.
  const [rating, setRating] = useState<{ gradcam_useful?: number; lesions_useful?: number; heatmap_on_lesions?: string }>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number>(Date.now());

  const loadQueue = useCallback(async () => {
    try {
      const all = await api<Referral[]>("/v1/referrals?limit=200");
      setRefs(all);
      setStats(await api<Stats>("/v1/referrals/stats"));
      setAgree(await api("/v1/reviews/agreement"));
    } catch (e) {
      setError((e as ApiError).message);
    }
  }, []);

  useEffect(() => {
    loadQueue();
    api<User[]>("/v1/users?role=ophthalmologist")
      .then((u) => { setReviewers(u); setReviewerId((v) => v || u[0]?.id || ""); })
      .catch(() => {});
  }, [loadQueue]);

  async function open(id: string) {
    setOpenId(id);
    setCtx(null);
    setNotes("");
    setFinalGrade(null);
    setRating({});
    setStartedAt(Date.now());
    setError(null);
    try {
      const c = await api<Context>(`/v1/referrals/${id}/context`);
      setCtx(c);
      setFinalGrade(c.referral.worst_grade);
    } catch (e) {
      setError((e as ApiError).message);
    }
  }

  /* Opens the first case in the list and fills the decision box. Same fields, same endpoint: change
     the grade or rewrite the note and it submits exactly the same way. */
  async function fillDemoReview() {
    const target = visible[0] ?? refs[0];
    if (!target) { setError("Nothing in the queue to review."); return; }
    if (openId !== target.id) await open(target.id);
    setNotes("Both eyes reviewed. Haemorrhages and exudates present outside the macula; grade agrees with the model.");
    setFinalGrade(target.worst_grade ?? 2);
  }

  async function submit(agreed: boolean) {
    if (!ctx || !reviewerId) return;
    setBusy("submit");
    setError(null);
    try {
      await api(`/v1/referrals/${ctx.referral.id}/review?reviewer_id=${encodeURIComponent(reviewerId)}`, {
        method: "POST",
        body: {
          agreed,
          final_grade: agreed ? ctx.referral.worst_grade : finalGrade,
          notes,
          seconds_spent: Math.max(1, Math.round((Date.now() - startedAt) / 1000)),
          explanation: Object.keys(rating).length ? rating : null,
        },
      });
      setOpenId(null);
      setCtx(null);
      await loadQueue();
    } catch (e) {
      const err = e as ApiError;
      setError(`${err.status}: ${err.message}`);
    } finally {
      setBusy(null);
    }
  }

  const visible = refs.filter((r) => (filter === "all" ? true : filter === "open" ? r.state !== "closed" : r.state === filter));
  const openCount = refs.filter((r) => r.state !== "closed").length;

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Review queue</div>
          <h1>The cases the model would not close on its own.</h1>
          <p className="dim">
            Referable eyes and the ones it refused to call, urgent first, then by the deadline. The
            evidence comes with the case, so you are checking a grade rather than reading a scan from
            scratch.
          </p>
        </div>
      </header>

      {error && <div className="error" role="alert">{error}</div>}

      <div className="row">
        <button className="demo-fill" onClick={fillDemoReview} disabled={busy !== null}>
          Open a demo case
        </button>
      </div>

      <div className="tiles">
        <div className="tile"><dt>Open</dt><dd>{openCount}</dd></div>
        <div className="tile"><dt>Past the 48-hour mark</dt>
          <dd className={stats?.open_past_sla ? "bad" : ""}>{stats?.open_past_sla ?? "—"}</dd></div>
        <div className="tile"><dt>Median turnaround</dt>
          <dd>{stats?.median_turnaround_hours != null ? `${stats.median_turnaround_hours} h` : "—"}</dd></div>
        <div className="tile"><dt>Closed within 48 h</dt>
          <dd>{stats?.within_48h_fraction != null ? `${Math.round(stats.within_48h_fraction * 100)}%` : "—"}</dd></div>
        <div className="tile"><dt>Kept the model's grade</dt>
          <dd>{agree?.n ? `${Math.round((agree.agreement_rate ?? 0) * 100)}%` : "no reviews yet"}
            <small>{agree?.n ? `${agree.n} reviews` : ""}</small></dd></div>
      </div>

      <div className="split">
        <aside className="queue">
          <div className="queue-head">
            <span>{visible.length} case{visible.length === 1 ? "" : "s"}</span>
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="open">open</option>
              <option value="assigned">assigned</option>
              <option value="closed">closed</option>
              <option value="all">all</option>
            </select>
          </div>
          <ul>
            {visible.map((r) => {
              const due = new Date(r.sla_due_at + (r.sla_due_at.endsWith("Z") ? "" : "Z"));
              const hours = (due.getTime() - Date.now()) / 36e5;
              return (
                <li key={r.id}>
                  <button className={openId === r.id ? "on" : ""} onClick={() => open(r.id)}>
                    <div className="row1">
                      <b className={`pri ${r.priority}`}>{r.priority}</b>
                      <span className="grade">grade {r.worst_grade ?? "—"}</span>
                    </div>
                    <div className="row2">{r.reason}</div>
                    <div className="row3">
                      {r.state === "closed"
                        ? "closed"
                        : hours < 0 ? `${Math.abs(hours).toFixed(0)} h overdue` : `${hours.toFixed(0)} h left`}
                    </div>
                  </button>
                </li>
              );
            })}
            {visible.length === 0 && <li className="none">Nothing in this state.</li>}
          </ul>
        </aside>

        <div className="case">
          {!ctx && <div className="placeholder">Pick a case on the left.</div>}
          {ctx && (
            <>
              <div className="case-head">
                <div>
                  <h2>Case {ctx.referral.id.slice(0, 8)}</h2>
                  <p className="dim small">
                    {ctx.referral.reason} · {ctx.referral.priority} ·{" "}
                    {ctx.referral.hours_remaining < 0
                      ? `${Math.abs(ctx.referral.hours_remaining).toFixed(1)} h overdue`
                      : `${ctx.referral.hours_remaining.toFixed(1)} h left`}
                  </p>
                </div>
                <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                  <a
                    className="sheet-link"
                    href={`/v1/encounters/${ctx.encounter.id}/export`}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="Export printable clinical screening report with ophthalmologist sign-off box"
                  >
                    Export Report 📄
                  </a>
                  <Link className="sheet-link" to={`/app/result/${ctx.encounter.id}`}>Result sheet →</Link>
                </div>
                <div className="model-grade">
                  <dt>Model says</dt>
                  <dd>{ctx.referral.worst_grade ?? "—"}
                    <small>{ctx.referral.worst_grade != null ? GRADE[ctx.referral.worst_grade] : ""}</small></dd>
                </div>
              </div>

              <div className="eyes">
                {ctx.eyes.map((e) => <EyeCard key={e.image_id} eye={e} />)}
              </div>

              <div className="decide">
                <h3>Your call</h3>
                <div className="row">
                  <label>Reviewer</label>
                  <select value={reviewerId} onChange={(e) => setReviewerId(e.target.value)}>
                    {reviewers.map((u) => <option key={u.id} value={u.id}>{u.name}</option>)}
                    {reviewers.length === 0 && <option value="">no reviewer on file</option>}
                  </select>
                </div>

                <label>Notes</label>
                <textarea
                  rows={3}
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="what you saw, and why you kept or changed the grade"
                />

                <label>How useful was the explanation? (optional, for the reader study)</label>
                <div className="row" style={{ gap: "0.5rem", flexWrap: "wrap" }}>
                  <select value={rating.gradcam_useful ?? ""} aria-label="Grad-CAM usefulness"
                    onChange={(e) => setRating((r) => ({ ...r, gradcam_useful: e.target.value ? Number(e.target.value) : undefined }))}>
                    <option value="">Grad-CAM: not rated</option>
                    {["1 misleading", "2 unhelpful", "3 neutral", "4 helpful", "5 decisive"].map((t, i) =>
                      <option key={t} value={i + 1}>Grad-CAM: {t}</option>)}
                  </select>
                  <select value={rating.lesions_useful ?? ""} aria-label="Lesion evidence usefulness"
                    onChange={(e) => setRating((r) => ({ ...r, lesions_useful: e.target.value ? Number(e.target.value) : undefined }))}>
                    <option value="">Lesions: not rated</option>
                    {["1 misleading", "2 unhelpful", "3 neutral", "4 helpful", "5 decisive"].map((t, i) =>
                      <option key={t} value={i + 1}>Lesions: {t}</option>)}
                  </select>
                  <select value={rating.heatmap_on_lesions ?? ""} aria-label="Heatmap on the lesions that justify the grade"
                    onChange={(e) => setRating((r) => ({ ...r, heatmap_on_lesions: e.target.value || undefined }))}>
                    <option value="">Heatmap on the grading lesions?</option>
                    <option value="yes">Heatmap on the grading lesions: yes</option>
                    <option value="partly">Heatmap on the grading lesions: partly</option>
                    <option value="no">Heatmap on the grading lesions: no</option>
                  </select>
                </div>

                <div className="grades">
                  <label>Final grade if you disagree</label>
                  <div className="row">
                    {[0, 1, 2, 3, 4].map((g) => (
                      <button
                        key={g}
                        className={`grade-btn ${finalGrade === g ? "on" : ""}`}
                        onClick={() => setFinalGrade(g)}
                      >
                        {g}<span>{GRADE[g]}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="row actions">
                  <button onClick={() => submit(true)} disabled={busy !== null || !reviewerId || ctx.referral.state === "closed"}>
                    {busy ? "Saving…" : "Agree and close"}
                  </button>
                  <button
                    className="ghost"
                    onClick={() => submit(false)}
                    disabled={busy !== null || !reviewerId || finalGrade === null || ctx.referral.state === "closed"}
                  >
                    Change to grade {finalGrade ?? "—"} and close
                  </button>
                </div>
                {ctx.referral.state === "closed" && (
                  <p className="small dim">This one is already closed. Reopening is not something the API does yet.</p>
                )}
                <p className="small dim" style={{ marginBottom: 0 }}>
                  Both answers are written to the audit log with your id and the time you spent. The
                  disagreement rate is the number that tells us whether the model is drifting, so a
                  changed grade is more useful to us than a kept one.
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
