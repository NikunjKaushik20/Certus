/* The result sheet: one visit on one page, printable, for the patient to take away.

   It says screening result, not diagnosis, and it names the model and calibration that produced it.
   A sheet that cannot be traced back to a version is a sheet nobody can check later. */
import { useParams } from "react-router-dom";

import { useApi } from "../useApi";
import { useAuthedImage } from "../useAuthedImage";

type ImageRow = { id: string; eye: string; field: string; sha256: string; created_at: string };
type Quality = { label: string; gate_action: string };
type Finding = {
  id: string; lesion_type: string; lesion_count: number; threshold: number;
  macular_count: number | null;
};
type Structures = {
  disc_found: boolean; fovea_found: boolean; reason?: string;
  disc_diameter_px?: number; fovea_x?: number | null; fovea_y?: number | null;
};
type Prediction = {
  id: string; image_id: string; dr_grade: number; p_referable: number; decision: string;
  abstained: boolean; abstain_reason: string | null; model_version_id: string;
  calibration_version_id: string | null; findings: Finding[];
  trust_score: number | null; quality_factor: number | null;
  camera_factor: number | null; reliability: number | null;
  structures: Structures | null;
};
type Report = {
  encounter: { id: string; patient_id: string; site_id: string; created_at: string };
  images: ImageRow[];
  quality: Record<string, Quality | null>;
  predictions: Record<string, Prediction | null>;
  worst_grade: number | null;
  decision: string;
  referral: { id: string; priority: string; reason: string; sla_due_at: string } | null;
};
type Patient = { id: string; pseudo_id: string; sex: string; birth_year: number | null };
type Site = { id: string; name: string; district: string; state: string };

const GRADE = ["No retinopathy", "Mild", "Moderate", "Severe", "Proliferative"];
const ACTION = [
  "Rescreen in one year.",
  "Rescreen in one year.",
  "Refer to an ophthalmologist.",
  "Refer to an ophthalmologist, sooner.",
  "Refer urgently.",
];
const LESION = { MA: "Microaneurysms", HE: "Haemorrhages", EX: "Hard exudates", SE: "Soft exudates" } as const;

/* Confidence and trust are different numbers and the sheet has to show both, because a reader
   who only sees "99.8% referable" has no way to tell a clean photograph from a marginal one taken
   on a camera nobody has calibrated. Trust is reliability x decisiveness: what the conditions
   allow, times how far the score sits outside the abstention band. */
function TrustPanel({ p }: { p: Prediction }) {
  if (p.trust_score == null || p.reliability == null) return null;
  const pct = (x: number | null) => (x == null ? "—" : `${Math.round(x * 100)}%`);
  const gated = p.abstain_reason === "low_reliability";
  const band = p.trust_score >= 70 ? "hi" : p.trust_score >= 40 ? "mid" : "lo";
  return (
    <div className={`trust trust-${band}`}>
      <div className="trust-head">
        <dt>Trust</dt>
        <dd>{p.trust_score}<small>out of 100</small></dd>
      </div>
      <ul className="trust-factors">
        <li><span>Image quality</span><b>{pct(p.quality_factor)}</b></li>
        <li><span>Camera{p.camera_factor != null && p.camera_factor < 1 ? " (unverified)" : ""}</span>
            <b>{pct(p.camera_factor)}</b></li>
        <li className="trust-product"><span>Reliability</span><b>{pct(p.reliability)}</b></li>
      </ul>
      {gated && (
        <p className="trust-note">
          Below the 70% reliability floor, so this eye goes to an ophthalmologist whatever the
          model scored.
        </p>
      )}
    </div>
  );
}

function Sheet({ eye, image, quality, prediction }: {
  eye: string; image: ImageRow | undefined;
  quality: Quality | null | undefined; prediction: Prediction | null | undefined;
}) {
  const photo = useAuthedImage(image ? `/v1/images/${image.id}/file` : null);
  const macula = !!prediction?.structures?.fovea_found;
  return (
    <div className="eye-panel">
      <header>
        <b>{eye === "L" ? "Left eye" : "Right eye"}</b>
        {quality && <span className={`tag q-${quality.label}`}>{quality.label}</span>}
      </header>
      <div className="shot">
        {photo ? <img src={photo} alt={`${eye} fundus`} /> : <div className="empty">not available</div>}
      </div>
      {prediction ? (
        <>
          <div className="readout">
            <div><dt>Grade</dt><dd>{prediction.dr_grade}<small>{GRADE[prediction.dr_grade]}</small></dd></div>
            <div><dt>Referable</dt><dd>{(prediction.p_referable * 100).toFixed(1)}%</dd></div>
          </div>
          <TrustPanel p={prediction} />
          <table className="data compact">
            <thead>
              <tr>
                <th>Lesion</th>
                <th className="n">Regions</th>
                {macula && <th className="n" title="Within one disc diameter of the fovea">In macula</th>}
              </tr>
            </thead>
            <tbody>
              {prediction.findings.map((f) => (
                <tr key={f.id}>
                  <td>{LESION[f.lesion_type as keyof typeof LESION] ?? f.lesion_type}</td>
                  <td className="n">{f.lesion_count}</td>
                  {macula && (
                    <td className={`n${f.macular_count ? " hot" : ""}`}>{f.macular_count ?? 0}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          {/* The macula carries central vision, so the same haemorrhage matters far more here than
              at the periphery. When the disc cannot be found the column is withheld rather than
              filled with zeroes, which would read as "nothing central" instead of "not measured". */}
          <p className="small dim struct-note">
            {macula
              ? "Macular counts are lesions within one disc diameter of the estimated fovea."
              : `Fovea not located, so macular involvement was not assessed${
                  prediction.structures?.reason ? ` (${prediction.structures.reason})` : ""
                }.`}
          </p>
        </>
      ) : (
        <p className="dim small">This eye was not graded.</p>
      )}
    </div>
  );
}

export default function Result() {
  const { id } = useParams();
  const { data: report, error } = useApi<Report>(id ? `/v1/encounters/${id}/report` : null);
  const { data: patients } = useApi<Patient[]>("/v1/patients");
  const { data: sites } = useApi<Site[]>("/v1/sites");

  const patient = patients?.find((p) => p.id === report?.encounter.patient_id);
  const site = sites?.find((s) => s.id === report?.encounter.site_id);
  const worst = report?.worst_grade;
  const anyPred = report ? Object.values(report.predictions).find(Boolean) : null;
  const byEye = (eye: string) => report?.images.find((i) => i.eye === eye);

  return (
    <div className="page sheet">
      <header className="page-head no-print">
        <div>
          <div className="eyebrow">Result sheet</div>
          <h1>One visit, on one page.</h1>
          <p className="dim">
            What the patient takes away, and what a clinic can check the machine against later.
          </p>
        </div>
        <button onClick={() => window.print()}>Print</button>
      </header>

      {error && <div className="error no-print" role="alert">{error}</div>}
      {!report && !error && <p className="dim">Loading the visit…</p>}

      {report && (
        <article className="paper">
          <div className="paper-head">
            <div className="wordmark">Cert<span>us</span></div>
            <div className="paper-meta">
              Screening result · {new Date(report.encounter.created_at + "Z").toLocaleDateString()}
            </div>
          </div>

          <div className="grid">
            <div><dt>Screening id</dt><dd>{patient?.pseudo_id ?? "—"}</dd></div>
            <div><dt>Site</dt><dd style={{ fontSize: "0.95rem" }}>{site?.name ?? "—"}<small>{site?.district}</small></dd></div>
            <div><dt>Visit</dt><dd style={{ fontSize: "0.95rem" }}><code>{report.encounter.id.slice(0, 12)}</code></dd></div>
          </div>

          <div className="verdict">
            <div><dt>Worst grade</dt>
              <dd>{worst ?? "—"}<small>{worst != null ? GRADE[worst] : "not graded"}</small></dd></div>
            <div><dt>Outcome</dt><dd className={`d-${report.decision}`}>{report.decision}</dd></div>
            <div><dt>Next step</dt>
              <dd style={{ fontSize: "0.95rem" }}>{worst != null ? ACTION[worst] : "Repeat the photographs."}</dd></div>
          </div>

          {report.referral && (
            <div className="alert">
              <b>Referred, {report.referral.priority}.</b> Reason recorded as {report.referral.reason}.
              A reviewer is expected to close this by{" "}
              {new Date(report.referral.sla_due_at + "Z").toLocaleString()}.
            </div>
          )}

          <div className="eyes">
            {(["R", "L"] as const).map((eye) => {
              const img = byEye(eye);
              return (
                <Sheet
                  key={eye}
                  eye={eye}
                  image={img}
                  quality={img ? report.quality[img.id] : null}
                  prediction={img ? report.predictions[img.id] : null}
                />
              );
            })}
          </div>

          <footer className="paper-foot">
            <p className="small">
              This is a screening result, not a diagnosis. Grades are produced by an automated system
              and, where a case was referred, confirmed by a clinician before it was closed. Bring this
              sheet to your appointment.
            </p>
            <p className="small dim">
              Model <code>{anyPred?.model_version_id.slice(0, 12) ?? "—"}</code> · calibration{" "}
              <code>{anyPred?.calibration_version_id?.slice(0, 12) ?? "none fitted"}</code>
            </p>
          </footer>
        </article>
      )}
    </div>
  );
}
