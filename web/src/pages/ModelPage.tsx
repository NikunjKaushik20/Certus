/* Model and calibration provenance.

   A report is only checkable if you can find out what produced it. Every prediction stores a model
   version and a calibration version, and both are rows here rather than a number in someone's notes.
   Refitting a calibration makes a new row; it never reaches back and changes what a clinician was
   told last month. */
import { useApi } from "../useApi";

type ModelVersion = {
  id: string; name: string; checkpoint_uri: string; step: number | null;
  metrics: Record<string, unknown>; input_spec: Record<string, unknown>; active: boolean;
};
type Calibration = {
  id: string; model_version_id: string; temperature: number; referral_threshold: number;
  lesion_thresholds: Record<string, number>; conformal_q: Record<string, number>;
  fitted_on: string; metrics: Record<string, number>; active: boolean;
};
type Engine = {
  device?: string; checkpoint?: string; step?: number;
  metrics?: Record<string, unknown>; calibration?: Record<string, unknown>;
  [k: string]: unknown;
};

type Split = { n?: number; auc_referable?: number; sens_at_85spec?: number; qwk?: number; accuracy?: number; ece_referable?: number };
type HeldOut = {
  raw: Split; calibrated: Split;
  band: { n?: number; abstain_rate?: number; system_sensitivity?: number; missed_referable?: number; auto_specificity?: number };
  two_readers?: { system_sensitivity?: number; system_specificity?: number; to_human?: number;
                  missed_referable?: number; false_auto_refer?: number };
};
const SPLIT_NAME: Record<string, string> = { test: "Test", external_messidor2: "Messidor-2 (unseen camera)" };

function pct(v: unknown) {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
}
function num(v: unknown, dp = 4) {
  return typeof v === "number" ? v.toFixed(dp) : "—";
}

export default function ModelPage() {
  const { data: models, error: mErr } = useApi<ModelVersion[]>("/v1/model/versions");
  const { data: cals, error: cErr } = useApi<Calibration[]>("/v1/calibrations");
  const { data: engine, error: eErr, loading, reload } = useApi<Engine>("/v1/model", false);
  const { data: trust } = useApi<{ temperature: number; held_out?: Record<string, HeldOut> }>("/v1/trust");
  const held = trust?.held_out ?? {};

  const active = models?.find((m) => m.active) ?? models?.[0];
  const g = (active?.metrics ?? {}) as Record<string, unknown>;
  const seg = (g.seg ?? {}) as Record<string, number>;

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Model</div>
          <h1>What produced the grade, and under which numbers.</h1>
          <p className="dim">
            These are the figures recorded when the checkpoint was validated, not figures typed in
            afterwards. Where a calibration has never been fitted, the page says so rather than
            showing a temperature of 1.0 as though it were a result.
          </p>
        </div>
        <button className="ghost" onClick={reload} disabled={loading}>
          {loading ? "Loading…" : engine ? "Reload runtime" : "Load runtime"}
        </button>
      </header>

      {(mErr || cErr || eErr) && <div className="error" role="alert">{mErr ?? cErr ?? eErr}</div>}

      {active && (
        <div className="tiles">
          <div className="tile"><dt>Referable AUC</dt><dd>{num(g.auc_referable)}</dd></div>
          <div className="tile"><dt>Sensitivity at 85% spec</dt><dd>{pct(g.sens_at_85spec)}</dd></div>
          <div className="tile"><dt>Quadratic kappa</dt><dd>{num(g.qwk)}</dd></div>
          <div className="tile"><dt>Calibration error</dt><dd>{num(g.ece_referable)}</dd></div>
          <div className="tile"><dt>Validation eyes (optimistic)</dt><dd>{typeof g.n === "number" ? g.n : "—"}</dd></div>
          <div className="tile"><dt>Step</dt><dd>{active.step ?? "—"}</dd></div>
        </div>
      )}

      {Object.keys(held).length > 0 && (
        <section className="panel wide">
          <h2>Held out, after calibration</h2>
          <table className="data compact">
            <thead>
              <tr><th>Split</th><th className="n">Eyes</th><th className="n">AUC</th><th className="n">Sens @ 85% spec</th><th className="n">Kappa</th>
                <th className="n">ECE raw → calibrated</th><th className="n">System sensitivity</th><th className="n">Abstained</th><th className="n">Missed referable</th><th className="n">Auto specificity</th></tr>
            </thead>
            <tbody>
              {Object.entries(held).map(([k, h]) => (
                <tr key={k}>
                  <td>{SPLIT_NAME[k] ?? k}</td>
                  <td className="n">{h.calibrated.n ?? "—"}</td>
                  <td className="n">{num(h.calibrated.auc_referable)}</td>
                  <td className="n">{pct(h.calibrated.sens_at_85spec)}</td>
                  <td className="n">{num(h.calibrated.qwk, 3)}</td>
                  <td className="n">{num(h.raw.ece_referable, 3)} → {num(h.calibrated.ece_referable, 3)}</td>
                  <td className="n">{pct(h.band.system_sensitivity)}</td>
                  <td className="n">{pct(h.band.abstain_rate)}</td>
                  <td className="n">{h.band.missed_referable ?? "—"}</td>
                  <td className="n">{pct(h.band.auto_specificity)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small dim" style={{ marginBottom: 0 }}>
            Temperature {num(trust?.temperature, 3)}, fitted on validation only. Scaling moves probabilities, not
            rankings, so AUC and kappa do not change. It lowered calibration error on test and raised it on
            Messidor-2: a temperature fitted on the training cameras does not transfer to an unseen one, which is
            why that camera is marked unverified and its eyes carry less trust.
          </p>
        </section>
      )}

      {Object.values(held).some((h) => h.two_readers && Object.keys(h.two_readers).length > 0) && (
        <section className="panel wide">
          <h2>Two readers, which is what decides</h2>
          <table className="data compact">
            <thead>
              <tr><th>Split</th><th className="n">System sensitivity</th><th className="n">System specificity</th>
                <th className="n">Sent to a person</th><th className="n">Referable auto-cleared</th><th className="n">Healthy auto-referred</th></tr>
            </thead>
            <tbody>
              {Object.entries(held).filter(([, h]) => h.two_readers).map(([k, h]) => (
                <tr key={k}>
                  <td>{SPLIT_NAME[k] ?? k}</td>
                  <td className="n">{pct(h.two_readers?.system_sensitivity)}</td>
                  <td className="n">{pct(h.two_readers?.system_specificity)}</td>
                  <td className="n">{pct(h.two_readers?.to_human)}</td>
                  <td className="n">{h.two_readers?.missed_referable ?? "—"}</td>
                  <td className="n">{h.two_readers?.false_auto_refer ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small dim" style={{ marginBottom: 0 }}>
            The table above is Certus on its own. In use, a second model grades every eye too: a plain
            whole-image grader with no lesion maps. An eye is cleared or referred automatically only when both
            are outside their bands on the same side. Anything else goes to an ophthalmologist, marked as a
            disagreement. System figures count those eyes as correctly handled, because a person reads them.
          </p>
        </section>
      )}

      {engine && (
        <section className="panel wide">
          <h2>Running right now</h2>
          <table className="data compact">
            <tbody>
              <tr><td>Device</td><td><code>{String(engine.device ?? "—")}</code></td></tr>
              <tr><td>Checkpoint</td><td><code>{String(engine.checkpoint ?? "—")}</code></td></tr>
              <tr><td>Step</td><td className="n">{String(engine.step ?? "—")}</td></tr>
            </tbody>
          </table>
        </section>
      )}

      <section className="panel wide">
        <h2>Model versions</h2>
        <table className="data">
          <thead><tr><th>Name</th><th>Checkpoint</th><th className="n">Step</th><th className="n">Active</th></tr></thead>
          <tbody>
            {(models ?? []).map((m) => (
              <tr key={m.id} className={m.active ? "hi" : ""}>
                <td>{m.name}</td>
                <td><code>{m.checkpoint_uri}</code></td>
                <td className="n">{m.step ?? "—"}</td>
                <td className="n">{m.active ? "yes" : "no"}</td>
              </tr>
            ))}
            {models?.length === 0 && <tr><td colSpan={4} className="dim">None recorded.</td></tr>}
          </tbody>
        </table>
      </section>

      {Object.keys(seg).length > 0 && (
        <section className="panel wide">
          <h2>Lesion segmentation on validation</h2>
          <table className="data compact">
            <thead>
              <tr><th>Lesion</th><th className="n">Dice at threshold</th><th className="n">Best Dice</th><th className="n">Best threshold</th><th className="n">Average precision</th></tr>
            </thead>
            <tbody>
              {["MA", "HE", "EX", "SE"].map((k) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td className="n">{num(seg[k], 4)}</td>
                  <td className="n">{num(seg[`${k}_best`], 4)}</td>
                  <td className="n">{num(seg[`${k}_thr`], 3)}</td>
                  <td className="n">{num(seg[`${k}_ap`], 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small dim" style={{ marginBottom: 0 }}>
            Dice measures area overlap, which is not the same as counting lesions correctly. Nothing
            here is evidence that the region counts on a report are accurate.
          </p>
        </section>
      )}

      <section className="panel wide">
        <h2>Calibration versions</h2>
        <table className="data">
          <thead>
            <tr><th>Fitted on</th><th className="n">Temperature</th><th className="n">Referral threshold</th><th>Lesion thresholds</th><th className="n">Active</th></tr>
          </thead>
          <tbody>
            {(cals ?? []).map((c) => (
              <tr key={c.id} className={c.active ? "hi" : ""}>
                <td>{c.fitted_on}</td>
                <td className="n">{c.temperature.toFixed(3)}</td>
                <td className="n">{c.referral_threshold.toFixed(3)}</td>
                <td><code>{Object.entries(c.lesion_thresholds ?? {}).map(([k, v]) => `${k} ${Number(v).toFixed(3)}`).join("  ")}</code></td>
                <td className="n">{c.active ? "yes" : "no"}</td>
              </tr>
            ))}
            {cals?.length === 0 && <tr><td colSpan={5} className="dim">None recorded.</td></tr>}
          </tbody>
        </table>
        <p className="small dim" style={{ marginBottom: 0 }}>
          Every prediction stores the calibration id it was produced under, so a report can always be
          re-read against the numbers that were in force when it was written.
        </p>
      </section>
    </div>
  );
}
