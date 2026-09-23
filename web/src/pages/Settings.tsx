/* Settings.

   Almost everything here is read-only on purpose. Thresholds and referral rules come from a fitted
   calibration, and a screen that let someone nudge them by hand would quietly break the promise that
   a report can be reproduced from its calibration id. */
import { useState } from "react";

import { API_BASE } from "../api";
import { ROLE_LABEL, type Session, signOut, useSession } from "../auth";
import { useApi } from "../useApi";

type Trust = {
  fitted: boolean; source: string; temperature: number; referral_threshold: number;
  lesion_thresholds: Record<string, number>;
};
type Engine = { device?: string; checkpoint?: string; step?: number };

const LESION = { MA: "Microaneurysms", HE: "Haemorrhages", EX: "Hard exudates", SE: "Soft exudates" } as const;

export default function Settings() {
  const session = useSession() as Session;
  const { data: trust, loading, reload } = useApi<Trust>("/v1/trust", false);
  const { data: engine, reload: reloadEngine } = useApi<Engine>("/v1/model", false);
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Settings</div>
          <h1>What this console is pointed at.</h1>
          <p className="dim">
            Most of it cannot be edited here, and that is deliberate. Thresholds belong to a fitted
            calibration; changing them by hand would make old reports impossible to reproduce.
          </p>
        </div>
      </header>

      <div className="cols">
        <section className="panel">
          <h2>This session</h2>
          <table className="data compact">
            <tbody>
              <tr><td>Role</td><td>{ROLE_LABEL[session.role]}</td></tr>
              <tr><td>Actor recorded in the log</td><td><code>{session.actor}</code></td></tr>
              <tr><td>Server</td><td><code>{API_BASE}</code></td></tr>
            </tbody>
          </table>
          <div className="row">
            <button
              className="ghost"
              onClick={async () => {
                // writeText rejects when the document is not focused, when the page is not a
                // secure context, or when the permission is refused. Setting "Copied" without
                // waiting for it reported success on every one of those, which is worse than
                // no button: the address the user pastes is then whatever was already there.
                try {
                  await navigator.clipboard.writeText(API_BASE);
                  setCopied(true);
                } catch {
                  setCopied(false);
                  setCopyFailed(true);
                }
              }}
            >
              {copied ? "Copied" : copyFailed ? "Copy it by hand" : "Copy server address"}
            </button>
            <button className="ghost" onClick={signOut}>Sign out</button>
          </div>
          <p className="small dim" style={{ marginBottom: 0 }}>
            The server address is fixed at build time by <code>VITE_API_BASE</code>. Pointing the
            console somewhere else is a deployment decision, not a per-user preference.
          </p>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Runtime</h2>
            <button className="ghost" onClick={() => { reloadEngine(); }} disabled={loading}>
              {engine ? "Reload" : "Load"}
            </button>
          </div>
          {engine ? (
            <table className="data compact">
              <tbody>
                <tr><td>Device</td><td><code>{engine.device ?? "—"}</code></td></tr>
                <tr><td>Checkpoint</td><td><code>{engine.checkpoint ?? "—"}</code></td></tr>
                <tr><td>Training step</td><td className="n">{engine.step ?? "—"}</td></tr>
              </tbody>
            </table>
          ) : (
            <p className="dim small">
              Not loaded. Asking pulls the checkpoint onto the server's device, which is worth doing
              on purpose while a training run has the GPU.
            </p>
          )}
        </section>
      </div>

      <section className="panel wide">
        <div className="panel-head">
          <h2>Decision rules in force</h2>
          <button className="ghost" onClick={reload} disabled={loading}>
            {loading ? "Loading…" : trust ? "Reload" : "Load calibration"}
          </button>
        </div>

        {!trust && <p className="dim small">Load the calibration to see the thresholds a report would be produced under.</p>}

        {trust && (
          <>
            {!trust.fitted && (
              <div className="alert">
                <b>No calibration has been fitted.</b> The thresholds below come from the checkpoint's
                own validation sweep. They are reasonable, but they are not a calibration, and the
                console will keep saying so until <code>calibrate.py</code> has run.
              </div>
            )}
            <div className="tiles small-tiles">
              <div className="tile"><dt>Source</dt><dd style={{ fontSize: "0.9rem" }}>{trust.source}</dd></div>
              <div className="tile"><dt>Temperature</dt><dd>{trust.temperature.toFixed(3)}</dd></div>
              <div className="tile"><dt>Referral threshold</dt><dd>{trust.referral_threshold.toFixed(3)}</dd></div>
            </div>
            <table className="data compact">
              <thead><tr><th>Lesion</th><th className="n">Threshold</th></tr></thead>
              <tbody>
                {Object.entries(trust.lesion_thresholds ?? {}).map(([k, v]) => (
                  <tr key={k}>
                    <td>{LESION[k as keyof typeof LESION] ?? k}</td>
                    <td className="n">{Number(v).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>

      <section className="panel wide">
        <h2>Fixed by the service</h2>
        <table className="data compact">
          <tbody>
            <tr><td>Referral deadline</td><td className="n">48 hours</td></tr>
            <tr><td>Grades treated as urgent</td><td className="n">3 and 4</td></tr>
            <tr><td>Referable threshold</td><td className="n">grade 2 or worse</td></tr>
            <tr><td>Tiles read per eye</td><td className="n">3 × 3, plus one whole-eye view</td></tr>
          </tbody>
        </table>
        <p className="small dim" style={{ marginBottom: 0 }}>
          These come from the server's configuration. Changing them changes what past reports would
          have said, so they move with a deployment, not with a click.
        </p>
      </section>
    </div>
  );
}
