/* Audit log.

   Every row carries a hash of its payload chained to the row before it, so a deleted or edited entry
   breaks the chain at a known sequence number. The verify button re-walks it on the server rather
   than showing a tick that means nothing. */
import { Fragment, useState } from "react";

import { api } from "../api";
import { useApi } from "../useApi";

type Row = {
  seq: number; at: string; actor: string; action: string; entity: string;
  payload: Record<string, unknown>; hash: string;
};
type Verify = { ok: boolean; checked?: number; broken_at?: number | null; [k: string]: unknown };

function when(at: string) {
  return new Date(at + (at.endsWith("Z") ? "" : "Z")).toLocaleString();
}

export default function Audit() {
  const [limit, setLimit] = useState(50);
  const { data: rows, error, reload } = useApi<Row[]>(`/v1/audit?limit=${limit}`);
  const [verify, setVerify] = useState<Verify | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState("");

  async function runVerify() {
    setBusy(true);
    try {
      setVerify(await api<Verify>("/v1/audit/verify"));
    } catch (e) {
      setVerify({ ok: false, error: String((e as Error).message) });
    } finally {
      setBusy(false);
    }
  }

  const shown = (rows ?? []).filter(
    (r) => !filter || r.action.includes(filter) || r.actor.includes(filter) || r.entity.includes(filter),
  );
  const actions = [...new Set((rows ?? []).map((r) => r.action))].sort();

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Audit</div>
          <h1>Everything that happened, in the order it happened.</h1>
          <p className="dim">
            Patients registered, images uploaded, eyes graded, cases reviewed. Each entry names the
            actor, and the chain makes a quiet edit detectable rather than merely discouraged.
          </p>
        </div>
        <button className="ghost" onClick={runVerify} disabled={busy}>
          {busy ? "Walking the chain…" : "Verify the chain"}
        </button>
      </header>

      {error && <div className="error" role="alert">{error}</div>}

      {verify && (
        <div className={`alert ${verify.ok ? "soft" : ""}`}>
          <b>{verify.ok ? "Chain intact." : "Chain broken."}</b>{" "}
          {verify.ok
            ? `Every entry hashes to the one before it${verify.checked ? `, ${verify.checked} checked` : ""}.`
            : verify.broken_at != null
              ? `First mismatch at sequence ${verify.broken_at}.`
              : String(verify.error ?? "")}
        </div>
      )}

      <div className="row">
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="">every action</option>
          {actions.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
        <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
          <option value={50}>last 50</option>
          <option value={200}>last 200</option>
          <option value={1000}>last 1000</option>
        </select>
        <button className="ghost" onClick={reload}>Refresh</button>
        <span className="dim small">{shown.length} shown</span>
      </div>

      <section className="panel wide">
        <table className="data compact">
          <thead>
            <tr><th className="n">#</th><th>When</th><th>Actor</th><th>Action</th><th>Entity</th><th>Hash</th><th /></tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <Fragment key={r.seq}>
                <tr>
                  <td className="n">{r.seq}</td>
                  <td>{when(r.at)}</td>
                  <td><code>{r.actor}</code></td>
                  <td>{r.action}</td>
                  <td><code>{r.entity.slice(0, 28)}</code></td>
                  <td><code>{r.hash}</code></td>
                  <td className="n">
                    <button className="link" onClick={() => setOpen(open === r.seq ? null : r.seq)}>
                      {open === r.seq ? "hide" : "payload"}
                    </button>
                  </td>
                </tr>
                {open === r.seq && (
                  <tr>
                    <td colSpan={7}>
                      <pre className="payload">{JSON.stringify(r.payload, null, 2)}</pre>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {shown.length === 0 && <tr><td colSpan={7} className="dim">Nothing logged yet.</td></tr>}
          </tbody>
        </table>
      </section>
    </div>
  );
}
