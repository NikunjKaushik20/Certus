/* Status: is the service actually working right now.

   Built after a demo where two images sat unqueued for four minutes and the console had no way to
   say so. The queue depth is the number that matters: jobs stuck in `queued` mean the worker is not
   draining them, which looks exactly like a slow model from the outside. */
import { useEffect, useState } from "react";

import { API_BASE, ping } from "../api";
import { useApi } from "../useApi";

type Stats = {
  encounters: number; images: number; predictions: number; referrals: number; audit_entries: number;
  jobs: Record<string, number>; median_inference_ms: number | null;
};
type QueueStats = { open_past_sla: number; closed: number; median_turnaround_hours: number | null };

export default function Status() {
  const [alive, setAlive] = useState<"checking" | "up" | "down">("checking");
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const { data: s, reload: reloadStats } = useApi<Stats>("/v1/stats");
  const { data: q, reload: reloadQueue } = useApi<QueueStats>("/v1/referrals/stats");

  async function check() {
    setAlive("checking");
    try {
      await ping();
      setAlive("up");
    } catch {
      setAlive("down");
    }
    setCheckedAt(new Date());
    reloadStats();
    reloadQueue();
  }

  useEffect(() => {
    check();
    const t = window.setInterval(check, 15000);
    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const jobs = s?.jobs ?? {};
  const waiting = (jobs.queued ?? 0) + (jobs.running ?? 0);
  const failed = jobs.failed ?? 0;

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Status</div>
          <h1>Is anything stuck.</h1>
          <p className="dim">
            Refreshes every fifteen seconds. If work is waiting and nothing is running, the grader is
            not picking jobs up, and no amount of waiting will change that.
          </p>
        </div>
        <button className="ghost" onClick={check}>Check now</button>
      </header>

      <div className={`status ${alive}`}>
        <i />
        <div>
          <b>{alive === "checking" ? "Checking" : alive === "up" ? "Service reachable" : "Service unreachable"}</b>
          <code>{API_BASE}{checkedAt ? ` · ${checkedAt.toLocaleTimeString()}` : ""}</code>
        </div>
      </div>

      <div className="tiles">
        <div className="tile"><dt>Waiting to grade</dt>
          <dd className={waiting ? "bad" : ""}>{waiting}<small>{jobs.running ?? 0} running</small></dd></div>
        <div className="tile"><dt>Graded</dt><dd>{jobs.done ?? 0}</dd></div>
        <div className="tile"><dt>Failed</dt>
          <dd className={failed ? "bad" : ""}>{failed}</dd></div>
        <div className="tile"><dt>Median inference</dt>
          <dd>{s?.median_inference_ms != null ? `${s.median_inference_ms} ms` : "—"}</dd></div>
        <div className="tile"><dt>Cases past the deadline</dt>
          <dd className={q?.open_past_sla ? "bad" : ""}>{q?.open_past_sla ?? "—"}</dd></div>
        <div className="tile"><dt>Audit entries</dt><dd>{s?.audit_entries ?? "—"}</dd></div>
      </div>

      {waiting > 0 && (jobs.running ?? 0) === 0 && (
        <div className="alert">
          <b>{waiting} image{waiting === 1 ? "" : "s"} queued with nothing running.</b> The background
          worker is not draining the queue. Either it was disabled at startup, or it stopped. Grading
          a visit by hand from the capture page will still work.
        </div>
      )}

      {failed > 0 && (
        <div className="alert">
          <b>{failed} job{failed === 1 ? "" : "s"} failed.</b> The error is recorded against the job;
          the encounter will stay unsettled until it is retried.
        </div>
      )}

      <section className="panel wide">
        <h2>Stored</h2>
        <table className="data compact">
          <tbody>
            <tr><td>Visits</td><td className="n">{s?.encounters ?? "—"}</td></tr>
            <tr><td>Images</td><td className="n">{s?.images ?? "—"}</td></tr>
            <tr><td>Eyes graded</td><td className="n">{s?.predictions ?? "—"}</td></tr>
            <tr><td>Referrals</td><td className="n">{s?.referrals ?? "—"}</td></tr>
            <tr><td>Referrals closed</td><td className="n">{q?.closed ?? "—"}</td></tr>
          </tbody>
        </table>
      </section>

      <div className="note">
        <b>What this page does not cover.</b> Capturing while offline and syncing later is not built.
        Everything above assumes the console can reach the server; there is no local queue behind it,
        and saying otherwise on a status page would defeat the point of having one.
      </div>
    </div>
  );
}
