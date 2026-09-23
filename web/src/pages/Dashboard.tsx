/* Dashboard: the shape of the workload, and what the model did with it.

   Six counts, then three breakdowns. Grades and decisions are separate questions: a grade says what
   the model saw, a decision says what it was willing to act on, and the gap between them is the
   abstention rate that any honest accuracy figure has to be read beside. */
import { useApi } from "../useApi";

type Stats = {
  encounters: number; images: number; predictions: number; referrals: number; audit_entries: number;
  jobs: Record<string, number>; grades: Record<string, number>; decisions: Record<string, number>;
  median_inference_ms: number | null;
};
type QueueStats = {
  open_past_sla: number; closed: number;
  median_turnaround_hours: number | null; within_48h_fraction: number | null;
};

const GRADE = ["No retinopathy", "Mild", "Moderate", "Severe", "Proliferative"];

export default function Dashboard() {
  const { data: s, error } = useApi<Stats>("/v1/stats");
  const { data: q } = useApi<QueueStats>("/v1/referrals/stats");

  const graded = s ? Object.values(s.grades).reduce((a, b) => a + b, 0) : 0;
  const abstained = s?.decisions["refer-to-human"] ?? 0;

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Dashboard</div>
          <h1>What the service has done so far.</h1>
          <p className="dim">
            Counts come straight from the database, not from a cache. An empty table means nothing has
            happened yet rather than nothing was recorded.
          </p>
        </div>
      </header>

      {error && <div className="error" role="alert">{error}</div>}

      <div className="tiles">
        <div className="tile"><dt>Visits</dt><dd>{s?.encounters ?? "—"}</dd></div>
        <div className="tile"><dt>Images</dt><dd>{s?.images ?? "—"}</dd></div>
        <div className="tile"><dt>Eyes graded</dt><dd>{s?.predictions ?? "—"}</dd></div>
        <div className="tile"><dt>Referrals</dt><dd>{s?.referrals ?? "—"}</dd></div>
        <div className="tile"><dt>Median inference</dt>
          <dd>{s?.median_inference_ms != null ? `${s.median_inference_ms} ms` : "—"}</dd></div>
        <div className="tile"><dt>Audit entries</dt><dd>{s?.audit_entries ?? "—"}</dd></div>
      </div>

      <div className="tiles">
        <div className="tile"><dt>Sent to a human</dt>
          <dd className="teal">{graded ? `${Math.round((abstained / graded) * 100)}%` : "—"}
            <small>{abstained} of {graded} eyes</small></dd></div>
        <div className="tile"><dt>Past the 48-hour mark</dt>
          <dd className={q?.open_past_sla ? "bad" : ""}>{q?.open_past_sla ?? "—"}</dd></div>
        <div className="tile"><dt>Median turnaround</dt>
          <dd>{q?.median_turnaround_hours != null ? `${q.median_turnaround_hours} h` : "—"}</dd></div>
        <div className="tile"><dt>Closed within 48 h</dt>
          <dd>{q?.within_48h_fraction != null ? `${Math.round(q.within_48h_fraction * 100)}%` : "—"}</dd></div>
      </div>

      <div className="cols">
        <section className="panel">
          <h2>Grades given</h2>
          {graded > 0 ? (
            <table className="data compact">
              <thead><tr><th>Grade</th><th>Meaning</th><th className="n">Eyes</th></tr></thead>
              <tbody>
                {[0, 1, 2, 3, 4].map((g) => (
                  <tr key={g} className={g >= 2 ? "hi" : ""}>
                    <td className="n">{g}</td><td>{GRADE[g]}</td>
                    <td className="n">{s?.grades[String(g)] ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="dim small">Nothing graded yet.</p>}
          <p className="small dim" style={{ marginBottom: 0 }}>
            Shaded rows are referable: grade 2 or worse.
          </p>
        </section>

        <section className="panel">
          <h2>What it decided</h2>
          {s && Object.keys(s.decisions).length > 0 ? (
            <table className="data compact">
              <thead><tr><th>Decision</th><th className="n">Eyes</th></tr></thead>
              <tbody>
                {Object.entries(s.decisions).map(([d, n]) => (
                  <tr key={d}><td className={`d-${d}`}>{d}</td><td className="n">{n}</td></tr>
                ))}
              </tbody>
            </table>
          ) : <p className="dim small">Nothing decided yet.</p>}
          <p className="small dim" style={{ marginBottom: 0 }}>
            <b>refer-to-human</b> is the model declining to call an eye. It is an outcome with a
            reason attached, never a missing value.
          </p>
        </section>

        <section className="panel">
          <h2>Inference jobs</h2>
          {s && Object.keys(s.jobs).length > 0 ? (
            <table className="data compact">
              <thead><tr><th>State</th><th className="n">Jobs</th></tr></thead>
              <tbody>
                {Object.entries(s.jobs).map(([k, n]) => (
                  <tr key={k} className={k === "failed" ? "hi" : ""}><td>{k}</td><td className="n">{n}</td></tr>
                ))}
              </tbody>
            </table>
          ) : <p className="dim small">No jobs have run.</p>}
          <p className="small dim" style={{ marginBottom: 0 }}>
            Anything sitting in <b>queued</b> means the worker is not draining the queue. Status has
            the detail.
          </p>
        </section>
      </div>
    </div>
  );
}
