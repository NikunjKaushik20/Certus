/* Cameras and trust.

   A confidence correction fitted on one set of cameras does not transfer to another, so the useful
   question is not "how accurate is the model" but "has this camera been calibrated". Any device whose
   domain key is missing from the calibration is shown as unverified rather than quietly trusted. */
import { useState } from "react";

import { api } from "../api";
import { useApi } from "../useApi";

type Site = { id: string; name: string; district: string; state: string; kind: string };
type Device = { id: string; site_id: string; make: string; model: string; serial: string; domain_key: string };
type Trust = {
  fitted: boolean; source: string; temperature: number; referral_threshold: number;
  lesion_thresholds: Record<string, number>;
  domains: Record<string, { n?: number; temperature?: number; threshold?: number; ece?: number }>;
};

const LESION = { MA: "Microaneurysms", HE: "Haemorrhages", EX: "Hard exudates", SE: "Soft exudates" } as const;

export default function Cameras() {
  const { data: sites, error: siteErr, reload: reloadSites } = useApi<Site[]>("/v1/sites");
  const { data: devices, error: devErr, reload: reloadDevices } = useApi<Device[]>("/v1/devices");
  const { data: trust, error: trustErr, loading, reload: loadTrust } = useApi<Trust>("/v1/trust", false);

  const [site, setSite] = useState({ name: "", district: "", state: "", kind: "camp" });
  const [device, setDevice] = useState({ site_id: "", make: "", model: "", serial: "", domain_key: "" });
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function add(kind: "site" | "device") {
    setBusy(kind);
    setError(null);
    try {
      if (kind === "site") {
        await api("/v1/sites", { method: "POST", body: site });
        setSite({ name: "", district: "", state: "", kind: "camp" });
        reloadSites();
      } else {
        await api("/v1/devices", { method: "POST", body: { ...device, site_id: device.site_id || sites?.[0]?.id } });
        setDevice({ site_id: "", make: "", model: "", serial: "", domain_key: "" });
        reloadDevices();
      }
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  function fillDemo() {
    setSite({ name: "Barabanki camp", district: "Barabanki", state: "Uttar Pradesh", kind: "camp" });
    // A Topcon, not another Remidio: the seeded cameras are Remidios on a domain key the
    // calibration has never seen, so adding one more only ever demonstrates "unverified".
    // Topcon_TRC_NW6 is the Messidor-2 camera, measured on 1,740 held-out eyes, so this one
    // comes back calibrated and the two states can be shown side by side. The old default also
    // reused RM-10-4471, a serial the Sitapur camera already has.
    setDevice({ site_id: sites?.[0]?.id ?? "", make: "Topcon", model: "TRC-NW6",
                serial: `TP-NW6-${String(1 + (devices?.length ?? 0)).padStart(4, "0")}`,
                domain_key: "Topcon_TRC_NW6" });
  }

  const domains = Object.entries(trust?.domains ?? {});

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Cameras</div>
          <h1>Which cameras this model has actually been calibrated on.</h1>
          <p className="dim">
            Every device carries a domain key. If the calibration has a correction fitted for that key,
            the camera is trusted for confidence; if it does not, the console says so instead of
            pretending the number means the same thing.
          </p>
        </div>
        <button className="demo-fill" onClick={fillDemo} type="button">Use demo data</button>
      </header>

      {(error || siteErr || devErr || trustErr) && (
        <div className="error" role="alert">{error ?? siteErr ?? devErr ?? trustErr}</div>
      )}

      <section className="panel wide">
        <div className="panel-head">
          <h2>Registered cameras</h2>
          <button className="ghost" onClick={loadTrust} disabled={loading}>
            {loading ? "Loading…" : trust ? "Reload calibration" : "Load calibration"}
          </button>
        </div>
        <p className="dim small">
          Loading the calibration pulls the checkpoint onto the server's device. While a training run
          has the GPU, that is worth doing on purpose rather than as a side effect of opening a page.
        </p>
        <table className="data">
          <thead>
            <tr><th>Camera</th><th>Site</th><th>Domain</th><th className="n">Confidence</th></tr>
          </thead>
          <tbody>
            {(devices ?? []).map((d) => {
              const known = trust?.domains?.[d.domain_key];
              return (
                <tr key={d.id}>
                  <td>{d.make} {d.model}<br /><span className="dim small">{d.serial || "no serial"}</span></td>
                  <td>{sites?.find((s) => s.id === d.site_id)?.name ?? "—"}</td>
                  <td><code>{d.domain_key}</code></td>
                  <td className="n">
                    {!trust ? "—" : known
                      ? <span className="tag ok">calibrated{known.n ? ` · n=${known.n}` : ""}</span>
                      : <span className="tag warn">unverified</span>}
                  </td>
                </tr>
              );
            })}
            {devices?.length === 0 && <tr><td colSpan={4} className="dim">No cameras registered.</td></tr>}
          </tbody>
        </table>
      </section>

      {trust && (
        <section className="panel wide">
          <h2>Calibration in force</h2>
          <div className="tiles small-tiles">
            <div className="tile"><dt>Fitted</dt>
              <dd>{trust.fitted ? "yes" : "no"}<small>{trust.source}</small></dd></div>
            <div className="tile"><dt>Temperature</dt><dd>{trust.temperature.toFixed(3)}</dd></div>
            <div className="tile"><dt>Referral threshold</dt><dd>{trust.referral_threshold.toFixed(3)}</dd></div>
          </div>
          <div className="cols">
            <div className="panel">
              <h3>Lesion thresholds</h3>
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
              <p className="small dim" style={{ marginBottom: 0 }}>
                One cut-off cannot serve every lesion. Microaneurysms are faint and tiny; exudates are
                bright and large.
              </p>
            </div>
            <div className="panel">
              <h3>Per-domain corrections</h3>
              {domains.length ? (
                <table className="data compact">
                  <thead><tr><th>Domain</th><th className="n">n</th><th className="n">Temp</th><th className="n">Threshold</th><th className="n">ECE</th></tr></thead>
                  <tbody>
                    {domains.map(([k, v]) => (
                      <tr key={k}>
                        <td><code>{k}</code></td>
                        <td className="n">{v.n ?? "—"}</td>
                        <td className="n">{v.temperature != null ? v.temperature.toFixed(3) : "—"}</td>
                        <td className="n">{v.threshold != null ? v.threshold.toFixed(3) : "—"}</td>
                        <td className="n">{v.ece != null ? v.ece.toFixed(4) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="dim small">
                  No per-domain corrections in this calibration. Every camera is therefore unverified,
                  and the confidence shown on a report is the global one.
                </p>
              )}
            </div>
          </div>
        </section>
      )}

      <div className="cols">
        <section className="panel">
          <h2>Add a site</h2>
          <label>Name</label>
          <input value={site.name} onChange={(e) => setSite({ ...site, name: e.target.value })} placeholder="camp or clinic name" />
          <div className="row">
            <input value={site.district} onChange={(e) => setSite({ ...site, district: e.target.value })} placeholder="district" />
            <input value={site.state} onChange={(e) => setSite({ ...site, state: e.target.value })} placeholder="state" />
            <select value={site.kind} onChange={(e) => setSite({ ...site, kind: e.target.value })}>
              <option value="camp">camp</option>
              <option value="phc">primary health centre</option>
              <option value="hospital">hospital</option>
            </select>
          </div>
          <button onClick={() => add("site")} disabled={!site.name.trim() || busy !== null}>
            {busy === "site" ? "Saving…" : "Add site"}
          </button>
        </section>

        <section className="panel">
          <h2>Add a camera</h2>
          <label>Site</label>
          <select value={device.site_id} onChange={(e) => setDevice({ ...device, site_id: e.target.value })}>
            {(sites ?? []).map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
          <div className="row">
            <input value={device.make} onChange={(e) => setDevice({ ...device, make: e.target.value })} placeholder="make" />
            <input value={device.model} onChange={(e) => setDevice({ ...device, model: e.target.value })} placeholder="model" />
          </div>
          <div className="row">
            <input value={device.serial} onChange={(e) => setDevice({ ...device, serial: e.target.value })} placeholder="serial" />
            <input value={device.domain_key} onChange={(e) => setDevice({ ...device, domain_key: e.target.value })} placeholder="domain key" />
          </div>
          <button onClick={() => add("device")} disabled={!device.make.trim() || busy !== null}>
            {busy === "device" ? "Saving…" : "Add camera"}
          </button>
          <p className="small dim" style={{ marginBottom: 0 }}>
            The domain key groups cameras that behave alike. Give two units of the same model the same
            key and they share a correction; give a new model its own and it stays unverified until
            someone fits one.
          </p>
        </section>
      </div>
    </div>
  );
}
