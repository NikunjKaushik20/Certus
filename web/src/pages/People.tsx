/* People.

   A reviewer has to exist here before a case can be assigned or signed off. That is the reason the
   audit log ends up with a person's id against every clinical decision instead of "system". */
import { useState } from "react";

import { api } from "../api";
import { useApi } from "../useApi";

type User = { id: string; name: string; role: string; site_id: string | null };
type Site = { id: string; name: string; district: string };

const BLURB: Record<string, string> = {
  technician: "Registers patients, captures images, retakes on a quality fail",
  ophthalmologist: "Works the review queue and signs off referred cases",
  admin: "Everything, plus cameras, model versions and the audit log",
};

export default function People() {
  const { data: users, error, reload } = useApi<User[]>("/v1/users");
  const { data: sites } = useApi<Site[]>("/v1/sites");
  const [draft, setDraft] = useState({ name: "", role: "ophthalmologist", site_id: "" });
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  async function add() {
    setBusy(true);
    setProblem(null);
    try {
      const q = new URLSearchParams({ name: draft.name.trim(), role: draft.role });
      if (draft.site_id) q.set("site_id", draft.site_id);
      await api(`/v1/users?${q}`, { method: "POST" });
      setDraft({ name: "", role: draft.role, site_id: draft.site_id });
      reload();
    } catch (e) {
      setProblem(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }

  const byRole = (role: string) => (users ?? []).filter((u) => u.role === role);

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">People</div>
          <h1>Who is allowed to do what.</h1>
          <p className="dim">
            Roles decide what the console offers, and the server checks them again on every request.
            Nothing here grants access on its own: a person still needs a key.
          </p>
        </div>
        <button
          className="demo-fill"
          type="button"
          onClick={() => setDraft({ name: "Dr A. Rao", role: "ophthalmologist", site_id: sites?.[0]?.id ?? "" })}
        >
          Use demo data
        </button>
      </header>

      {(error || problem) && <div className="error" role="alert">{error ?? problem}</div>}

      <div className="tiles">
        {["technician", "ophthalmologist", "admin"].map((r) => (
          <div className="tile" key={r}>
            <dt>{r}</dt>
            <dd>{byRole(r).length}<small>{BLURB[r]}</small></dd>
          </div>
        ))}
      </div>

      <section className="panel wide">
        <h2>On file</h2>
        <table className="data">
          <thead><tr><th>Name</th><th>Role</th><th>Site</th><th>Id</th></tr></thead>
          <tbody>
            {(users ?? []).map((u) => (
              <tr key={u.id}>
                <td>{u.name}</td>
                <td>{u.role}</td>
                <td>{sites?.find((s) => s.id === u.site_id)?.name ?? "—"}</td>
                <td><code>{u.id.slice(0, 12)}</code></td>
              </tr>
            ))}
            {users?.length === 0 && <tr><td colSpan={4} className="dim">Nobody on file yet.</td></tr>}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>Add someone</h2>
        <label>Name</label>
        <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="full name" />
        <div className="row">
          <select value={draft.role} onChange={(e) => setDraft({ ...draft, role: e.target.value })}>
            <option value="technician">technician</option>
            <option value="ophthalmologist">ophthalmologist</option>
            <option value="admin">admin</option>
          </select>
          <select value={draft.site_id} onChange={(e) => setDraft({ ...draft, site_id: e.target.value })}>
            <option value="">no fixed site</option>
            {(sites ?? []).map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
          <button onClick={add} disabled={!draft.name.trim() || busy}>{busy ? "Adding…" : "Add"}</button>
        </div>
        <p className="small dim" style={{ marginBottom: 0 }}>
          {BLURB[draft.role]}.
        </p>
      </section>
    </div>
  );
}
