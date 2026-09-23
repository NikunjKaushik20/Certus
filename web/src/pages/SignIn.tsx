/* Sign in.

   The backend maps an API key to a role, so this page does two things: take a key, and offer the
   three demo keys for people who do not have one. It checks the key against /v1/me before letting
   anyone through, because a sign-in screen that accepts anything and fails on the next page is worse
   than no sign-in screen. */
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { API_BASE, ApiError, api } from "../api";
import { ROLE_LABEL, type Role, setSession } from "../auth";
import "../index.css";
import "../app.css";

type Demo = { key: string; role: Role; does: string };

/* The keys in certus_api/config.py. The role and what it can do are what a visitor needs to choose;
   the key itself is machinery, so it stays out of the button. */
const DEMOS: Demo[] = [
  { key: "demo-tech", role: "technician", does: "Register patients, capture images, retake on a quality fail" },
  { key: "demo-doc", role: "ophthalmologist", does: "Work the review queue and adjudicate referred cases" },
  { key: "demo-admin", role: "admin", does: "Everything, plus cameras, model versions and the audit log" },
];

export default function SignIn() {
  const navigate = useNavigate();
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function enter(candidate: string) {
    const trimmed = candidate.trim();
    if (!trimmed) {
      setError("Enter a key, or use one of the demo accounts below.");
      return;
    }
    setBusy(trimmed);
    setError(null);
    try {
      const me = await api<{ role: Role; actor: string }>("/v1/me", { key: trimmed });
      setSession({ key: trimmed, role: me.role, actor: me.actor });
      navigate("/app", { replace: true });
    } catch (e) {
      const err = e as ApiError;
      setError(
        err.status === 401 ? "That key is not one the server knows."
          : err.unreachable ? `Cannot reach the API at ${API_BASE}. Start it and try again.`
          : err.message,
      );
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="auth">
      <section className="auth-side">
        <h1>Hello there.<br />Sign in to the screening console.</h1>
        <p className="dim">
          Certus grades a retinal photograph, shows the lesions behind the grade, and sends the
          doubtful cases to a person. What you can see here depends on which of those jobs you do.
        </p>
      </section>

      <section className="auth-main">
        <div className="card auth-card">
          <div className="eyebrow">Sign in</div>

          <form
            onSubmit={(e) => { e.preventDefault(); enter(key); }}
          >
            <label htmlFor="apikey">API key</label>
            <div className="field">
              <input
                id="apikey"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder="paste your key"
                autoComplete="off"
                spellCheck={false}
              />
              <button type="submit" disabled={busy !== null}>
                {busy === key.trim() ? "Checking…" : "Enter"}
              </button>
            </div>
          </form>

          {error && <div className="error" role="alert">{error}</div>}

          <div className="or"><span>or enter as a demo user</span></div>

          <div className="demos">
            {DEMOS.map((d) => (
              <button
                key={d.key}
                className="demo"
                onClick={() => enter(d.key)}
                disabled={busy !== null}
              >
                <b>{ROLE_LABEL[d.role]}</b>
                <span>{d.does}</span>
                {busy === d.key && <code>checking…</code>}
              </button>
            ))}
          </div>

          <p className="small dim" style={{ marginBottom: 0 }}>
            Demo keys are not authentication. They exist so the console can be shown without an
            identity provider behind it. The server still maps each key to a role, enforces what that
            role may do, and writes the actor into the audit log for every action.
          </p>
        </div>
      </section>
    </div>
  );
}
