/* Who is signed in.

   The backend's auth is a demo API key mapped to a role, so this is deliberately thin: hold the key,
   remember the role the server reported, and let components read both. The role is never trusted for
   anything that matters -- it only decides what the UI offers. Every request is authorised again on
   the server, which is the only place that counts. */
import { useSyncExternalStore } from "react";

export type Role = "technician" | "ophthalmologist" | "admin";

export type Session = {
  key: string;
  role: Role;
  actor: string;
};

const STORE_KEY = "certus.session";

function load(): Session | null {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;                       // private window, blocked storage, or corrupt value
  }
}

let session: Session | null = load();
const subscribers = new Set<() => void>();

function emit() {
  subscribers.forEach((fn) => fn());
}

export function getSession() {
  return session;
}

export function setSession(next: Session | null) {
  session = next;
  try {
    if (next) localStorage.setItem(STORE_KEY, JSON.stringify(next));
    else localStorage.removeItem(STORE_KEY);
  } catch { /* storage unavailable; the session still works for this tab */ }
  emit();
}

export function signOut() {
  setSession(null);
}

export function useSession() {
  return useSyncExternalStore(
    (fn) => {
      subscribers.add(fn);
      return () => subscribers.delete(fn);
    },
    getSession,
    () => null,
  );
}

export const ROLE_LABEL: Record<Role, string> = {
  technician: "Technician",
  ophthalmologist: "Ophthalmologist",
  admin: "Administrator",
};
