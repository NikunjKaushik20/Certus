/* The signed-in shell: a floating glass bar, the nav for whatever role the server reported, and the
   page under it.

   Admins see every tab because the server lets an admin through every check. A technician sees the
   two they can reach, because a third would only lead to a 403. */
import { useEffect, useState } from "react";
import { Link, NavLink, Navigate, Outlet, useLocation } from "react-router-dom";

import { ROLE_LABEL, type Role, type Session, signOut, useSession } from "../auth";
import "../index.css";
import "../app.css";
import "../console.css";
import "../navbar.css";

type Tab = { to: string; label: string; roles: Role[] };

const TABS: Tab[] = [
  { to: "/app/capture", label: "Capture", roles: ["technician", "admin"] },
  { to: "/app/review", label: "Review", roles: ["ophthalmologist", "admin"] },
  { to: "/app/dashboard", label: "Dashboard", roles: ["admin"] },
  { to: "/app/cameras", label: "Cameras", roles: ["admin"] },
  { to: "/app/model", label: "Model", roles: ["admin"] },
  { to: "/app/audit", label: "Audit", roles: ["admin"] },
  { to: "/app/people", label: "People", roles: ["admin"] },
  { to: "/app/status", label: "Status", roles: ["technician", "ophthalmologist", "admin"] },
  { to: "/app/settings", label: "Settings", roles: ["technician", "ophthalmologist", "admin"] },
];

/** Where a role lands when it opens /app. */
export function RoleHome() {
  const session = useSession();
  if (!session) return <Navigate to="/signin" replace />;
  const first = TABS.find((t) => t.roles.includes(session.role));
  return <Navigate to={first ? first.to : "/app/status"} replace />;
}

export default function Console() {
  const session = useSession() as Session;
  const { pathname } = useLocation();
  const [lifted, setLifted] = useState(false);
  const tabs = TABS.filter((t) => t.roles.includes(session.role));

  // The bar is translucent; it needs a stronger edge once content is sliding underneath it.
  useEffect(() => {
    const onScroll = () => setLifted(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <>
      <header className={`navbar ${lifted ? "lifted" : ""}`}>
        <div className="nav-inner">
          <Link to="/" className="wordmark">Cert<span>us</span></Link>

          <nav className="nav-links">
            {tabs.map((t) => (
              <NavLink key={t.to} to={t.to} className={({ isActive }) => (isActive ? "on" : "")}>
                <span data-label={t.label}>{t.label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="who">
            <span>{ROLE_LABEL[session.role]}</span>
            <button onClick={signOut}>Sign out</button>
          </div>
        </div>
      </header>

      <main className="shell-body" key={pathname}>
        <Outlet />
      </main>
    </>
  );
}
