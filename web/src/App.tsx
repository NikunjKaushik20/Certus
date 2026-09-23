/* Routes. The landing page is public; everything under /app needs a session, and which tab you land
   on depends on the role the server reported. */
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import type { ReactElement } from "react";

import { useSession } from "./auth";
import Audit from "./pages/Audit";
import Cameras from "./pages/Cameras";
import Console, { RoleHome } from "./pages/Console";
import Dashboard from "./pages/Dashboard";
import Landing from "./pages/Landing";
import ModelPage from "./pages/ModelPage";
import People from "./pages/People";
import Result from "./pages/Result";
import Review from "./pages/Review";
import Settings from "./pages/Settings";
import SignIn from "./pages/SignIn";
import Status from "./pages/Status";
import Technician from "./pages/Technician";

function RequireSession({ children }: { children: ReactElement }) {
  const session = useSession();
  return session ? children : <Navigate to="/signin" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/app" element={<RequireSession><Console /></RequireSession>}>
          <Route index element={<RoleHome />} />
          <Route path="capture" element={<Technician />} />
          <Route path="review" element={<Review />} />
          <Route path="result/:id" element={<Result />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="cameras" element={<Cameras />} />
          <Route path="model" element={<ModelPage />} />
          <Route path="audit" element={<Audit />} />
          <Route path="people" element={<People />} />
          <Route path="status" element={<Status />} />
          <Route path="settings" element={<Settings />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
